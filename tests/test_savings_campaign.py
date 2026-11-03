"""절감 campaign의 후보/품질예산/MAPE/split/실행계약 회귀 테스트."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from controllers.control_benchmark import CategoricalCEM, create_control_planner
from controllers.control_mape_audit import power_metric_rows
from controllers.energy_cem import EnergyCategoricalCEM, physical_changes
from data_provider.control_data import ControlEncoder, ControlVariable
from models.control import ContextSpec, create_control_model, load_control_model
from models.control.neural import contiguous_rollout_starts
from runners.full_control_benchmark_runner import scope_audit
from test_control_benchmark import data_fixture, CONSTRAINTS, OPTIONS, PARAMETERS
from test_full_control_benchmark import small_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'savings_campaign_test_module', ROOT/'src/scripts/control/savings_campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class ToyModel:
    def __init__(self, spec):
        self.spec = spec

    def predict(self, context, control, disturbance):
        controls = np.atleast_2d(control)
        on = controls[:, 1]
        return np.stack([19+5*on, 500+0*on], -1), (10+90*on)[:, None]


class SavingsCampaignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.data = data_fixture()
        self.spec = ContextSpec(self.data)
        self.model = ToyModel(self.spec)
        self.context = self.spec.initial([24, 500], np.array([0., 1.]))
        self.disturbances = np.ones((3, 1))*25

    def planner(self, **options):
        return EnergyCategoricalCEM(
            self.model, np.eye(2), CONSTRAINTS, [], {**OPTIONS, **options}, np.empty((0, 1)))

    def test_default_ablation_matches_original_cem(self):
        old = CategoricalCEM(self.model, np.eye(2), CONSTRAINTS, [], OPTIONS, np.empty((0, 1)))
        new = self.planner()
        for _ in range(2):
            a = old.solve(self.context, self.disturbances, np.array([0., 1.]), np.ones(2, bool))
            b = new.solve(self.context, self.disturbances, np.array([0., 1.]), np.ones(2, bool))
            np.testing.assert_array_equal(a.controls, b.controls)
            self.assertAlmostEqual(a.objective, b.objective)

    def test_large_energy_weight_changes_tradeoff(self):
        normal = self.planner().solve(
            self.context, self.disturbances, np.array([0., 1.]), np.ones(2, bool))
        energy = self.planner(energy_weight=100).solve(
            self.context, self.disturbances, np.array([0., 1.]), np.ones(2, bool))
        np.testing.assert_array_equal(normal.controls[0], [0, 1])
        np.testing.assert_array_equal(energy.controls[0], [1, 0])

    def test_quality_budget_rejects_energy_saving_that_worsens_state(self):
        planner = self.planner(quality_budget=True, energy_weight=100)
        result = planner.solve(
            self.context, self.disturbances, np.array([0., 1.]), np.ones(2, bool))
        np.testing.assert_array_equal(result.controls[0], [0, 1])
        self.assertTrue(planner.last_diagnostics['quality_budget_feasible'])
        self.assertFalse(planner.last_diagnostics['quality_recovery_mode'])

    def test_recovery_is_disclosed_when_first_mask_has_no_budget_feasible_action(self):
        planner = self.planner(quality_budget=True, control_horizon=1)
        result = planner.solve(
            self.context, self.disturbances, np.array([0., 1.]), np.array([True, False]))
        self.assertIn('recovery', result.status)
        self.assertTrue(planner.last_diagnostics['quality_recovery_mode'])

    def test_mask_and_raw_transition_cap_hold(self):
        for options in ({'explicit_warm_candidate': True}, {'cem_block_steps': 2},
                        {'stratified_first_actions': True}, campaign.SMART):
            # SMART has config dotted keys; direct planner options are tested separately above.
            planner = self.planner(**{k.split('.')[-1]: v for k, v in options.items()})
            result = planner.solve(
                self.context, self.disturbances, np.array([0., 1.]), np.array([False, True]))
            np.testing.assert_array_equal(result.controls[0], [0, 1])
        no_action = self.planner().solve(
            self.context, self.disturbances, np.array([0., 1.]), np.zeros(2, bool))
        self.assertEqual(len(no_action.controls), 0)

    def test_explicit_warm_does_not_increase_candidate_budget(self):
        a, b = self.planner(), self.planner(explicit_warm_candidate=True)
        for planner in (a, b):
            planner.warm = np.array([1, 0])
            planner.solve(self.context, self.disturbances, np.array([0., 1.]), np.ones(2, bool))
        self.assertEqual(a.last_diagnostics['candidate_rollout_count'],
                         b.last_diagnostics['candidate_rollout_count'])

    def test_physical_transition_ignores_only_off_ancillary_changes(self):
        encoder = ControlEncoder([
            ControlVariable('냉난방전원', ('OFF', 'ON'), 'OFF'),
            ControlVariable('냉난방풍량', ('약풍', '강풍'), '약풍'),
            ControlVariable('냉난방운전모드', ('냉방', '난방'), '냉방')])
        off = encoder.encode_action()
        on = encoder.encode_action({'냉난방전원': 'ON', '냉난방풍량': '강풍',
                                    '냉난방운전모드': '난방'})
        other = encoder.encode_action({'냉난방전원': 'ON'})
        self.assertEqual(float(physical_changes(encoder, off, on)), 1.)
        self.assertEqual(float(physical_changes(encoder, on, other)), 2.)
        np.testing.assert_array_equal(physical_changes(
            encoder, np.stack([off, on]), np.stack([on, other])), [1., 2.])

    def test_assumed_dwell_applies_to_observed_actions_and_resets_per_window(self):
        data = data_fixture()
        data.encoder = ControlEncoder([
            ControlVariable('냉난방전원', ('OFF', 'ON'), 'OFF')])
        model = ToyModel(ContextSpec(data))
        planner = EnergyCategoricalCEM(model, np.eye(2), CONSTRAINTS, [], {
            **OPTIONS, 'assumed_minimum_onoff_steps': 3}, np.empty((0, 1)))
        off, on = np.array([1., 0.]), np.array([0., 1.])
        planner.observe_applied(off, on)
        self.assertEqual(planner.remaining.tolist(), [2])
        for remaining in (2, 1):
            result = planner.solve(self.context, self.disturbances, on, [True, False])
            self.assertEqual(len(result.controls), 0)
            self.assertEqual(planner.remaining.tolist(), [remaining])
            planner.observe_applied(on, on)
        result = planner.solve(self.context, self.disturbances, on, [True, False])
        self.assertEqual(len(result.controls), 2)
        planner.observe_applied(on, off)
        planner.reset()
        self.assertEqual(planner.remaining.tolist(), [0])
        self.assertIsNone(planner.warm)

    def test_dwell_bootstrap_does_not_confuse_gap_reset_memory_with_previous_command(self):
        from types import SimpleNamespace
        from controllers.full_control_backend import ControllerAdapter
        data = data_fixture()
        data.encoder = ControlEncoder([ControlVariable('냉난방전원', ('OFF', 'ON'), 'OFF')])
        data.timestamps.iloc[80:] += pd.Timedelta(minutes=10)
        spec = ContextSpec(data)
        backend = SimpleNamespace(
            data=data, spec=spec, contexts=spec.contexts(data), online_context=None,
            state_projection=None, power_projection=None, cfg=OmegaConf.create({}))
        planner = EnergyCategoricalCEM(ToyModel(spec), np.eye(2), CONSTRAINTS, [], {
            **OPTIONS, 'assumed_minimum_onoff_steps': 3}, np.empty((0, 1)))
        controller = ControllerAdapter(backend, object(), planner)
        controller.prepare_factual(data, 80)
        np.testing.assert_array_equal(
            backend.online_context[2*spec.nx:2*spec.nx+spec.nz], [1, 0])
        np.testing.assert_array_equal(data.controls[79], [0, 1])
        controller.observe_step(data.states[80], data.controls[79], data.states[81])
        self.assertEqual(planner.remaining.tolist(), [0])
        controller.observe_step(data.states[81], np.array([1., 0.]), data.states[82])
        self.assertEqual(planner.remaining.tolist(), [2])

    def test_mape_floors_coverage_and_no_wape_substitution(self):
        rows = power_metric_rows(np.array([[0.], [1.], [10.]]),
                                 np.array([[9.], [2.], [11.]]), ['p'],
                                 protocol='factual', split='test', floor_w=.1,
                                 target_percent=2., minimum_samples=2)
        row = rows[0]
        self.assertAlmostEqual(row['mape_percent'], 55.)
        self.assertEqual(row['mape_valid_samples'], 2)
        self.assertTrue(row['metric_reportable'])
        self.assertFalse(row['mape_target_pass'])
        self.assertAlmostEqual(row['denominator_coverage_percent'], 200/3)
        self.assertNotEqual(row['mape_percent'], row['wape_percent'])

    def test_nonfinite_predictions_cannot_pass(self):
        rows = power_metric_rows(np.array([[10.], [10.]]),
                                 np.array([[10.], [np.nan]]), ['p'],
                                 protocol='factual', split='test', floor_w=.1,
                                 target_percent=2., minimum_samples=1)
        self.assertFalse(rows[0]['metric_reportable'])
        self.assertFalse(rows[0]['mape_target_pass'])
        self.assertEqual(rows[0]['nonfinite_count'], 1)

    def test_no_nonzero_denominator_is_null(self):
        rows = power_metric_rows(np.zeros((2, 1)), np.zeros((2, 1)), ['p'],
                                 protocol='factual', split='test', floor_w=5.,
                                 target_percent=2.)
        self.assertIsNone(rows[0]['mape_percent'])
        self.assertIsNone(rows[0]['wape_percent'])
        self.assertFalse(rows[0]['mape_target_pass'])

    def test_mape_target_is_optional(self):
        rows = power_metric_rows(np.array([[10.], [10.]]), np.array([[10.], [10.]]), ['p'],
                                 protocol='factual', split='test', floor_w=.1,
                                 minimum_samples=1)
        self.assertIsNone(rows[0]['target_percent'])
        self.assertIsNone(rows[0]['mape_target_pass'])

    def test_rollout_starts_never_cross_gaps_or_split_end(self):
        np.testing.assert_array_equal(contiguous_rollout_starts(
            np.array([0, 1, 2, 10, 11, 12, 13]), 3), [0, 3, 4])
        self.assertEqual(len(contiguous_rollout_starts(np.arange(2), 3)), 0)

    def test_power_heads_multistep_fit_save_reload(self):
        model = create_control_model('physics_residual', self.spec, {
            **PARAMETERS, 'separate_power_heads': True, 'rollout_training_steps': 3}, 42)
        contexts = self.spec.contexts(self.data)
        train = self.spec.batch(self.data, contexts, 0, 100)
        val = self.spec.batch(self.data, contexts, 100, 120)
        model.fit(train, val)
        self.assertEqual(model.training_log[0]['rollout_training_steps'], 3)
        self.assertLess(model.fit_rows.max(), 99)
        expected = model.predict(val.context, val.controls, val.disturbances)
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            loaded = load_control_model(tmp)
            actual = loaded.predict(val.context, val.controls, val.disturbances)
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a, b, atol=1e-5)

    def test_all_predeclared_arms_preserve_air_quality_and_scope(self):
        original = OmegaConf.load(ROOT/'src/configs/default_setting.yaml')
        original_bytes = OmegaConf.to_yaml(original)
        for arm in campaign.ARMS:
            cfg = campaign.configure_policy(original, arm, 'validation')
            self.assertEqual(cfg.MPC.prediction_horizon, 60)
            self.assertEqual(cfg.MPC.control_horizon, 10)
            self.assertEqual(cfg.MPC.evaluation.max_windows, 0)
            self.assertTrue(cfg.MPC.evaluation.action_model_screen.state_safety.enabled)
            self.assertEqual(cfg.MPC.state_constraints[:4], original.MPC.state_constraints[:4])
            self.assertEqual(cfg.MPC.evaluation.policy_evaluation_split, 'validation')
            if 'service' not in arm:
                self.assertEqual(cfg.MPC.state_constraints, original.MPC.state_constraints)
        self.assertEqual(OmegaConf.to_yaml(original), original_bytes)
        self.assertEqual(len(campaign.CELLS), 12)
        self.assertEqual(len(campaign.METHODS), 6)
        self.assertEqual(len(campaign.ARMS), 20)

    def test_validation_scope_does_not_enter_test(self):
        data = data_fixture(1800)
        cfg = small_config()
        cfg.MPC.evaluation.max_windows = 0
        cfg.MPC.evaluation.closed_loop_steps = 60
        cfg.MPC.evaluation.stride = 60
        cfg.MPC.evaluation.action_options.episode_group_split.enabled = False
        OmegaConf.update(cfg, 'MPC.evaluation.policy_evaluation_split', 'validation', force_add=True)
        with patch('data_provider.control_data.load_control_data', return_value=data):
            audit = scope_audit(cfg)
        self.assertTrue(all(start >= audit['train_end_exclusive'] for start in audit['window_starts']))
        self.assertTrue(all(start+60 < audit['test_start'] for start in audit['window_starts']))

    def test_report_mape_never_scores_changed_action_against_recorded_truth(self):
        cfg = OmegaConf.create({'MPC': {
            'power_variables': ['p'], 'control_variables': [{'name': 'mode'}]}})
        frame = pd.DataFrame({
            'recorded_actual_p': [10., 10.],
            'recorded_action_report_predicted_p': [11., 11.],
            'mpc_action_report_predicted_p': [10., 1000.],
            'recorded_control_mode': ['ON', 'ON'], 'mpc_control_mode': ['ON', 'OFF']})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            frame.to_csv(path/'mpc_counterfactual_total_power_report_predictions.csv', index=False)
            rows, _ = campaign.artifact_mape(cfg, path)
        selected = [r for r in rows if 'exact_recorded_action' in r['protocol']]
        self.assertTrue(selected)
        self.assertTrue(all(r['mape_percent'] == 0 for r in selected))
        self.assertTrue(all(r['action_match_coverage_percent'] == 50 for r in selected))

    def test_mape_worker_refits_without_milp_and_writes_reproducible_arrays(self):
        data = data_fixture(1800)
        cfg = small_config()
        cfg.MPC.evaluation.action_options.episode_group_split.enabled = False
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            job = {'method': campaign.M3, 'checkpoint_source': str(directory/'missing'),
                   'artifact_source': str(directory/'source')}
            with patch.object(campaign, 'split_data', return_value=(data, 1440, 1620)):
                campaign.run_mape(cfg, job, directory)
            metrics = pd.read_csv(directory/'mape_metrics.csv')
            self.assertEqual(set(metrics.split), {'validation', 'test'})
            self.assertTrue({'controller_factual_one_step', 'controller_recorded_recursive_all60'}
                            <= set(metrics.protocol))
            arrays = np.load(directory/'predictions.npz')
            self.assertEqual(arrays['test_actual_power'].shape, (179, 1))
            self.assertTrue((arrays['test_rows'] >= 1620).all())
            self.assertTrue((arrays['validation_rollout_rows_lead_major']+1 < 1620).all())

    def test_opt_in_validation_pipeline_and_standard_band_supplement(self):
        from controllers.full_control_backend import FullControlBackend
        from controllers.pipeline import run_mpc_pipeline
        data = data_fixture(180)
        data.control_preprocessing = {'category_consolidation': {'enabled': False, 'mappings': {}}}
        cfg = campaign.configure_policy(small_config(), 'quality_budget', 'validation')
        cfg.MPC.evaluation.action_options.episode_group_split.enabled = False
        cfg.MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.candidate_families = [
            'action_median']
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            backend = FullControlBackend(cfg, campaign.M3, directory)
            with patch('controllers.pipeline.load_control_data', return_value=data):
                run_mpc_pipeline(cfg, directory, control_backend=backend)
            trace = pd.read_csv(directory/'mpc_closed_loop_steps.csv')
            self.assertTrue('next_state_temperature' in trace)
            times = pd.to_datetime(trace.timestamp)
            self.assertTrue((times < data.timestamps.iloc[162]).all())
            with patch.object(campaign, 'split_data', return_value=(data, 144, 162)):
                campaign.policy_supplement(cfg, directory, cfg.MPC.state_constraints)
            self.assertTrue((directory/'state_bands_audit.json').exists())
            from runners.counterfactual_report_input_ablation_runner import run_counterfactual_report_input_ablation
            cfg.RESULT.savedir = str(directory/'cf_inputs')
            cfg.MPC.counterfactual_report_input_ablation.reference_mpc_result_dir = str(directory)
            with patch('runners.counterfactual_report_input_ablation_runner.load_control_data',
                       return_value=data):
                result = run_counterfactual_report_input_ablation(cfg)
            self.assertEqual(len(result['case_results']), 3)
            self.assertEqual(result['reference_mpc_run']['counterfactual_transition_count'], 2)

    def test_observed_report_input_runner_and_topk_execute_without_milp(self):
        from runners.report_model_input_ablation_runner import run_report_model_input_ablation
        data = data_fixture(1800)
        # Existing report markdown is explicitly a ventilation+HVAC two-component report.
        data.powers = np.column_stack([data.powers[:, 0], 2*data.powers[:, 0]])
        data.power_names = ('환기유효전력(W)', '냉난방유효전력(W)')
        data.control_preprocessing = {'category_consolidation': {'enabled': False, 'mappings': {}}}
        cfg = small_config()
        cfg.MPC.evaluation.action_options.episode_group_split.enabled = False
        cfg.MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.candidate_families = [
            'action_median']
        with tempfile.TemporaryDirectory() as tmp:
            cfg.RESULT.savedir = tmp
            with patch('runners.report_model_input_ablation_runner.load_control_data', return_value=data):
                result = run_report_model_input_ablation(cfg)
        self.assertEqual(len(result['cases']), 3)
        self.assertTrue(result['expanded_candidate_pool']['enabled'])
        self.assertEqual(len(result['expanded_candidate_pool']['cases']), 3)

    def test_lossless_compression_preserves_digest_and_only_own_diagnostics(self):
        import gzip
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            raw = directory/'mpc_action_candidate_diagnostics.csv'
            content = b'candidate,state,decision\n1,20,accepted\n'*100
            raw.write_bytes(content)
            keep = directory/'mpc_closed_loop_steps.csv'
            keep.write_text('preserved')
            campaign.compress_diagnostics(directory)
            self.assertFalse(raw.exists())
            with gzip.open(directory/(raw.name+'.gz'), 'rb') as stream:
                self.assertEqual(stream.read(), content)
            evidence = json.loads((directory/'lossless_compression.json').read_text())
            self.assertTrue(evidence[raw.name]['recovered_sha256_verified'])
            self.assertEqual(keep.read_text(), 'preserved')

    def test_supervisor_waits_for_predecessor_and_never_exceeds_eight_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = []
            for i in range(9):
                directory = root/str(i)
                directory.mkdir()
                jobs.append({'kind': 'mape', 'cohort': 'original',
                             'relative_directory': str(i)})
            manifest = {'jobs': jobs, 'predecessor_root': str(root/'previous'),
                        'repository_root': str(root), 'minimum_available_memory_gb': 24,
                        'minimum_free_disk_gb': 32}
            (root/'manifest.json').write_text(json.dumps(manifest))
            tracking = {'created': 0, 'active': 0, 'peak': 0, 'predecessor_calls': 0}

            def predecessor(_):
                tracking['predecessor_calls'] += 1
                return ([], 'running' if tracking['predecessor_calls'] <= 2 else 'completed')

            class FakeProcess:
                def __init__(self, arguments, **kwargs):
                    self.index = int(arguments[-1])
                    self.pid = 900000+self.index
                    self.finished = False
                    self.directory = root/str(self.index)
                    tracking['created'] += 1
                    tracking['active'] += 1
                    tracking['peak'] = max(tracking['peak'], tracking['active'])
                    self_test.assertGreater(tracking['predecessor_calls'], 2)
                    self_test.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '1')
                    (self.directory/'campaign_status.json').write_text(json.dumps({
                        'status': 'running', 'pid': self.pid}))

                def poll(self):
                    if tracking['created'] < 8:
                        return None
                    if not self.finished:
                        tracking['active'] -= 1
                        self.finished = True
                        (self.directory/'campaign_status.json').write_text(
                            json.dumps({'status': 'completed'}))
                    return 0

            self_test = self
            with patch.object(campaign, 'validate'), patch.object(campaign, 'aggregate'), \
                    patch.object(campaign, 'live_predecessor_pids', side_effect=predecessor), \
                    patch.object(campaign, 'available_memory_gb', return_value=128), \
                    patch.object(campaign.shutil, 'disk_usage', return_value=type(
                        'Disk', (), {'free': 128*1024**3})()), \
                    patch.object(campaign.subprocess, 'Popen', FakeProcess), \
                    patch.object(campaign.time, 'sleep'):
                result = campaign.supervise(root)
            self.assertEqual(result, 0)
            self.assertEqual(tracking['peak'], 8)
            self.assertEqual(tracking['created'], 9)
            self.assertEqual(json.loads((root/'manifest.json').read_text())['counts'], {'completed': 9})


if __name__ == '__main__':
    unittest.main()
