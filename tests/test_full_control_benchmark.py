"""Original-pipeline integration and causal full-benchmark adapter contracts."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch

from controllers.full_control_backend import FullControlBackend, HistoryPlant
from controllers.pipeline import run_mpc_pipeline
from dynamics.affine import fit_affine_models
from runners.full_control_benchmark_runner import mpc_fingerprint, run_full_matrix
from test_control_benchmark import data_fixture, CONSTRAINTS, PARAMETERS


def small_config():
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1]/'src/configs/default_setting.yaml')
    return OmegaConf.merge(cfg, {
        'MPC': {
            'prediction_horizon': 3, 'control_horizon': 2, 'logical_rules': [],
            'state_constraints': CONSTRAINTS,
            'objective': {'maximum_changed_control_variables_per_step': 1,
                          'action_history': {'groups': None}},
            'solver': {'time_limit_seconds': 5, 'mip_relative_gap': 0},
            'independent_evaluator': {'model_family': 'extra_trees_residual_state',
                                      'n_estimators': 5, 'min_samples_leaf': 1},
            'evaluation': {
                'closed_loop_steps': 2, 'stride': 2, 'max_windows': 1,
                'lead_time_steps': [1, 2], 'bootstrap_samples': 10,
                'action_options': {
                    'repeated_joint_actions': {'minimum_training_samples': 1,
                                               'minimum_training_episodes': 1},
                },
                'action_trust': {
                    'global_support': {'minimum_samples': 1, 'minimum_episodes': 1},
                    'local_context_support': {
                        'minimum_validation_contexts': 2, 'neighbor_count': 1,
                        'minimum_samples_within_radius': 1,
                    },
                },
                'fixed_optimal_action_holdout': {'enabled': False},
                'fixed_action_candidate_pool': {'enabled': False},
                'selected_action_crossfit': {'enabled': False},
                'mpc_counterfactual_total_power_report': {
                    'enabled': True, 'minimum_validation_mape_samples': 1,
                    'minimum_test_mape_samples': 1,
                },
            },
            'counterfactual_report_input_ablation': {'run_after_mpc': False},
            'vent_isolated_kpi': {'enabled': False},
        },
        'CONTROL_BENCHMARK': {'model_parameters': PARAMETERS,
                              'planner': {'mode_matrix_strategy': 'exact_disjunction'}},
    })


class FullControlBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_original_pipeline_all_six_and_baseline_parity(self):
        data = data_fixture(180)
        data.control_preprocessing = {
            'category_consolidation': {'enabled': False, 'mappings': {}},
        }
        cfg = small_config()
        cfg.MPC.evaluation.fixed_optimal_action_holdout.total_power_report_model.candidate_families = [
            'action_median',
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('controllers.pipeline.load_control_data', return_value=data):
                original = run_mpc_pipeline(cfg, root/'original')
            for method in cfg.CONTROL_FULL_BENCHMARK.methods:
                with self.subTest(method=method):
                    before = mpc_fingerprint(cfg)
                    directory = root/method
                    directory.mkdir()
                    backend = FullControlBackend(cfg, method, directory)
                    with patch('controllers.pipeline.load_control_data', return_value=data):
                        result = run_mpc_pipeline(cfg, directory, control_backend=backend)
                    self.assertEqual(mpc_fingerprint(cfg), before)
                    self.assertEqual(result['evaluated_windows'], original['evaluated_windows'])
                    self.assertEqual(result['evaluation_protocol']['closed_loop_steps_per_window'], 2)
                    self.assertTrue((directory/'model/model.joblib').is_file())
                    self.assertTrue((directory/'mpc_closed_loop_steps.csv').is_file())
                    effective = result['control_backend']['effective_planner_parameters']
                    self.assertEqual(effective['time_limit_seconds'], cfg.MPC.solver.time_limit_seconds)
                    self.assertEqual(effective['mip_relative_gap'], cfg.MPC.solver.mip_relative_gap)
                    self.assertEqual(set(result['quality_assessment']['model_validation']['gates']),
                                     set(original['quality_assessment']['model_validation']['gates']))
                    if method == 'm1_affine_milp':
                        np.testing.assert_allclose(result['optimized_energy_wh'],
                                                   original['optimized_energy_wh'])
                        self.assertEqual(result['quality_assessment'], original['quality_assessment'])
                    if method in ('m3_physics_residual', 'm5_neural_cem', 'm6_decision_focused'):
                        self.assertEqual(result['policy_evaluation']['solver']['optimal_success_count'], 0)

    def test_history_batch_rollout_and_applied_fallback_observation(self):
        cfg, data = small_config(), data_fixture(180)
        arrays = data.model_arrays(0, 119)
        state, power = fit_affine_models(
            *arrays, state_names=data.state_names,
            control_feature_names=data.encoder.feature_names,
            disturbance_names=data.disturbance_names, power_names=data.power_names,
            state_alpha=1, power_alpha=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            backend = FullControlBackend(cfg, 'm2_mode_history_milp', temporary)
            backend.attach(data, 120, 150, state, power)
            plant = HistoryPlant(backend)
            predicted = plant.predict_transitions(data, 150, 160)
            expected = backend.model.predict(backend.contexts[150:160], data.controls[150:160],
                                             data.disturbances[150:160])
            np.testing.assert_allclose(predicted.predicted_states, expected[0])
            session = plant.start_rollout()
            context = backend.contexts[150].copy()
            for row in (150, 151):
                x, p = session.step(context[:2], data.controls[row], data.disturbances[row],
                                    timestamp=data.timestamps.iloc[row])
                ex, ep = backend.model.predict(context, data.controls[row], data.disturbances[row])
                np.testing.assert_allclose(x, ex[0])
                np.testing.assert_allclose(p, ep[0])
                context = backend.spec.advance(context, data.controls[row], x)
            # A calibration row path must never silently bootstrap default history.
            np.testing.assert_allclose(
                backend.power_projection.predict_components_for_rows(data, np.arange(150, 160)),
                expected[1],
            )
            with self.assertRaises(RuntimeError):
                backend.power_projection.predict_components(data.states[150], data.controls[150],
                                                             data.disturbances[150])

    def test_matrix_defaults_do_not_reduce_original_evaluation(self):
        cfg = OmegaConf.load(Path(__file__).resolve().parents[1]/'src/configs/default_setting.yaml')
        full = cfg.CONTROL_FULL_BENCHMARK
        self.assertEqual(len(full.dataset_groups)*len(full.environments)*len(full.methods)*len(full.seeds), 72)
        self.assertEqual(cfg.MPC.evaluation.closed_loop_steps, 60)
        self.assertEqual(cfg.MPC.evaluation.max_windows, 0)
        self.assertEqual(cfg.MPC.solver.time_limit_seconds, 60)
        self.assertEqual(cfg.MPC.solver.mip_relative_gap, 0.01)
        self.assertEqual(full.mode_matrix_strategy, 'exact_disjunction')

    def test_prepared_matrix_freezes_config_and_archives_failed_attempt(self):
        cfg = small_config()
        cfg.CONTROL_FULL_BENCHMARK.dataset_groups = [
            {'name': 'fixture', 'versions': ['fixture']},
        ]
        cfg.CONTROL_FULL_BENCHMARK.environments = ['room1']
        cfg.CONTROL_FULL_BENCHMARK.methods = ['m1_affine_milp', 'm2_mode_history_milp']
        with tempfile.TemporaryDirectory() as temporary:
            cfg.CONTROL_FULL_BENCHMARK.output_directory = temporary
            source = Path(temporary)/'source.csv'
            source.write_text('fixture,data\n1,2\n')
            scope = {
                'window_starts': [150], 'expected_policy_steps': 2,
                'counterfactual_report_rows': 2, 'mpc_config_sha256': mpc_fingerprint(cfg),
            }
            with patch('runners.full_control_benchmark_runner.scope_audit', return_value=scope), patch(
                    'data_provider.dataset_versions.resolve_dataset_csv_path', return_value=source):
                root = run_full_matrix(cfg, prepare_only=True)
            manifest = json.loads((root/'manifest.json').read_text())
            self.assertEqual(manifest['case_count'], 2)
            for case in manifest['cases']:
                saved = OmegaConf.load(root/case['relative_directory']/'configs.yaml')
                self.assertEqual(mpc_fingerprint(saved), mpc_fingerprint(cfg))
            case_dir = root/manifest['cases'][0]['relative_directory']
            (case_dir/'error.txt').write_text('original failed attempt')
            (case_dir/'status.json').write_text('{"status":"failed"}')
            run_full_matrix(cfg, existing_root=root, prepare_only=True)
            self.assertFalse((case_dir/'error.txt').exists())
            archive = next(case_dir.parent.glob(case_dir.name+'_attempt_*'))
            self.assertEqual((archive/'error.txt').read_text(), 'original failed attempt')
            self.assertTrue((case_dir/'scope_audit.json').exists())

    def test_resume_queue_order_and_memory_guard_only_change_launch_order(self):
        cfg = small_config()
        cfg.CONTROL_FULL_BENCHMARK.dataset_groups = [
            {'name': 'fixture', 'versions': ['fixture']},
        ]
        cfg.CONTROL_FULL_BENCHMARK.environments = ['room1', 'room2']
        cfg.CONTROL_FULL_BENCHMARK.methods = ['m1_affine_milp', 'm2_mode_history_milp',
                                              'm3_physics_residual']
        with tempfile.TemporaryDirectory() as temporary:
            cfg.CONTROL_FULL_BENCHMARK.output_directory = temporary
            source = Path(temporary)/'source.csv'
            source.write_text('fixture,data\n1,2\n')
            scope = {
                'window_starts': [150], 'expected_policy_steps': 2,
                'counterfactual_report_rows': 2, 'mpc_config_sha256': mpc_fingerprint(cfg),
            }
            with patch('runners.full_control_benchmark_runner.scope_audit', return_value=scope), patch(
                    'data_provider.dataset_versions.resolve_dataset_csv_path', return_value=source):
                root = run_full_matrix(cfg, prepare_only=True)
            manifest = json.loads((root/'manifest.json').read_text())
            done = root/manifest['cases'][0]['relative_directory']/'status.json'
            done.write_text('{"status":"completed"}')
            configs_before = {
                case['relative_directory']:
                    (root/case['relative_directory']/'configs.yaml').read_bytes()
                for case in manifest['cases']
            }
            launched, alive = [], {}

            class FakeProcess:
                def __init__(self, command, **kwargs):
                    self.pid = 1000+len(launched)
                    self.directory = Path(command[command.index('--worker-config')+1]).parent
                    launched.append((command[command.index('--method')+1], len(alive)))
                    alive[self.pid] = self
                    self.polls = 0

                def poll(self):
                    self.polls += 1
                    if self.polls < 2:
                        return None
                    (self.directory/'status.json').write_text('{"status":"completed"}')
                    alive.pop(self.pid, None)
                    return 0

            memory = iter([1.0, 64.0]*20)
            with patch('runners.full_control_benchmark_runner.subprocess.Popen', FakeProcess), patch(
                    'runners.full_control_benchmark_runner.time.sleep'), patch(
                    'runners.full_control_benchmark_runner.available_memory_gb',
                    side_effect=lambda: next(memory)), patch(
                    'reporting.full_control_comparison.summarize_full_matrix'):
                run_full_matrix(cfg, existing_root=root, workers=3,
                                method_order=['m3_physics_residual', 'm1_affine_milp'],
                                min_available_memory_gb=8)
            self.assertEqual([m for m, _ in launched], [
                'm3_physics_residual', 'm3_physics_residual', 'm1_affine_milp',
                'm2_mode_history_milp', 'm2_mode_history_milp',
            ])
            final = json.loads((root/'manifest.json').read_text())
            self.assertEqual(final['completed_cases'], 6)
            # Low memory deferred a launch while another case was still active.
            self.assertEqual(final['last_memory_deferral']['available_gb'], 1.0)
            self.assertGreaterEqual(final['last_memory_deferral']['active'], 1)
            self.assertLessEqual(max(count for _, count in launched), 2)
            self.assertEqual(final['supervisor_launches'][-1]['method_order'],
                             ['m3_physics_residual', 'm1_affine_milp'])
            self.assertEqual(final['supervisor_launches'][-1]['parallel_workers'], 3)
            for relative, content in configs_before.items():
                self.assertEqual((root/relative/'configs.yaml').read_bytes(), content)
            with self.assertRaises(ValueError):
                run_full_matrix(cfg, existing_root=root, method_order=['m9_unknown'])


if __name__ == '__main__':
    unittest.main()
