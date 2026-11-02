"""Regression, causality, integer feasibility and artifact contracts."""

from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from controllers.control_benchmark import create_control_planner
from data_provider.control_data import ControlData, ControlEncoder, ControlVariable
from models.control import ContextSpec, create_control_model, load_control_model
from models.control.base import stage_cost, categorical_changes
from runners.control_benchmark_runner import select_windows, run_control_benchmark


def data_fixture(n=150):
    encoder = ControlEncoder([ControlVariable('mode', ('OFF', 'ON'), 'OFF')])
    z = np.eye(2)[(np.arange(n)//5) % 2]
    d = np.sin(np.arange(n)/15)[:, None] + 25
    x = np.empty((n, 2))
    p = np.empty((n, 1))
    x[0] = [24, 500]
    for i in range(n-1):
        x[i+1] = x[i] + [0.03*(d[i, 0]-x[i, 0])-0.1*z[i, 1], 1-2*z[i, 1]]
        p[i] = 10+70*z[i, 1]
    p[-1] = 10+70*z[-1, 1]
    return ControlData(pd.Series(pd.date_range('2026-01-01', periods=n, freq='min')),
                       x, z, d, p, ('temperature', 'co2'), ('outdoor',), ('power',), encoder,
                       segment_ids=np.zeros(n, dtype=int))


CONSTRAINTS = [
    {'name': 'temperature', 'lower': 20, 'upper': 26, 'penalty': 10},
    {'name': 'co2', 'lower': None, 'upper': 1000, 'penalty': 1},
]
PARAMETERS = {
    'epochs': 2, 'patience': 2, 'batch_size': 32, 'hidden_size': 8,
    'ensemble_size': 2, 'bootstrap_block': 8, 'device': 'cpu',
    'decision_samples': 8, 'decision_horizon': 2, 'decision_epochs': 1,
    'decision_batch_size': 4,
}
OPTIONS = {
    'prediction_horizon': 3, 'control_horizon': 2, 'maximum_changed_variables': 1,
    'time_limit_seconds': 5, 'mip_relative_gap': 0, 'switch_penalty': 0.05,
    'cem_population': 12, 'cem_iterations': 2, 'seed': 42,
    'state_box_padding': 3, 'risk_weight': 0.5,
}


class ControlBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = data_fixture()
        cls.spec = ContextSpec(cls.data)
        cls.contexts = cls.spec.contexts(cls.data)
        cls.train = cls.spec.batch(cls.data, cls.contexts, 0, 100)
        cls.val = cls.spec.batch(cls.data, cls.contexts, 100, 120)

    def model(self, family):
        model = create_control_model(family, self.spec, PARAMETERS, 42)
        if family == 'decision_focused':
            model.decision_actions = np.eye(2)
            model.decision_constraints = CONSTRAINTS
        return model.fit(self.train, self.val)

    def test_all_six_families_fit_save_reload(self):
        for family in ('affine', 'mode_history', 'physics_residual',
                       'scenario_ensemble', 'neural_ensemble', 'decision_focused'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as root:
                model = self.model(family)
                expected = model.predict(self.val.context, self.val.controls, self.val.disturbances)
                self.assertTrue(np.isfinite(expected[0]).all())
                self.assertTrue((expected[1] >= 0).all())
                model.save(root)
                loaded = load_control_model(root)
                actual = loaded.predict(self.val.context, self.val.controls, self.val.disturbances)
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b, atol=1e-5)
                meta = json.loads((Path(root)/'model_metadata.json').read_text())
                self.assertEqual(meta['family'], family)
                self.assertLess(max(loaded.fit_rows), 99)
                if family == 'decision_focused':
                    self.assertLess(model.teacher_rows.max(), model.decision_rows.min())
                    self.assertTrue(any(x['stage'] == 'decision_focused' for x in model.training_log))
                if hasattr(model, 'members'):
                    self.assertFalse(np.array_equal(model.members[0].fit_rows, model.members[1].fit_rows))

    def test_causal_history_and_gap_reset(self):
        data = data_fixture()
        before = self.spec.contexts(data)
        data.controls[70:] = data.controls[70:, ::-1]
        after = self.spec.contexts(data)
        np.testing.assert_array_equal(before[:71], after[:71])
        self.assertFalse(np.array_equal(before[71], after[71]))
        data.timestamps.iloc[80:] += pd.Timedelta(minutes=10)
        contexts = self.spec.contexts(data)
        np.testing.assert_array_equal(contexts[80], self.spec.initial(data.states[80]))
        batch = self.spec.batch(data, contexts, 70, 90)
        self.assertNotIn(79, batch.rows)
        self.assertNotIn(89, batch.rows)

    def test_linear_coefficients_equal_direct_prediction(self):
        for family in ('affine', 'mode_history'):
            model = self.model(family)
            context = self.contexts[50]
            d = self.data.disturbances[50]
            A, b, F, g = model.linear_coefficients(np.eye(2), d)
            x, p = model.predict(np.repeat(context[None], 2, axis=0), np.eye(2), np.repeat(d[None], 2, axis=0))
            np.testing.assert_allclose(np.einsum('kij,j->ki', A, context)+b, x, atol=1e-8)
            np.testing.assert_allclose(np.maximum(np.einsum('kij,j->ki', F, context)+g, 0), p, atol=1e-8)

    def test_joint_milp_matches_exhaustive_sequences(self):
        for family in ('mode_history', 'scenario_ensemble'):
            model = self.model(family)
            planner = create_control_planner('joint_milp', model, np.eye(2),
                                             CONSTRAINTS, (), OPTIONS, self.train.context)
            context, previous = self.contexts[50], np.eye(2)[0]
            disturbances = np.repeat(self.data.disturbances[50][None], 3, axis=0)
            plan = planner.solve(context, disturbances, previous, np.ones(2, dtype=bool))
            self.assertEqual(plan.controls.shape, (2, 2))
            values = []
            for sequence in product(range(2), repeat=2):
                costs = []
                for member in getattr(model, 'members', [model]):
                    ctx, value = context.copy(), 0
                    for t in range(3):
                        z = np.eye(2)[sequence[min(t, 1)]]
                        nx, p = member.predict(ctx, z, disturbances[t])
                        value += float(stage_cost(nx[0], p[0], CONSTRAINTS))
                        ctx = self.spec.advance(ctx, z, nx[0])
                    costs.append(value)
                risk = 0.5 if family == 'scenario_ensemble' else 0
                value = (1-risk)*np.mean(costs)+risk*max(costs)
                prev = previous
                for a in sequence:
                    z = np.eye(2)[a]
                    value += 0.1*float(categorical_changes(z, prev))
                    prev = z
                values.append(value)
            self.assertAlmostEqual(plan.objective, min(values), places=4)

    def test_cem_and_baseline_respect_first_mask(self):
        for family, planner_name in (('affine', 'baseline_milp'), ('neural_ensemble', 'categorical_cem')):
            model = self.model(family)
            planner = create_control_planner(planner_name, model, np.eye(2),
                                             CONSTRAINTS, (), OPTIONS, self.train.context)
            plan = planner.solve(self.contexts[50], np.repeat(self.data.disturbances[50][None], 3, axis=0),
                                 np.eye(2)[0], np.array([False, True]))
            np.testing.assert_array_equal(plan.controls[0], [0, 1])
            self.assertTrue(np.isin(plan.controls, [0, 1]).all())

    def test_nominal_linearization_is_exact_at_one_step_reference(self):
        model = self.model('mode_history')
        options = {**OPTIONS, 'prediction_horizon': 1, 'control_horizon': 1,
                   'mode_matrix_strategy': 'nominal_linearization'}
        planner = create_control_planner('joint_milp', model, np.eye(2),
                                         CONSTRAINTS, (), options, self.train.context)
        context, previous = self.contexts[50], np.eye(2)[0]
        d = self.data.disturbances[50]
        plan = planner.solve(context, d[None], previous, np.ones(2, dtype=bool))
        values = []
        for z in np.eye(2):
            x, p = model.predict(context, z, d)
            values.append(float(stage_cost(x[0], p[0], CONSTRAINTS))
                          + 0.1*float(categorical_changes(z, previous)))
        self.assertEqual(len(plan.controls), 1)
        self.assertAlmostEqual(plan.objective, min(values), places=4)

    def test_physics_backbone_cooling_depends_on_categorical_setpoint(self):
        from models.control.neural import TransportBackbone
        cfg = OmegaConf.load('src/configs/default_setting.yaml')
        data = data_fixture()
        data.encoder = ControlEncoder.from_config(cfg.MPC.control_variables)
        data.state_names = ('실내온도(℃)', '실내이산화탄소(ppm)')
        data.disturbance_names = ('실외온도(℃)',)
        spec = ContextSpec(data)
        backbone = TransportBackbone(spec)
        backbone.state_coef[0, 3] = 0.1
        z = data.encoder.encode_action({'냉난방전원': 'ON', '냉난방풍량': '강풍',
                                       '냉난방운전모드': '냉방', '냉난방설정온도(℃)': 18})
        warm = data.encoder.encode_action({'냉난방전원': 'ON', '냉난방풍량': '강풍',
                                          '냉난방운전모드': '냉방', '냉난방설정온도(℃)': 26})
        context = torch.tensor(np.repeat(spec.initial(np.array([24., 500.]))[None], 2, axis=0), dtype=torch.float32)
        out = backbone(context, torch.tensor(np.stack([z, warm]), dtype=torch.float32),
                       torch.tensor([[30.], [30.]]))
        self.assertLess(out[0, 0].item(), out[1, 0].item())

    def test_incompatible_planner_is_rejected_and_empty_mask_has_no_action(self):
        model = self.model('neural_ensemble')
        with self.assertRaises(TypeError):
            create_control_planner('baseline_milp', model, np.eye(2),
                                   CONSTRAINTS, (), OPTIONS, self.train.context)
        planner = create_control_planner('categorical_cem', model, np.eye(2),
                                         CONSTRAINTS, (), OPTIONS, self.train.context)
        plan = planner.solve(self.contexts[50], np.repeat(self.data.disturbances[50][None], 3, axis=0),
                             np.eye(2)[0], np.zeros(2, dtype=bool))
        self.assertEqual(len(plan.controls), 0)

    def test_windows_exclude_gaps(self):
        data = data_fixture()
        data.segment_ids[130:] = 1
        starts = select_windows(data, 100, 10, 4)
        self.assertTrue(all(not (i < 130 <= i+10) for i in starts))

    def test_forecast_import_compatibility(self):
        from models.MICN import Model as old
        from models.prediction.MICN import Model as new
        from models import create_model
        from models.prediction import create_model as prediction_factory
        self.assertIs(old, new)
        self.assertIs(create_model, prediction_factory)

    def test_six_method_runner_end_to_end(self):
        with tempfile.TemporaryDirectory() as root:
            data = data_fixture()
            folder = Path(root)/'data'/'toy'
            folder.mkdir(parents=True)
            pd.DataFrame({
                '데이터 시간': data.timestamps, 'temperature': data.states[:, 0],
                'co2': data.states[:, 1], 'outdoor': data.disturbances[:, 0],
                'power': data.powers[:, 0], 'mode': np.where(data.controls[:, 0], 'OFF', 'ON'),
            }).to_csv(folder/'toy.csv', index=False)
            cfg = OmegaConf.load('src/configs/default_setting.yaml')
            cfg.DATASET.datadir = str(Path(root)/'data')
            cfg.DATASET.data_version = 'toy'
            cfg.DATASET.start_date = None
            cfg.DATASET.split_rate = [0.65, 0.15, 0.2]
            cfg.MPC.control_variables = [{'name': 'mode', 'categories': ['OFF', 'ON'], 'default': 'OFF'}]
            cfg.MPC.state_constraints = CONSTRAINTS
            cfg.MPC.logical_rules = []
            cfg.MPC.disturbance_variables = ['outdoor']
            cfg.MPC.power_variables = ['power']
            cfg.MPC.prediction_horizon, cfg.MPC.control_horizon = 3, 2
            cfg.MPC.independent_evaluator.model_family = 'extra_trees_residual_state'
            cfg.MPC.independent_evaluator.n_estimators = 4
            cfg.CONTROL_BENCHMARK.output_directory = str(Path(root)/'results')
            cfg.CONTROL_BENCHMARK.environments = ['toy']
            cfg.CONTROL_BENCHMARK.align_episode_boundaries = False
            cfg.CONTROL_BENCHMARK.evaluation_windows = 2
            cfg.CONTROL_BENCHMARK.closed_loop_steps = 2
            cfg.CONTROL_BENCHMARK.model_parameters = PARAMETERS
            cfg.CONTROL_BENCHMARK.planner = OPTIONS
            cfg.CONTROL_BENCHMARK.action_trust = {
                'enabled': True, 'global_support': {'minimum_samples': 1, 'minimum_episodes': 1},
                'local_context_support': {'enabled': False},
            }
            result = run_control_benchmark(cfg)
            self.assertEqual(len(result['results']), 6)
            self.assertTrue(all(r['status'] == 'completed' for r in result['results']))
            directory = Path(result['result_directory'])
            self.assertTrue((directory/'comparison.csv').is_file())
            self.assertTrue(all((directory/'toy'/'seed_42'/m/'model'/'model.joblib').is_file()
                                for m in cfg.CONTROL_BENCHMARK.selected_methods))
            # Swapping stored checkpoints must work without fitting again.
            cfg.CONTROL_BENCHMARK.resume_from = str(directory)
            cfg.CONTROL_BENCHMARK.evaluation_windows = 1
            resumed = run_control_benchmark(cfg)
            self.assertTrue(all(r['loaded_from_checkpoint'] for r in resumed['results']))
            self.assertNotEqual(resumed['result_directory'], result['result_directory'])
            manifest = json.loads((Path(resumed['result_directory'])/'manifest.json').read_text())
            self.assertEqual(manifest['source_changed_during_run'], [])


if __name__ == '__main__':
    unittest.main()
