"""새 절감 실험의 변경 범위와 paired 비교 계약."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('savings_runner', ROOT/'src/scripts/control/savings_improvement.py')
analysis = load('savings_analysis', ROOT/'src/reporting/savings_improvement_analysis.py')


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


class SavingsImprovementTest(unittest.TestCase):
    def test_only_declared_keys_and_common_output_changes(self):
        cfg = OmegaConf.load(ROOT/'src/configs/default_setting.yaml')
        OmegaConf.update(cfg, runner.SAFETY, 'reject', force_add=True)
        with tempfile.TemporaryDirectory() as tmp:
            source, root = Path(tmp)/'source.yaml', Path(tmp)/'out'
            OmegaConf.save(cfg, source)
            before = source.read_bytes()
            original = flatten(OmegaConf.to_container(cfg))
            for arm, (_, overrides) in runner.ARMS.items():
                with self.subTest(arm=arm):
                    actual = runner.case_config(source, root, overrides)
                    changed = {key for key, value in flatten(OmegaConf.to_container(actual)).items()
                               if original.get(key) != value}
                    allowed = set(overrides) | {
                        'RESULT.savedir', 'MPC.counterfactual_report_input_ablation.run_after_mpc',
                        'CONTROL_BENCHMARK.resume_from',
                    }
                    self.assertTrue(changed <= allowed, changed-allowed)
                    self.assertEqual(actual.MPC.prediction_horizon, 60)
                    self.assertEqual(actual.MPC.control_horizon, 10)
                    self.assertEqual(actual.MPC.evaluation.closed_loop_steps, 60)
                    self.assertEqual(actual.MPC.evaluation.max_windows, cfg.MPC.evaluation.max_windows)
                    self.assertTrue(actual.MPC.evaluation.action_model_screen.state_safety.enabled)
            self.assertEqual(source.read_bytes(), before)

    def test_design_includes_matched_controls(self):
        self.assertEqual(len(runner.CELLS)*len(runner.ARMS), 15)
        self.assertEqual(runner.ARMS['m3_original'][0], runner.M3)
        self.assertEqual(runner.ARMS['m6_original'][0], runner.M6)
        self.assertEqual(runner.ARMS['m3_b1_cem128x5'][1][
            'CONTROL_BENCHMARK.planner.cem_population'], 128)

    def test_paired_sign_and_repeatability(self):
        a = pd.DataFrame({'ts': ['a', 'b'], 'base': [12., 12.],
                          'energy': [10., 10.], 'soft': [1., 1.]})
        b = a.copy()
        b['energy'] = 8.
        first, second = analysis.paired(a, b), analysis.paired(a, b)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first['energy_b_vs_a_percent'], -20)
        self.assertTrue(first['ci_high'] < 0)

    def test_paired_rejects_misalignment(self):
        a = pd.DataFrame({'ts': ['a', 'b'], 'base': [12., 12.],
                          'energy': [10., 10.], 'soft': [1., 1.]})
        with self.assertRaises(ValueError):
            analysis.paired(a, a.iloc[::-1])
        b = a.copy()
        b['base'] += 1
        with self.assertRaises(ValueError):
            analysis.paired(a, b)

    def test_validation_rejects_modified_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root/'source_snapshot/src/example.py'
            p.parent.mkdir(parents=True)
            p.write_text('original')
            manifest = {'source_sha256': {'example.py': runner.digest(p)}}
            p.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Snapshot changed'):
                runner.validate(root, manifest)


if __name__ == '__main__':
    unittest.main()
