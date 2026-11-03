"""전체 범위·재사용 무결성·concat 분리 계약."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('savings_expansion', ROOT/'src/scripts/control/savings_expansion.py')
analysis = load('expansion_analysis', ROOT/'src/reporting/savings_improvement_final_analysis.py')


class SavingsExpansionTest(unittest.TestCase):
    def test_full_scope_and_disjoint_reporting(self):
        cases = [{'group': g, 'environment': r, 'arm': a}
                 for g, r in runner.CELLS for a in runner.ARMS]
        self.assertEqual(len(cases), 36)
        manifest = {'experiment_kind': 'm3_full_scope_expansion', 'cases': cases}
        cohorts = analysis.evaluation_cohorts(manifest)
        self.assertEqual(len(cohorts['single_versions']), 9)
        self.assertEqual(len(cohorts['concat']), 3)
        self.assertFalse(set(cohorts['single_versions']) & set(cohorts['concat']))
        manifest['cases'] = cases[:-1]
        with self.assertRaises(ValueError):
            analysis.evaluation_cohorts(manifest)

    def make_case(self, old, target):
        old.mkdir()
        target.mkdir()
        cfg = OmegaConf.create({'RESULT': {'savedir': 'old'},
                               'CONTROL_BENCHMARK': {'resume_from': None},
                               'MPC': {'horizon': 60}})
        OmegaConf.save(cfg, old/'configs.yaml')
        cfg.RESULT.savedir = 'new'
        OmegaConf.save(cfg, target/'configs.yaml')
        for directory in (old, target):
            (directory/'scope_audit.json').write_text('{"steps":60}')
        (old/'status.json').write_text(json.dumps({'status': 'completed'}))
        (old/'mpc_evaluation.json').write_text('{}')
        (old/'mpc_closed_loop_steps.csv').write_text('energy\n1\n')
        return {'relative_directory': 'case',
                'config_sha256': runner.base.digest(old/'configs.yaml'),
                'mpc_config_sha256': 'test'}

    def test_completed_copy_is_independent_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, previous = Path(tmp)/'new', Path(tmp)/'old'
            root.mkdir()
            previous.mkdir()
            prior = self.make_case(previous/'case', root/'case')
            case = {'relative_directory': 'case'}
            runner.reuse_case(root, case, previous, prior)
            self.assertIn('reused_from', case)
            self.assertFalse((root/'case').is_symlink())
            (root/'case/mpc_evaluation.json').write_text('changed')
            self.assertEqual((previous/'case/mpc_evaluation.json').read_text(), '{}')

    def test_reuse_rejects_changed_science_or_incomplete_status(self):
        for incomplete in (False, True):
            with self.subTest(incomplete=incomplete), tempfile.TemporaryDirectory() as tmp:
                root, previous = Path(tmp)/'new', Path(tmp)/'old'
                root.mkdir()
                previous.mkdir()
                prior = self.make_case(previous/'case', root/'case')
                if incomplete:
                    (previous/'case/status.json').write_text('{"status":"running"}')
                else:
                    cfg = OmegaConf.load(root/'case/configs.yaml')
                    cfg.MPC.horizon = 10
                    OmegaConf.save(cfg, root/'case/configs.yaml')
                with self.assertRaises(ValueError):
                    runner.reuse_case(root, {'relative_directory': 'case'}, previous, prior)


if __name__ == '__main__':
    unittest.main()
