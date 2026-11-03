"""최종 보고서의 층화 paired 비교 부호·재현성·분모 검증."""

import importlib.util
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

path = Path(__file__).resolve().parents[1]/'src/reporting/savings_improvement_final_analysis.py'
spec = importlib.util.spec_from_file_location('savings_final', path)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class SavingsFinalAnalysisTest(unittest.TestCase):
    def test_completed_single_cohort_writes_interim_not_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = []
            violation = {'violation_magnitude_minutes': 0., 'maximum_violation': 0.,
                         'violating_steps': 0}
            policy = {'energy_wh': 1., 'constraint_violations': {'state': violation}}
            ev = {
                'windows': [{'timestamp': '2026-01-01', 'policies': {'mpc': policy, 'recorded': policy}}],
                'policy_evaluation': {
                    'solver': {'control_switch_count': 0, 'fallback_count': 0, 'deadline_miss_count': 0},
                    'constraint_violations': {'mpc': {'state': violation}, 'recorded': {'state': violation}}},
                'quality_assessment': {
                    'model_validation': {'model_validation_passed': True, 'gates': {}},
                    'simulated_policy_acceptance': {'passed': True, 'gates': {}}}}
            for group in ('v3', 'v4', 'v5', 'concat'):
                for env in ('livingroom', 'room1', 'room2'):
                    for arm in ('m3_original', 'm3_b1', 'm3_b1_cem128x5'):
                        relative = f'{group}/{env}/{arm}'
                        d = root/relative
                        d.mkdir(parents=True)
                        OmegaConf.save(OmegaConf.create({
                            'MPC': {'state_constraints': [{'name': 'state', 'penalty': 1}],
                                    'objective': {'switch_penalty': .05}}}), d/'configs.yaml')
                        case = {'group': group, 'environment': env, 'arm': arm, 'seed': 42,
                                'method': 'm3', 'relative_directory': relative,
                                'config_sha256': hashlib.sha256((d/'configs.yaml').read_bytes()).hexdigest()}
                        cases.append(case)
                        if group == 'concat':
                            continue  # 아직 실행하지 않은 concat을 분석해서는 안 된다.
                        (d/'status.json').write_text(json.dumps({'status': 'completed', 'wall_seconds': 1}))
                        (d/'scope_audit.json').write_text(json.dumps({
                            'window_starts': [1], 'evaluated_windows': 1, 'expected_policy_steps': 1}))
                        (d/'mpc_evaluation.json').write_text(json.dumps(ev))
                        (d/'mpc_closed_loop_steps.csv').write_text(
                            'window_timestamp,action_control,solve_seconds,end_to_end_step_seconds\n'
                            '2026-01-01,1,0.1,0.2\n')
            manifest = {'status': 'running', 'experiment_kind': 'm3_full_scope_expansion',
                        'cases': cases, 'source_sha256': {}, 'input_sources': []}
            (root/'manifest.json').write_text(json.dumps(manifest))
            data = SimpleNamespace(
                controls=np.array([[1., 0.]]),
                encoder=SimpleNamespace(groups=['control'], encode_action=lambda *args: np.array([1., 0.])))
            before = (root/'manifest.json').read_bytes()
            with patch('data_provider.control_data.load_control_data', return_value=data), \
                    contextlib.redirect_stdout(io.StringIO()):
                analysis.main(root, cohort='single_versions')
            out = root/'analysis/interim_single_versions'
            metadata = json.loads((out/'analysis_metadata.json').read_text())
            self.assertEqual(metadata['selected_predeclared_cohort'], 'single_versions')
            self.assertEqual(metadata['case_count'], 27)
            self.assertEqual(metadata['run_status_at_analysis_start'], 'running')
            self.assertFalse((root/'analysis/final').exists())
            self.assertEqual((root/'manifest.json').read_bytes(), before)
            # 완결되지 않은 단일버전 cohort도 중간 집계에서 거부한다.
            (root/cases[0]['relative_directory']/'status.json').write_text('{"status":"running"}')
            with patch('data_provider.control_data.load_control_data', return_value=data):
                with self.assertRaisesRegex(ValueError, 'Incomplete case'):
                    analysis.main(root, cohort='single_versions')

    def test_running_expansion_requires_explicit_complete_cohort_selection(self):
        cases = [{'group': g, 'environment': e, 'arm': a}
                 for g in ('v3', 'v4', 'v5', 'concat')
                 for e in ('livingroom', 'room1', 'room2')
                 for a in ('m3_original', 'm3_b1', 'm3_b1_cem128x5')]
        manifest = {'experiment_kind': 'm3_full_scope_expansion',
                    'status': 'running', 'cases': cases}
        with self.assertRaisesRegex(ValueError, 'completed cohort'):
            analysis.analysis_scope(manifest)
        selected, cohorts = analysis.analysis_scope(manifest, 'single_versions')
        self.assertEqual(len(selected), 27)
        self.assertEqual(len(cohorts['single_versions']), 9)
        self.assertTrue(all(c['group'] != 'concat' for c in selected))
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            analysis.analysis_scope(manifest, 'chosen_best_cells')
        manifest['cases'] = cases[:-1]
        with self.assertRaisesRegex(ValueError, 'all36'):
            analysis.analysis_scope(manifest, 'single_versions')

    def test_stratified_sign_and_fixed_cell_weight(self):
        a = np.tile([10., 8., 100., 50., 111., 59.], (2, 1))
        b = np.tile([100., 80., 200., 100., 301., 181.], (3, 1))
        result = analysis.stratified_paired([a, b], samples=100)
        self.assertEqual(result['windows'], 5)
        self.assertAlmostEqual(result['energy_b_vs_a_percent'], -20)
        self.assertAlmostEqual(result['energy_b_vs_a_percent_ci_low'], -20)
        self.assertAlmostEqual(result['soft_b_minus_a'], -400)
        self.assertAlmostEqual(result['objective_b_minus_a'], -464)
        self.assertEqual(result, analysis.stratified_paired([a, b], samples=100))

    def test_identical_conditions_have_zero_difference(self):
        result = analysis.stratified_paired([np.array([
            [10, 10, 20, 20, 31, 31], [20, 20, 30, 30, 51, 51],
        ])], samples=100)
        self.assertEqual(result['energy_b_vs_a_percent'], 0)
        self.assertEqual(result['objective_b_minus_a_ci_high'], 0)

    def test_zero_energy_denominator_is_rejected(self):
        with self.assertRaises(ValueError):
            analysis.stratified_paired([np.zeros((2, 6))], samples=100)


if __name__ == '__main__':
    unittest.main()
