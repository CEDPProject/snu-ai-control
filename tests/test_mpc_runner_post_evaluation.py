import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from runners.mpc_runner import _run_counterfactual_input_ablation_if_enabled


def _config(run_after_mpc):
    return OmegaConf.create({
        'MPC': {
            'counterfactual_report_input_ablation': {
                'run_after_mpc': run_after_mpc,
                'reference_mpc_result_dir': None,
            },
            'report_model_input_ablation': {
                'cases': [
                    {'name': 'state_action_disturbance'},
                    {'name': 'state_action'},
                    {'name': 'action_only'},
                ],
            },
        },
    })


class MpcRunnerPostEvaluationTest(unittest.TestCase):
    def test_enabled_sweep_uses_completed_mpc_result_as_immutable_reference(self):
        with TemporaryDirectory() as temporary_directory:
            mpc_result = Path(temporary_directory) / 'mpc_result'
            sweep_result = Path(temporary_directory) / 'input_sweep'
            mpc_result.mkdir()
            sweep_result.mkdir()
            sweep_summary = {
                'result_directory': str(sweep_result),
                'reference_mpc_run': {
                    'counterfactual_transition_count': 12,
                    'mpc_action_trace_sha256': 'trace-hash',
                },
                'case_results': {
                    'state_action_disturbance': {},
                    'state_action': {},
                    'action_only': {},
                },
            }
            with patch(
                'runners.counterfactual_report_input_ablation_runner.'
                'run_counterfactual_report_input_ablation',
                return_value=sweep_summary,
            ) as run_sweep:
                sidecar = _run_counterfactual_input_ablation_if_enabled(
                    _config(True),
                    mpc_result,
                )

            self.assertEqual(
                run_sweep.call_args.args[0]
                .MPC.counterfactual_report_input_ablation.reference_mpc_result_dir,
                str(mpc_result),
            )
            self.assertEqual(
                sidecar['case_names'],
                ['action_only', 'state_action', 'state_action_disturbance'],
            )
            with (mpc_result / 'mpc_counterfactual_report_input_ablation_manifest.json').open(
                encoding='utf-8'
            ) as stream:
                persisted = json.load(stream)
            self.assertEqual(persisted, sidecar)

    def test_disabled_sweep_creates_no_sidecar(self):
        with TemporaryDirectory() as temporary_directory:
            result_dir = Path(temporary_directory)
            self.assertIsNone(
                _run_counterfactual_input_ablation_if_enabled(
                    _config(False),
                    result_dir,
                )
            )
            self.assertFalse(
                (result_dir / 'mpc_counterfactual_report_input_ablation_manifest.json')
                .exists()
            )


if __name__ == '__main__':
    unittest.main()
