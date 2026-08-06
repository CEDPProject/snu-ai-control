from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import pandas as pd

from controllers.evaluator_benchmark import benchmark_evaluator_families
from controllers.total_power_evaluation import (
    evaluate_all_chronological_test_total_power,
)
from data_provider.control_data import (
    ControlData,
    ControlEncoder,
    ControlVariable,
)
from runners.evaluator_benchmark_runner import (
    _candidate_pool_experiments,
    _write_candidate_artifacts,
    _write_evaluator_artifacts,
)


def _data(rows=300):
    encoder = ControlEncoder(
        [ControlVariable('switch', ('OFF', 'ON'), 'OFF')]
    )
    controls = np.zeros((rows, 2), dtype=float)
    controls[np.arange(rows), np.arange(rows) % 2] = 1.0
    disturbances = (15.0 + np.sin(np.arange(rows) / 10.0))[:, None]
    states = np.empty((rows, 2), dtype=float)
    powers = np.empty((rows, 2), dtype=float)
    states[0] = [22.0, 500.0]
    for step in range(rows - 1):
        enabled = controls[step, 1]
        powers[step] = [20.0 + 20.0 * enabled, 5.0 + 55.0 * enabled]
        states[step + 1, 0] = (
            states[step, 0]
            + 0.03 * (disturbances[step, 0] - states[step, 0])
            - 0.08 * enabled
        )
        states[step + 1, 1] = states[step, 1] + 1.0 - 2.0 * enabled
    powers[-1] = [
        20.0 + 20.0 * controls[-1, 1],
        5.0 + 55.0 * controls[-1, 1],
    ]
    return ControlData(
        timestamps=pd.Series(
            pd.date_range('2025-01-01', periods=rows, freq='min')
        ),
        states=states,
        controls=controls,
        disturbances=disturbances,
        powers=powers,
        state_names=('temperature', 'co2'),
        disturbance_names=('outdoor_temperature',),
        power_names=('ventilation_power', 'hvac_power'),
        encoder=encoder,
    )


def _report_model_config():
    return OmegaConf.create({
        'selection_rule': 'simplest_within_target',
        'candidate_families': [
            'action_median',
            'direct_total_extra_trees',
        ],
        'minimum_action_samples': 2,
        'n_estimators': 8,
        'min_samples_leaf': 2,
        'max_features': 1.0,
        'n_jobs': 1,
        'huber_epsilon': 1.35,
        'huber_alpha': 0.0001,
        'huber_max_iter': 100,
    })


def _benchmark_config():
    return OmegaConf.create({
        'DEFAULT': {'seed': 42},
        'MPC': {
            'prediction_horizon': 60,
            'state_constraints': [
                {
                    'name': 'temperature',
                    'lower': 18.0,
                    'upper': 27.0,
                },
                {
                    'name': 'co2',
                    'lower': None,
                    'upper': 1000.0,
                },
            ],
            'affine_models': {'mape_min_actual_power_w': 0.1},
            'evaluation': {
                'closed_loop_steps': 10,
                'stride': 10,
                'lead_time_steps': [1, 10],
            },
            'independent_evaluator': {
                'model_family': 'extra_trees_residual_state',
                'n_estimators': 8,
                'min_samples_leaf': 2,
                'max_features': 1.0,
                'max_iter': 8,
                'n_jobs': 1,
                'minimum_test_power_r2': -10.0,
                'maximum_rollout_power_wape_percent': 1000.0,
                'maximum_constraint_false_safe_rate_percent': 100.0,
                'model_setting_path': None,
                'neural_residual': {},
                'forecasting': {},
            },
            'evaluator_benchmark': {
                'model_families': [
                    'extra_trees_residual_state',
                    'random_forest_residual_state',
                ],
                'validation_max_windows': 2,
                'test_max_windows': 2,
                'evaluate_all_models_on_test': True,
            },
        },
    })


class TotalPowerExperimentTest(unittest.TestCase):
    def test_relaxed_scenario_uses_its_own_holdout(self):
        cfg = OmegaConf.create({
            'DEFAULT': {'seed': 42},
            'DATASET': {'split_rate': [0.8, 0.1, 0.1]},
            'MPC': {
                'affine_models': {'mape_min_actual_power_w': 1.0},
                'candidate_pool_experiments': {
                    'scenarios': [{
                        'name': 'expanded_17',
                        'description': 'relaxed support',
                        'maximum_validation_mape_percent_per_candidate': 100.0,
                        'maximum_candidates': 17,
                        'holdout_overrides': {
                            'minimum_training_samples_per_candidate': 3,
                            'minimum_validation_samples_per_candidate': 2,
                        },
                        'candidate_pool_overrides': {
                            'minimum_validation_mape_samples_per_candidate': 2,
                        },
                    }],
                },
                'evaluation': {
                    'fixed_action_candidate_pool': {},
                    'fixed_optimal_action_holdout': {
                        'total_power_report_model': {},
                    },
                },
            },
        })
        base_holdout = SimpleNamespace(
            metadata={'candidate_count': 12, 'protocol': {}}
        )
        relaxed_holdout = SimpleNamespace(metadata={
            'candidate_count': 17,
            'protocol': {'minimum_validation_samples': 2},
        })
        evaluation_result = {
            'selection': {},
            'test_evaluation': {},
            'primary_metric': {},
        }
        with TemporaryDirectory() as temporary_directory:
            with patch(
                'runners.evaluator_benchmark_runner.'
                'build_fixed_action_episode_holdout',
                return_value=relaxed_holdout,
            ) as build_holdout:
                with patch(
                    'runners.evaluator_benchmark_runner.'
                    'evaluate_fixed_action_candidate_pool_holdout',
                    return_value=(evaluation_result, []),
                ) as evaluate:
                    detailed, _, summary_rows = _candidate_pool_experiments(
                        cfg,
                        object(),
                        base_holdout,
                        Path(temporary_directory),
                    )

        effective_holdout_config = build_holdout.call_args.args[1]
        self.assertEqual(
            effective_holdout_config.minimum_training_samples_per_candidate,
            3,
        )
        self.assertIs(evaluate.call_args.args[1], relaxed_holdout)
        scenario = detailed['expanded_17']['experiment_scenario']
        self.assertEqual(scenario['available_candidate_count'], 17)
        self.assertEqual(
            scenario['candidate_pool_overrides'][
                'minimum_validation_mape_samples_per_candidate'
            ],
            2,
        )
        self.assertEqual(summary_rows[0]['available_candidate_count'], 17)

    def test_all_chronological_test_evaluates_every_test_transition(self):
        data = _data(160)
        result, audit_rows = evaluate_all_chronological_test_total_power(
            data,
            training_rows=np.arange(0, 99),
            validation_rows=np.arange(100, 129),
            test_rows=np.arange(130, 159),
            report_model_config=_report_model_config(),
            minimum_actual_power_w=0.1,
            target_mape_percent=1.95,
            minimum_validation_samples=10,
            minimum_test_samples=10,
            random_state=42,
        )
        self.assertTrue(result['reportable'])
        self.assertEqual(result['split']['test_transition_count'], 29)
        self.assertEqual(len(audit_rows), 29)
        self.assertEqual(
            result['model_selection']['selected_family'],
            'action_median',
        )
        self.assertAlmostEqual(result['primary_metric']['value'], 0.0)

    def test_evaluator_selection_uses_validation_and_reports_total_power(self):
        benchmark = benchmark_evaluator_families(
            _benchmark_config(),
            _data(),
            train_end=180,
            test_start=240,
        )
        self.assertIn(
            benchmark['selected_family'],
            {
                'extra_trees_residual_state',
                'random_forest_residual_state',
            },
        )
        selected = [
            item
            for item in benchmark['models']
            if item['selected_by_validation']
        ]
        self.assertEqual(len(selected), 1)
        for item in benchmark['models']:
            self.assertTrue(item['test_fit_succeeded'])
            self.assertIn('total_power', item['test']['one_step'])
            self.assertIn('total_power', item['test']['recursive'])


    def test_each_experiment_and_evaluator_has_an_independent_folder(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate_result = {
                'experiment_scenario': {'name': 'previous_7'},
                'selection': {
                    'candidates': [
                        {
                            'candidate_id': 'action_001',
                            'action': {'switch': 'ON'},
                            'selected_for_pool': True,
                        }
                    ],
                    'selected_actions': [{'switch': 'ON'}],
                },
            }
            candidate_directory = _write_candidate_artifacts(
                root / '02_candidate_pool_experiments',
                'previous_7',
                candidate_result,
                [{'actual_total_power_w': 100.0}],
            )
            self.assertTrue((candidate_directory / 'evaluation.json').is_file())
            self.assertTrue(
                (candidate_directory / 'candidate_selection.csv').is_file()
            )
            self.assertTrue(
                (candidate_directory / 'test_predictions.csv').is_file()
            )

            benchmark = {
                'selected_family': 'forecasting:TimeXer',
                'models': [
                    {
                        'priority': 0,
                        'requested_family': 'forecasting:TimeXer',
                        'evaluator_label': 'forecasting_sequence:TimeXer',
                        'selected_by_validation': True,
                        'validation_fit_succeeded': False,
                        'validation_failure': 'test fixture',
                        'validation': None,
                        'test_fit_succeeded': False,
                        'test_failure': None,
                        'test': None,
                    }
                ],
            }
            evaluator_root = root / '03_evaluators'
            _write_evaluator_artifacts(evaluator_root, benchmark)
            model_directory = evaluator_root / 'forecasting_TimeXer'
            self.assertTrue((evaluator_root / 'model_comparison.csv').is_file())
            self.assertTrue((model_directory / 'evaluation.json').is_file())
            self.assertTrue((model_directory / 'summary.csv').is_file())

if __name__ == '__main__':
    unittest.main()
