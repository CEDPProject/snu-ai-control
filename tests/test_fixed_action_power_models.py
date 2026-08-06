import json
import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from controllers.candidate_pool_evaluation import (
    evaluate_fixed_action_candidate_pool_holdout,
    evaluate_fixed_action_candidate_pool_top_k_sweep,
)
from controllers.fixed_action_evaluation import (
    build_fixed_action_episode_holdout,
    evaluate_fixed_optimal_action_holdout,
)
from controllers.fixed_action_power_models import (
    FixedActionTotalPowerModel,
)
from data_provider.control_data import ControlData, ControlEncoder, ControlVariable


class FixedActionPowerModelContractTest(unittest.TestCase):
    @staticmethod
    def _data():
        encoder = ControlEncoder(
            [ControlVariable('control', ('OFF', 'ON'), 'OFF')]
        )
        actions = []
        for episode_number in range(20):
            action = 'OFF' if episode_number % 2 == 0 else 'ON'
            actions.extend([action] * 4)
        controls = np.vstack(
            [encoder.encode_action({'control': action}) for action in actions]
        )
        row_count = len(actions)
        return ControlData(
            timestamps=pd.Series(
                pd.date_range('2025-01-01', periods=row_count, freq='min')
            ),
            states=np.zeros((row_count, 1)),
            controls=controls,
            disturbances=np.empty((row_count, 0)),
            powers=np.tile(np.array([[10.0, 20.0]]), (row_count, 1)),
            state_names=('state',),
            disturbance_names=(),
            power_names=('ventilation_power_w', 'hvac_power_w'),
            encoder=encoder,
        )

    @staticmethod
    def _holdout_config():
        return OmegaConf.create(
            {
                'top_k_actions': 2,
                'minimum_training_samples_per_candidate': 8,
                'minimum_training_episodes_per_candidate': 3,
                'minimum_validation_samples_per_candidate': 3,
                'minimum_validation_episodes_per_candidate': 3,
                'minimum_test_exact_matches': 8,
                'minimum_test_episodes_per_candidate': 3,
            }
        )

    @staticmethod
    def _power_model_config():
        return {
            'enabled': True,
            'target_mape_percent': 1.95,
            'selection_rule': 'simplest_within_target',
            'candidate_families': ['action_median'],
            'minimum_validation_mape_samples': 3,
            'minimum_action_samples': 5,
            'n_estimators': 20,
            'min_samples_leaf': 2,
            'max_features': 1.0,
            'huber_epsilon': 1.35,
            'huber_alpha': 0.0001,
            'huber_max_iter': 100,
            'n_jobs': 1,
        }

    @staticmethod
    def _candidate_pool_config():
        return {
            'target_mape_percent': 1.95,
            'selection_model_family': 'action_median',
            'maximum_validation_mape_percent_per_candidate': 1.95,
            'minimum_validation_mape_samples_per_candidate': 3,
            'minimum_test_mape_samples_per_candidate': 8,
            'minimum_pooled_test_mape_samples': 8,
            'maximum_candidates': 0,
        }

    def test_action_median_uses_global_train_fallback_for_unseen_action(self):
        data = self._data()
        off_rows = np.flatnonzero(data.controls[:, 0] > 0.5)[:8]
        model = FixedActionTotalPowerModel.fit(
            data,
            off_rows,
            family='action_median',
            minimum_action_samples=5,
            n_estimators=20,
            min_samples_leaf=2,
            max_features=1.0,
            huber_epsilon=1.35,
            huber_alpha=0.0001,
            huber_max_iter=100,
            n_jobs=1,
            random_state=42,
        )
        on_control = data.encoder.encode_action({'control': 'ON'})[None, :]
        prediction = model.predict(
            np.zeros((1, 1)),
            on_control,
            np.empty((1, 0)),
        )

        self.assertEqual(prediction.shape, (1, 1))
        self.assertAlmostEqual(prediction[0, 0], 30.0)
        self.assertEqual(
            model.to_metadata(on_control[0])['unseen_or_sparse_action_fallback'],
            'global training median',
        )

    def test_official_mape_uses_validation_selected_direct_total_model_only(self):
        data = self._data()
        holdout = build_fixed_action_episode_holdout(
            data,
            self._holdout_config(),
            split_rate=(0.8, 0.1, 0.1),
        )

        class SelectionEvaluator:
            def predict_state(self, states, controls, disturbances):
                return np.zeros((len(states), 1))

            def predict_power(self, states, controls, disturbances):
                is_on = controls[:, 1] > 0.5
                return np.where(
                    is_on[:, None],
                    np.array([[4.0, 6.0]]),
                    np.array([[20.0, 20.0]]),
                )

        class FrozenComponentEvaluator:
            def predict_state(self, states, controls, disturbances):
                return np.zeros((len(states), 1))

            def predict_power(self, states, controls, disturbances):
                return np.tile(np.array([[11.0, 22.0]]), (len(states), 1))

        result, audit_rows = evaluate_fixed_optimal_action_holdout(
            data,
            holdout,
            SelectionEvaluator(),
            FrozenComponentEvaluator(),
            state_constraints=[
                {'name': 'state', 'lower': -1.0, 'upper': 1.0}
            ],
            minimum_predicted_state_satisfaction_percent=100.0,
            allow_soft_constraint_fallback=True,
            minimum_test_exact_matches=8,
            minimum_test_episodes=3,
            minimum_test_mape_samples=8,
            mape_min_actual_power_w=0.1,
            selection_evaluator_name='train-only-selector',
            frozen_report_evaluator_name='legacy-component-reporter',
            total_power_report_config=self._power_model_config(),
            total_power_random_state=42,
            target_mape_percent=1.95,
        )

        self.assertTrue(result['reportable'])
        self.assertEqual(result['selection']['selected_action'], {'control': 'ON'})
        self.assertAlmostEqual(result['primary_metric']['value'], 0.0)
        self.assertTrue(result['primary_metric']['target_achieved'])
        self.assertEqual(
            result['total_power_model_selection']['selected_family'],
            'action_median',
        )
        self.assertFalse(
            result['total_power_model_selection'][
                'test_targets_used_to_select_family'
            ]
        )
        self.assertAlmostEqual(
            result['total_power_model_selection'][
                'legacy_component_sum_test_mape_percent'
            ],
            10.0,
        )
        self.assertTrue(
            all(
                row['official_total_power_model_family'] == 'action_median'
                and row['predicted_total_hvac_power_w'] == 30.0
                and row['legacy_component_sum_predicted_total_hvac_power_w']
                == 33.0
                for row in audit_rows
            )
        )
        json.dumps(result, allow_nan=False)

    def test_candidate_pool_reports_pooled_and_per_action_mape(self):
        data = self._data()
        holdout = build_fixed_action_episode_holdout(
            data,
            self._holdout_config(),
            split_rate=(0.8, 0.1, 0.1),
        )

        result, audit_rows = evaluate_fixed_action_candidate_pool_holdout(
            data,
            holdout,
            config=self._candidate_pool_config(),
            total_power_report_config=self._power_model_config(),
            mape_min_actual_power_w=0.1,
            random_state=42,
        )

        self.assertTrue(result['reportable'])
        self.assertEqual(result['selection']['candidate_count'], 2)
        self.assertEqual(result['selection']['selected_candidate_count'], 2)
        self.assertAlmostEqual(result['primary_metric']['value'], 0.0)
        self.assertAlmostEqual(
            result['test_evaluation']['equal_action_macro_mape_percent'],
            0.0,
        )
        self.assertAlmostEqual(
            result['test_evaluation']['worst_action_mape_percent'],
            0.0,
        )
        self.assertAlmostEqual(
            result['test_evaluation']['action_target_pass_rate_percent'],
            100.0,
        )
        self.assertEqual(
            len(audit_rows),
            result['test_evaluation']['pool_exact_match_count'],
        )
        self.assertFalse(
            result['protocol'][
                'test_power_targets_used_for_candidate_or_model_selection'
            ]
        )
        json.dumps(result, allow_nan=False)

    def test_candidate_pool_excludes_action_that_fails_validation_only(self):
        original = self._data()
        holdout = build_fixed_action_episode_holdout(
            original,
            self._holdout_config(),
            split_rate=(0.8, 0.1, 0.1),
        )
        powers = original.powers.copy()
        on_validation_rows = holdout.validation_rows[
            original.controls[holdout.validation_rows, 1] > 0.5
        ]
        powers[on_validation_rows] = np.array([20.0, 40.0])
        data = ControlData(
            timestamps=original.timestamps,
            states=original.states,
            controls=original.controls,
            disturbances=original.disturbances,
            powers=powers,
            state_names=original.state_names,
            disturbance_names=original.disturbance_names,
            power_names=original.power_names,
            encoder=original.encoder,
        )

        result, _ = evaluate_fixed_action_candidate_pool_holdout(
            data,
            holdout,
            config=self._candidate_pool_config(),
            total_power_report_config=self._power_model_config(),
            mape_min_actual_power_w=0.1,
            random_state=42,
        )

        self.assertTrue(result['reportable'])
        self.assertEqual(result['selection']['selected_candidate_count'], 1)
        selected = [
            candidate
            for candidate in result['selection']['candidates']
            if candidate['selected_for_pool']
        ]
        rejected = [
            candidate
            for candidate in result['selection']['candidates']
            if not candidate['selected_for_pool']
        ]
        self.assertEqual(selected[0]['action'], {'control': 'OFF'})
        self.assertIn(
            'validation_mape_above_pool_threshold',
            rejected[0]['rejection_reasons'],
        )
        self.assertAlmostEqual(result['primary_metric']['value'], 0.0)


    def test_top_k_sweep_selects_largest_validation_passing_k(self):
        data = self._data()
        holdout = build_fixed_action_episode_holdout(
            data,
            self._holdout_config(),
            split_rate=(0.8, 0.1, 0.1),
        )
        sweep_config = {
            **self._candidate_pool_config(),
            'minimum_k': 1,
            'maximum_k': 2,
            'step': 1,
            'k_selection_rule': 'largest_k_within_validation_target',
            'fallback_rule': 'minimum_validation_mape',
            'minimum_pooled_validation_mape_samples': 1,
        }

        result, audit_rows = (
            evaluate_fixed_action_candidate_pool_top_k_sweep(
                data,
                holdout,
                config=sweep_config,
                total_power_report_config=self._power_model_config(),
                mape_min_actual_power_w=0.1,
                random_state=42,
            )
        )

        self.assertTrue(result['reportable'])
        self.assertEqual([row['k'] for row in result['curve']], [1, 2])
        self.assertEqual(result['official_selection']['selected_k'], 2)
        self.assertFalse(
            result['official_selection']['test_targets_used_for_k_selection']
        )
        self.assertTrue(
            result['protocol']['same_frozen_report_model_used_for_every_k']
        )
        self.assertTrue(result['curve'][1]['selected_by_validation'])
        self.assertEqual(
            result['curve'][0]['test_interpretation'],
            'sensitivity_only',
        )
        self.assertEqual(
            result['curve'][1]['test_interpretation'],
            'official_selected_k',
        )
        self.assertAlmostEqual(result['primary_metric']['value'], 0.0)
        self.assertTrue(
            all(
                row['included_in_official_selected_k']
                for row in audit_rows
            )
        )
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
