import json
import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from controllers.candidate_pool_evaluation import (
    evaluate_fixed_action_candidate_pool_holdout,
    evaluate_fixed_action_candidate_pool_top_k_sweep,
)
from controllers.action_stratified_holdout import (
    build_action_stratified_row_holdout,
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
        component_prediction = model.predict_components(
            np.zeros((1, 1)),
            on_control,
            np.empty((1, 0)),
        )
        np.testing.assert_allclose(component_prediction, [[10.0, 20.0]])
        self.assertEqual(
            model.to_metadata(on_control[0])['unseen_or_sparse_action_fallback'],
            'global training median',
        )

    def test_report_model_input_feature_groups_are_explicit_and_selectable(self):
        data = self._data()
        model = FixedActionTotalPowerModel.fit(
            data,
            np.arange(0, 40),
            family='direct_total_extra_trees',
            minimum_action_samples=5,
            n_estimators=20,
            min_samples_leaf=2,
            max_features=1.0,
            huber_epsilon=1.35,
            huber_alpha=0.0001,
            huber_max_iter=100,
            n_jobs=1,
            random_state=42,
            input_features={
                'current_state': False,
                'action': True,
                'current_disturbance': False,
            },
        )
        metadata = model.to_metadata()
        self.assertEqual(metadata['input_features']['enabled_groups'], ['action'])
        self.assertEqual(metadata['input_features']['feature_dimension'], 2)
        components = model.predict_components(
            np.zeros((2, 1)),
            data.controls[:2],
            np.empty((2, 0)),
        )
        self.assertEqual(components.shape, (2, 2))
        np.testing.assert_allclose(
            model.predict(
                np.zeros((2, 1)),
                data.controls[:2],
                np.empty((2, 0)),
            ),
            components.sum(axis=1, keepdims=True),
        )
        action_disabled_model = FixedActionTotalPowerModel.fit(
            data,
            np.arange(0, 40),
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
            input_features={
                'current_state': True,
                'action': False,
                'current_disturbance': False,
            },
        )
        np.testing.assert_allclose(
            action_disabled_model.predict_components(
                np.zeros((2, 1)),
                data.controls[:2],
                np.empty((2, 0)),
            )[0],
            action_disabled_model.predict_components(
                np.zeros((2, 1)),
                data.controls[1:3],
                np.empty((2, 0)),
            )[0],
        )
        with self.assertRaises(ValueError):
            FixedActionTotalPowerModel.fit(
                data,
                np.arange(0, 40),
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
                input_features={
                    'current_state': False,
                    'action': False,
                    'current_disturbance': False,
                },
            )

    def test_two_stage_hvac_report_model_separates_low_and_active_power(self):
        encoder = ControlEncoder(
            [ControlVariable('control', ('OFF', 'ON'), 'OFF')]
        )
        actions = ['OFF'] * 16 + ['ON'] * 16
        controls = np.vstack(
            [encoder.encode_action({'control': action}) for action in actions]
        )
        states = np.arange(len(actions), dtype=float)[:, None]
        hvac_power = np.concatenate(
            [
                np.zeros(16),
                35.0 + np.arange(16, dtype=float) * 0.25,
            ]
        )
        data = ControlData(
            timestamps=pd.Series(
                pd.date_range('2025-01-01', periods=len(actions), freq='min')
            ),
            states=states,
            controls=controls,
            disturbances=np.empty((len(actions), 0)),
            powers=np.column_stack(
                [np.full(len(actions), 7.0), hvac_power]
            ),
            state_names=('state',),
            disturbance_names=(),
            power_names=('ventilation_power_w', 'hvac_power_w'),
            encoder=encoder,
        )
        model = FixedActionTotalPowerModel.fit(
            data,
            np.arange(len(actions) - 1),
            family='component_hvac_two_stage_extra_trees',
            minimum_action_samples=2,
            n_estimators=20,
            min_samples_leaf=1,
            max_features=1.0,
            huber_epsilon=1.35,
            huber_alpha=0.0001,
            huber_max_iter=100,
            n_jobs=1,
            random_state=42,
            hvac_low_power_threshold_w=0.1,
            hvac_minimum_active_samples=2,
        )
        off_control = encoder.encode_action({'control': 'OFF'})[None, :]
        on_control = encoder.encode_action({'control': 'ON'})[None, :]
        components = model.predict_components(
            np.array([[0.0], [20.0]]),
            np.vstack([off_control, on_control]),
            np.empty((2, 0)),
        )

        self.assertEqual(components.shape, (2, 2))
        self.assertLessEqual(components[0, 1], 0.1)
        self.assertGreater(components[1, 1], 30.0)
        np.testing.assert_allclose(
            model.predict(
                np.array([[0.0], [20.0]]),
                np.vstack([off_control, on_control]),
                np.empty((2, 0)),
            ),
            components.sum(axis=1, keepdims=True),
        )
        two_stage_metadata = model.to_metadata()['component_models']['hvac_power_w']
        self.assertTrue(two_stage_metadata['classifier_fitted'])
        self.assertTrue(two_stage_metadata['conditional_regressor_fitted'])
        self.assertEqual(
            two_stage_metadata['fallback_mode'],
            'two_stage_classifier_and_conditional_regressor',
        )

    def test_two_stage_hvac_report_model_has_deterministic_one_class_fallbacks(self):
        encoder = ControlEncoder(
            [ControlVariable('control', ('OFF', 'ON'), 'OFF')]
        )
        actions = ['OFF', 'ON'] * 12
        controls = np.vstack(
            [encoder.encode_action({'control': action}) for action in actions]
        )

        def make_data(hvac_power):
            return ControlData(
                timestamps=pd.Series(
                    pd.date_range(
                        '2025-01-01', periods=len(actions), freq='min'
                    )
                ),
                states=np.arange(len(actions), dtype=float)[:, None],
                controls=controls,
                disturbances=np.empty((len(actions), 0)),
                powers=np.column_stack(
                    [np.full(len(actions), 7.0), hvac_power]
                ),
                state_names=('state',),
                disturbance_names=(),
                power_names=('ventilation_power_w', 'hvac_power_w'),
                encoder=encoder,
            )

        kwargs = {
            'minimum_action_samples': 2,
            'n_estimators': 12,
            'min_samples_leaf': 1,
            'max_features': 1.0,
            'huber_epsilon': 1.35,
            'huber_alpha': 0.0001,
            'huber_max_iter': 100,
            'n_jobs': 1,
            'random_state': 42,
            'hvac_low_power_threshold_w': 0.1,
            'hvac_minimum_active_samples': 2,
        }
        all_low = FixedActionTotalPowerModel.fit(
            make_data(np.zeros(len(actions))),
            np.arange(len(actions) - 1),
            family='component_hvac_two_stage_extra_trees',
            **kwargs,
        )
        all_active = FixedActionTotalPowerModel.fit(
            make_data(np.full(len(actions), 23.0)),
            np.arange(len(actions) - 1),
            family='component_hvac_two_stage_extra_trees',
            **kwargs,
        )
        prediction_inputs = (
            np.array([[1.0], [2.0]]),
            controls[:2],
            np.empty((2, 0)),
        )
        low_prediction = all_low.predict_components(*prediction_inputs)[:, 1]
        active_prediction = all_active.predict_components(*prediction_inputs)[:, 1]

        np.testing.assert_allclose(low_prediction, [0.0, 0.0])
        np.testing.assert_allclose(active_prediction, [23.0, 23.0])
        self.assertEqual(
            all_low.to_metadata()['component_models']['hvac_power_w'][
                'fallback_mode'
            ],
            'one_class_low_power_median',
        )
        self.assertEqual(
            all_active.to_metadata()['component_models']['hvac_power_w'][
                'fallback_mode'
            ],
            'one_class_active_conditional_regressor',
        )

    def test_action_stratified_row_holdout_balances_each_action_without_targets(self):
        data = self._data()
        holdout = build_action_stratified_row_holdout(
            data,
            self._holdout_config(),
            split_rate=(0.8, 0.1, 0.1),
            random_state=42,
        )
        changed_outcomes = ControlData(
            timestamps=data.timestamps.copy(),
            states=np.full_like(data.states, 999.0),
            controls=data.controls.copy(),
            disturbances=data.disturbances.copy(),
            powers=np.full_like(data.powers, 12345.0),
            state_names=data.state_names,
            disturbance_names=data.disturbance_names,
            power_names=data.power_names,
            encoder=data.encoder,
        )
        changed_holdout = build_action_stratified_row_holdout(
            changed_outcomes,
            self._holdout_config(),
            split_rate=(0.8, 0.1, 0.1),
            random_state=42,
        )

        split_rows = [
            set(holdout.train_rows.tolist()),
            set(holdout.validation_rows.tolist()),
            set(holdout.test_rows.tolist()),
        ]
        self.assertFalse(split_rows[0] & split_rows[1])
        self.assertFalse(split_rows[0] & split_rows[2])
        self.assertFalse(split_rows[1] & split_rows[2])
        self.assertEqual(
            sum(len(rows) for rows in split_rows),
            len(data.valid_transition_rows(np.arange(len(data.states) - 1))),
        )
        self.assertEqual(holdout.metadata['candidate_count'], 2)
        self.assertTrue(
            holdout.metadata['protocol']['episode_overlap_between_splits_allowed']
        )
        self.assertFalse(
            holdout.metadata['protocol']['temporal_generalization_claim']
        )
        self.assertTrue(
            holdout.metadata['protocol'][
                'minimum_one_row_per_split_when_possible'
            ]
        )
        self.assertEqual(
            holdout.metadata['protocol']['outcome_columns_used_for_assignment'],
            [],
        )
        np.testing.assert_array_equal(
            holdout.train_rows,
            changed_holdout.train_rows,
        )
        np.testing.assert_array_equal(
            holdout.validation_rows,
            changed_holdout.validation_rows,
        )
        np.testing.assert_array_equal(
            holdout.test_rows,
            changed_holdout.test_rows,
        )
        for candidate in holdout.metadata['candidates']:
            self.assertGreaterEqual(
                candidate['support']['train']['sample_count'],
                8,
            )
            self.assertGreaterEqual(
                candidate['support']['validation']['sample_count'],
                3,
            )
            self.assertGreaterEqual(
                candidate['support']['test']['sample_count'],
                8,
            )

    def test_action_stratified_row_holdout_keeps_three_row_rare_action_in_all_splits(self):
        encoder = ControlEncoder(
            [ControlVariable('control', ('A', 'B', 'C'), 'A')]
        )
        actions = ['A'] * 3 + ['B'] * 20 + ['C'] * 20
        controls = np.vstack(
            [encoder.encode_action({'control': action}) for action in actions]
        )
        data = ControlData(
            timestamps=pd.Series(
                pd.date_range('2025-01-01', periods=len(actions), freq='min')
            ),
            states=np.zeros((len(actions), 1)),
            controls=controls,
            disturbances=np.empty((len(actions), 0)),
            powers=np.tile(np.array([[10.0, 20.0]]), (len(actions), 1)),
            state_names=('state',),
            disturbance_names=(),
            power_names=('ventilation_power_w', 'hvac_power_w'),
            encoder=encoder,
        )
        config = OmegaConf.create(
            {
                'top_k_actions': 0,
                'minimum_training_samples_per_candidate': 5,
                'minimum_validation_samples_per_candidate': 3,
                'minimum_test_exact_matches': 5,
            }
        )
        holdout = build_action_stratified_row_holdout(
            data,
            config,
            split_rate=(0.8, 0.1, 0.1),
            random_state=42,
        )
        rare_action = next(
            row
            for row in holdout.metadata['allocation_audit']
            if row['control_control'] == 'A'
        )
        self.assertEqual(rare_action['total_sample_count'], 3)
        self.assertEqual(rare_action['train_sample_count'], 1)
        self.assertEqual(rare_action['validation_sample_count'], 1)
        self.assertEqual(rare_action['test_sample_count'], 1)
        self.assertFalse(rare_action['candidate_eligible'])

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
