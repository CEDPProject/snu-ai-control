import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from controllers.action_crossfit import (
    align_boundaries_to_control_episodes,
    chronological_action_split_support,
    evaluate_mpc_selected_action_crossfit,
)
from controllers.evaluation import (
    constraint_prediction_metrics,
    evaluate_observed_control_combinations,
    paired_bootstrap_energy_reduction,
    run_receding_horizon_policy,
)
from controllers.fixed_action_evaluation import (
    build_fixed_action_episode_holdout,
    evaluate_fixed_optimal_action_holdout,
)
from controllers.mixed_integer_mpc import MixedIntegerMPC
from controllers.pipeline import _training_control_support
from data_provider.control_data import (
    ControlData,
    ControlEncoder,
    ControlVariable,
    consolidate_control_categories,
)
from data_provider.build_dataset import HVACDataset
from dynamics.affine import (
    AffinePowerModel,
    AffineStateModel,
    power_mape_metrics,
)
from dynamics.evaluator_base import TransitionPredictions
from models.common import target_channel_indices
from runners.action_history_ablation_runner import (
    _scenario_descriptors,
    _lookbacks,
    _normalized_recency_weights,
    _scaled_rolling_change_budget,
    _scenario_cfg as history_ablation_scenario_cfg,
)
from utils.metrics import Kweather_metric, cal_metric
from utils.tools import EarlyStopping, create_lr_scheduler


class ActionHistoryAblationContractTest(unittest.TestCase):
    def test_lookback_sweep_keeps_penalty_mass_and_factual_boundary(self):
        """L changes history duration, not the total regularization mass."""
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[1]
            / 'src/configs/default_setting.yaml'
        )
        self.assertEqual(_lookbacks(cfg), [0, 1, 2, 5, 10, 30])
        expected_budgets = {0: None, 1: 1, 2: 2, 5: 3, 10: 6, 30: 18}
        for lookback, expected_budget in expected_budgets.items():
            weights = _normalized_recency_weights(cfg, lookback)
            self.assertEqual(len(weights), lookback)
            if lookback:
                self.assertAlmostEqual(sum(weights), 3.0)
                self.assertGreaterEqual(weights[0], weights[-1])
            self.assertEqual(
                _scaled_rolling_change_budget(cfg, lookback), expected_budget
            )

        factual = history_ablation_scenario_cfg(cfg, 0)
        self.assertFalse(factual.MPC.objective.action_history.enabled)
        self.assertEqual(
            factual.MPC.objective.maximum_changed_control_variables_per_step, 0
        )
        self.assertEqual(
            factual.MPC.evaluation.mpc_counterfactual_total_power_report
            .first_action_reference,
            'recorded_current_action',
        )

        smoothed = history_ablation_scenario_cfg(cfg, 5)
        self.assertTrue(smoothed.MPC.objective.action_history.enabled)
        self.assertEqual(smoothed.MPC.objective.action_history.lookback_steps, 5)
        self.assertEqual(
            smoothed.MPC.objective.action_history.rolling_change_budget, 3
        )
        self.assertEqual(
            smoothed.MPC.evaluation.mpc_counterfactual_total_power_report
            .first_action_reference,
            'recorded_previous_action',
        )

    def test_history_constraint_matrix_exposes_k_group_mode_and_penalty(self):
        """L>0 points carry every requested regularization control explicitly."""
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[1]
            / 'src/configs/default_setting.yaml'
        )
        cfg = OmegaConf.merge(cfg, {
            'MPC': {'action_history_ablation': {
                'lookback_steps': [0, 1],
                'hard_distances': [1, 2],
                'history_group_modes': ['per_control_variable'],
                'history_penalties': [0.5, 2.0],
                'per_group_rolling_change_budget_reference_values': {
                    'equipment': None,
                    'per_control_variable': {'환기전원': 1},
                },
            }},
        })

        descriptors = _scenario_descriptors(cfg)
        self.assertEqual(len(descriptors), 5)  # one factual + 2 K × 2 penalties
        self.assertEqual(descriptors[0]['scenario_id'], 'l00_factual')
        constrained = descriptors[1:]
        self.assertEqual(
            {item['hard_distance_k'] for item in constrained}, {1, 2}
        )
        self.assertEqual(
            {item['history_penalty'] for item in constrained}, {0.5, 2.0}
        )
        self.assertEqual(
            {item['history_group_mode'] for item in constrained},
            {'per_control_variable'},
        )
        self.assertTrue(all(
            item['per_group_rolling_change_budgets'] == {'환기전원': 1}
            for item in constrained
        ))

        scenario = history_ablation_scenario_cfg(
            cfg,
            1,
            hard_distance=1,
            history_group_mode='per_control_variable',
            history_penalty=2.0,
            per_group_rolling_change_budgets={'환기전원': 1},
        )
        self.assertEqual(
            scenario.MPC.objective.maximum_changed_control_variables_per_step,
            1,
        )
        self.assertIsNone(scenario.MPC.objective.action_history.groups)
        self.assertEqual(scenario.MPC.objective.action_history.penalty, 2.0)
        self.assertEqual(
            scenario.MPC.objective.action_history.per_group_rolling_change_budgets,
            {'환기전원': 1},
        )

    def test_per_group_only_history_budget_can_disable_aggregate_budget(self):
        """A variable-specific experiment need not retain equipment aggregate caps."""
        cfg = OmegaConf.load(
            Path(__file__).resolve().parents[1]
            / 'src/configs/default_setting.yaml'
        )
        cfg = OmegaConf.merge(cfg, {
            'MPC': {'action_history_ablation': {
                'rolling_change_budget_reference_value': None,
            }},
        })
        self.assertIsNone(_scaled_rolling_change_budget(cfg, 1))
        scenario = history_ablation_scenario_cfg(
            cfg,
            1,
            hard_distance=1,
            history_group_mode='per_control_variable',
            history_penalty=2.0,
            per_group_rolling_change_budgets={'환기전원': 1},
        )
        self.assertIsNone(
            scenario.MPC.objective.action_history.rolling_change_budget
        )


class TargetChannelContractTest(unittest.TestCase):
    def test_target_is_selected_by_name_instead_of_last_position(self):
        configs = {
            'METADATA': {
                'input_numerical_features': ['first', 'power', 'last'],
                'target_features': ['power'],
                'feature_types': {'n_cat': 2},
            }
        }
        self.assertEqual(target_channel_indices(configs, 4), (9,))


class SegmentedDatasetContractTest(unittest.TestCase):
    def test_forecast_dataset_does_not_cross_segment_boundaries(self):
        data = np.arange(10, dtype=float).reshape(-1, 1)
        timestamps = pd.date_range('2025-01-01', periods=10, freq='min')
        segment_ids = np.array([0] * 5 + [1] * 5)

        dataset = HVACDataset(
            data=data,
            timestamps=timestamps,
            seq_len=3,
            pred_len=2,
            stride_len=1,
            features=['power'],
            target_features=['power'],
            numerical_feature=['power'],
            segment_ids=segment_ids,
            prevent_cross_segment_windows=True,
        )

        self.assertEqual(dataset.start_indices.tolist(), [0, 5])
        for start in dataset.start_indices:
            window_segments = segment_ids[start : start + dataset.total_len]
            self.assertEqual(len(set(window_segments.tolist())), 1)

    def test_mpc_transition_mask_requires_same_segment(self):
        encoder = ControlEncoder(
            [ControlVariable('control', ('OFF', 'ON'), 'OFF')]
        )
        controls = np.vstack(
            [encoder.encode_action({'control': 'OFF'}) for _ in range(4)]
        )
        data = ControlData(
            timestamps=pd.Series(
                pd.date_range('2025-01-01', periods=4, freq='min')
            ),
            states=np.zeros((4, 1)),
            controls=controls,
            disturbances=np.empty((4, 0)),
            powers=np.ones((4, 1)),
            state_names=('state',),
            disturbance_names=(),
            power_names=('power_w',),
            encoder=encoder,
            segment_ids=np.array([0, 0, 1, 1]),
            source_versions=('v1', 'v2'),
        )

        self.assertEqual(data.transition_mask(0, 3).tolist(), [True, False, True])


class MixedIntegerMPCContractTest(unittest.TestCase):
    def test_soft_constraint_and_logical_rule_keep_problem_feasible(self):
        encoder = ControlEncoder(
            [
                ControlVariable('power', ('OFF', 'ON'), 'OFF'),
                ControlVariable('mode', ('low', 'high'), 'low'),
            ]
        )
        state_model = AffineStateModel(
            A=np.array([[1.0]]),
            B=np.zeros((1, 4)),
            E=np.zeros((1, 0)),
            c=np.zeros(1),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        # OFF categories cost 0 W; ON categories cost 10 W.
        power_model = AffinePowerModel(
            F=np.zeros((1, 1)),
            G=np.array([[0.0, 10.0, 0.0, 0.0]]),
            H=np.zeros((1, 0)),
            b=np.zeros(1),
            power_names=('power_w',),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        controller = MixedIntegerMPC(
            state_model=state_model,
            power_model=power_model,
            encoder=encoder,
            state_constraints=[
                {'name': 'state', 'lower': None, 'upper': 5.0, 'penalty': 100.0}
            ],
            logical_rules=[
                {
                    'when': {'variable': 'power', 'value': 'OFF'},
                    'require': {'mode': 'low'},
                }
            ],
            prediction_horizon=3,
            control_horizon=2,
            switch_penalty=0.0,
        )

        solution = controller.solve(
            initial_state=np.array([10.0]),
            disturbances=np.empty((3, 0)),
        )

        self.assertTrue(solution.success)
        self.assertEqual(solution.actions[0], {'power': 'OFF', 'mode': 'low'})
        self.assertGreater(solution.upper_slack.sum(), 0.0)
        self.assertAlmostEqual(solution.energy_wh, 0.0)

    def test_unobserved_complete_control_combination_is_forbidden(self):
        encoder = ControlEncoder(
            [
                ControlVariable('power', ('OFF', 'ON'), 'OFF'),
                ControlVariable('mode', ('low', 'high'), 'low'),
            ]
        )
        state_model = AffineStateModel(
            A=np.array([[1.0]]),
            B=np.array([[0.0, -10.0, 0.0, -10.0]]),
            E=np.zeros((1, 0)),
            c=np.zeros(1),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        power_model = AffinePowerModel(
            F=np.zeros((1, 1)),
            G=np.zeros((1, 4)),
            H=np.zeros((1, 0)),
            b=np.zeros(1),
            power_names=('power_w',),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        forbidden = encoder.encode_action(
            {'power': 'ON', 'mode': 'high'}
        )[None, :]
        controller = MixedIntegerMPC(
            state_model=state_model,
            power_model=power_model,
            encoder=encoder,
            state_constraints=[
                {'name': 'state', 'lower': None, 'upper': 5.0, 'penalty': 100.0}
            ],
            logical_rules=[],
            prediction_horizon=1,
            control_horizon=1,
            switch_penalty=0.0,
            forbidden_control_vectors=forbidden,
        )

        solution = controller.solve(
            initial_state=np.array([20.0]),
            disturbances=np.empty((1, 0)),
        )

        self.assertTrue(solution.success)
        self.assertNotEqual(
            solution.actions[0],
            {'power': 'ON', 'mode': 'high'},
        )
        self.assertGreaterEqual(solution.state_trajectory[1, 0], 10.0)

    def test_dynamic_forbidden_action_applies_only_to_first_planned_step(self):
        """A local action gate must not incorrectly freeze the full plan."""
        encoder = ControlEncoder(
            [ControlVariable('switch', ('OFF', 'ON'), 'OFF')]
        )
        state_model = AffineStateModel(
            A=np.array([[1.0]]),
            B=np.zeros((1, 2)),
            E=np.zeros((1, 0)),
            c=np.zeros(1),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        # ON is cheaper at every minute, so it is the unconstrained optimum.
        power_model = AffinePowerModel(
            F=np.zeros((1, 1)),
            G=np.array([[0.0, -9.0]]),
            H=np.zeros((1, 0)),
            b=np.array([10.0]),
            power_names=('power_w',),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        controller = MixedIntegerMPC(
            state_model=state_model,
            power_model=power_model,
            encoder=encoder,
            state_constraints=[
                {'name': 'state', 'lower': None, 'upper': None, 'penalty': 1.0}
            ],
            logical_rules=[],
            prediction_horizon=2,
            control_horizon=2,
            switch_penalty=0.0,
        )

        on = encoder.encode_action({'switch': 'ON'})
        solution = controller.solve(
            initial_state=np.array([0.0]),
            disturbances=np.empty((2, 0)),
            first_step_forbidden_control_vectors=on[None, :],
        )

        self.assertTrue(solution.success)
        self.assertEqual(solution.actions[0], {'switch': 'OFF'})
        # The local exclusion is deliberately only for z_0; z_1 remains a
        # provisional, unconstrained (and cheaper) future plan action.
        self.assertEqual(solution.actions[1], {'switch': 'ON'})

    def test_cap_limits_changed_categorical_variables_not_one_hot_bits(self):
        encoder = ControlEncoder(
            [
                ControlVariable('first', ('OFF', 'ON'), 'OFF'),
                ControlVariable('second', ('OFF', 'ON'), 'OFF'),
            ]
        )
        state_model = AffineStateModel(
            A=np.array([[1.0]]),
            B=np.zeros((1, 4)),
            E=np.zeros((1, 0)),
            c=np.zeros(1),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        # Turning either switch ON reduces nonnegative power.  The optimizer
        # would turn both ON immediately without the categorical-change cap.
        power_model = AffinePowerModel(
            F=np.zeros((1, 1)),
            G=np.array([[0.0, -10.0, 0.0, -10.0]]),
            H=np.zeros((1, 0)),
            b=np.array([25.0]),
            power_names=('power_w',),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        controller = MixedIntegerMPC(
            state_model=state_model,
            power_model=power_model,
            encoder=encoder,
            state_constraints=[
                {'name': 'state', 'lower': None, 'upper': None, 'penalty': 1.0}
            ],
            logical_rules=[],
            prediction_horizon=2,
            control_horizon=2,
            switch_penalty=0.0,
            maximum_changed_control_variables_per_step=1,
        )
        previous = encoder.encode_action({'first': 'OFF', 'second': 'OFF'})
        solution = controller.solve(
            initial_state=np.array([0.0]),
            disturbances=np.empty((2, 0)),
            previous_control=previous,
        )

        self.assertTrue(solution.success)
        planned = np.vstack((previous, solution.control_vectors))
        for prior, current in zip(planned[:-1], planned[1:]):
            changed_variables = sum(
                not np.array_equal(prior[list(group)], current[list(group)])
                for group in encoder.groups.values()
            )
            self.assertLessEqual(changed_variables, 1)
        # The first action changes exactly one *variable*, even though a
        # category switch flips two one-hot bits.
        self.assertEqual(
            sum(
                not np.array_equal(previous[list(group)], solution.control_vectors[0, list(group)])
                for group in encoder.groups.values()
            ),
            1,
        )

    def test_recent_action_history_penalty_prefers_a_stable_first_action(self):
        """The first action is compared with every configured past action."""
        encoder = ControlEncoder(
            [ControlVariable('switch', ('OFF', 'ON'), 'OFF')]
        )
        state_model = AffineStateModel(
            A=np.array([[1.0]]),
            B=np.zeros((1, 2)),
            E=np.zeros((1, 0)),
            c=np.zeros(1),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        power_model = AffinePowerModel(
            F=np.zeros((1, 1)),
            G=np.array([[0.0, -9.0]]),
            H=np.zeros((1, 0)),
            b=np.array([10.0]),
            power_names=('power_w',),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        controller = MixedIntegerMPC(
            state_model=state_model,
            power_model=power_model,
            encoder=encoder,
            state_constraints=[
                {'name': 'state', 'lower': None, 'upper': None, 'penalty': 1.0}
            ],
            logical_rules=[],
            prediction_horizon=1,
            control_horizon=1,
            switch_penalty=0.0,
            maximum_changed_control_variables_per_step=1,
            action_history_enabled=True,
            action_history_lookback_steps=3,
            action_history_penalty=1.0,
            action_history_recency_weights=[1.0, 0.8, 0.6],
        )
        off = encoder.encode_action({'switch': 'OFF'})
        solution = controller.solve(
            initial_state=np.array([0.0]),
            disturbances=np.empty((1, 0)),
            previous_control=off,
            action_history=np.vstack([off, off, off]),
        )

        # ON would reduce power from 10 W to 1 W, but differs from all three
        # recent actions and is therefore dominated by the history penalty.
        self.assertTrue(solution.success)
        self.assertEqual(solution.actions[0], {'switch': 'OFF'})
        diagnostics = controller.action_history_diagnostics(
            solution.control_vectors[0],
            np.vstack([off, off, off]),
            previous_control=off,
        )
        self.assertEqual(diagnostics['distance_by_lag'], {
            'lag_1': 0,
            'lag_2': 0,
            'lag_3': 0,
        })
        self.assertAlmostEqual(diagnostics['weighted_distance'], 0.0)

    def test_categorical_solution_boundary_canonicalizes_residue_and_rejects_relaxation(self):
        """A fractional LP point is never reused as a categorical action."""
        encoder = ControlEncoder(
            [
                ControlVariable('first', ('OFF', 'ON'), 'OFF'),
                ControlVariable('second', ('LOW', 'HIGH'), 'LOW'),
            ]
        )
        state_model = AffineStateModel(
            A=np.array([[1.0]]),
            B=np.zeros((1, encoder.n_features)),
            E=np.zeros((1, 0)),
            c=np.zeros(1),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        power_model = AffinePowerModel(
            F=np.zeros((1, 1)),
            G=np.zeros((1, encoder.n_features)),
            H=np.zeros((1, 0)),
            b=np.ones(1),
            power_names=('power_w',),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        controller = MixedIntegerMPC(
            state_model=state_model,
            power_model=power_model,
            encoder=encoder,
            state_constraints=[
                {'name': 'state', 'lower': None, 'upper': None, 'penalty': 1.0}
            ],
            logical_rules=[],
            prediction_horizon=1,
            control_horizon=1,
        )
        canonical = controller._canonicalize_categorical_controls(
            np.array([[1e-9, 0.999999999, 0.999999999, 1e-9]])
        )
        np.testing.assert_array_equal(
            canonical,
            encoder.encode_action({'first': 'ON', 'second': 'LOW'})[None, :],
        )
        self.assertIsNone(
            controller._canonicalize_categorical_controls(
                np.array([[0.5, 0.5, 1.0, 0.0]])
            )
        )

    def test_recent_action_rolling_budget_limits_repeated_switching(self):
        """Past changes consume the same five-minute budget as z_t changes."""
        encoder = ControlEncoder(
            [ControlVariable('switch', ('OFF', 'ON'), 'OFF')]
        )
        state_model = AffineStateModel(
            A=np.array([[1.0]]),
            B=np.zeros((1, 2)),
            E=np.zeros((1, 0)),
            c=np.zeros(1),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        power_model = AffinePowerModel(
            F=np.zeros((1, 1)),
            G=np.array([[0.0, -9.0]]),
            H=np.zeros((1, 0)),
            b=np.array([10.0]),
            power_names=('power_w',),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        controller = MixedIntegerMPC(
            state_model=state_model,
            power_model=power_model,
            encoder=encoder,
            state_constraints=[
                {'name': 'state', 'lower': None, 'upper': None, 'penalty': 1.0}
            ],
            logical_rules=[],
            prediction_horizon=1,
            control_horizon=1,
            switch_penalty=0.0,
            maximum_changed_control_variables_per_step=1,
            action_history_enabled=True,
            action_history_lookback_steps=3,
            action_history_penalty=0.0,
            action_history_rolling_change_budget=2,
            action_history_budget_slack_penalty=100.0,
        )
        off = encoder.encode_action({'switch': 'OFF'})
        on = encoder.encode_action({'switch': 'ON'})
        # OFF -> ON -> OFF has already consumed two action changes. Selecting
        # ON again would create a third change and pay the large slack cost.
        history = np.vstack([off, on, off])
        solution = controller.solve(
            initial_state=np.array([0.0]),
            disturbances=np.empty((1, 0)),
            previous_control=off,
            action_history=history,
        )

        self.assertTrue(solution.success)
        self.assertEqual(solution.actions[0], {'switch': 'OFF'})
        diagnostics = controller.action_history_diagnostics(
            solution.control_vectors[0],
            history,
            previous_control=off,
        )
        self.assertEqual(diagnostics['historical_change_count'], 2)
        self.assertEqual(diagnostics['rolling_change_count'], 2)
        self.assertEqual(diagnostics['rolling_budget_overflow'], 0.0)

    def test_per_control_variable_history_budget_limits_only_named_variable(self):
        """Independent group budgets work with the automatic per-variable mode."""
        encoder = ControlEncoder(
            [
                ControlVariable('guarded', ('OFF', 'ON'), 'OFF'),
                ControlVariable('free', ('OFF', 'ON'), 'OFF'),
            ]
        )
        state_model = AffineStateModel(
            A=np.array([[1.0]]),
            B=np.zeros((1, encoder.n_features)),
            E=np.zeros((1, 0)),
            c=np.zeros(1),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        # The guarded ON setting would save more power than free ON. Its
        # independent recent-history budget is zero, so the optimizer must
        # select only the unrestricted variable despite the energy incentive.
        power_model = AffinePowerModel(
            F=np.zeros((1, 1)),
            G=np.array([[0.0, -9.0, 0.0, -2.0]]),
            H=np.zeros((1, 0)),
            b=np.array([10.0]),
            power_names=('power_w',),
            state_names=('state',),
            control_feature_names=encoder.feature_names,
            disturbance_names=(),
        )
        controller = MixedIntegerMPC(
            state_model=state_model,
            power_model=power_model,
            encoder=encoder,
            state_constraints=[
                {'name': 'state', 'lower': None, 'upper': None, 'penalty': 1.0}
            ],
            logical_rules=[],
            prediction_horizon=1,
            control_horizon=1,
            switch_penalty=0.0,
            maximum_changed_control_variables_per_step=2,
            action_history_enabled=True,
            action_history_lookback_steps=2,
            action_history_penalty=0.0,
            action_history_recency_weights=[1.0, 1.0],
            # No explicit groups means one budget unit per control variable.
            action_history_per_group_rolling_change_budgets={'guarded': 0},
            action_history_budget_slack_penalty=100.0,
        )
        off = encoder.encode_action({'guarded': 'OFF', 'free': 'OFF'})
        solution = controller.solve(
            initial_state=np.array([0.0]),
            disturbances=np.empty((1, 0)),
            previous_control=off,
            action_history=np.vstack([off, off]),
        )

        self.assertTrue(solution.success)
        self.assertEqual(solution.actions[0], {'guarded': 'OFF', 'free': 'ON'})
        diagnostics = controller.action_history_diagnostics(
            solution.control_vectors[0],
            np.vstack([off, off]),
            previous_control=off,
        )
        self.assertEqual(
            diagnostics['per_group_rolling_change_budgets'], {'guarded': 0}
        )
        self.assertEqual(diagnostics['current_change_by_group']['guarded'], 0)
        self.assertEqual(diagnostics['current_change_by_group']['free'], 1)
        self.assertEqual(
            diagnostics['per_group_rolling_budget_overflow'], {'guarded': 0.0}
        )

    def test_power_mape_reports_threshold_and_coverage(self):
        actual = np.array(
            [
                [0.0, 10.0],
                [0.1, 20.0],
                [10.0, 0.0],
                [20.0, 40.0],
            ]
        )
        predicted = np.array(
            [
                [100.0, 9.0],
                [100.0, 18.0],
                [9.0, 100.0],
                [18.0, 36.0],
            ]
        )

        metrics = power_mape_metrics(actual, predicted, 0.1)

        self.assertEqual(metrics['mape_percent_per_output'], [10.0, 10.0])
        self.assertAlmostEqual(metrics['mape_percent'], 10.0)
        self.assertAlmostEqual(metrics['mape_percent_macro_mean'], 10.0)
        self.assertEqual(metrics['mape_valid_samples_per_output'], [2, 3])
        self.assertEqual(metrics['mape_total_samples_per_output'], 4)
        self.assertEqual(
            metrics['mape_coverage_percent_per_output'],
            [50.0, 75.0],
        )

    def test_power_mape_is_none_when_every_actual_is_near_zero(self):
        metrics = power_mape_metrics(
            np.array([[0.0], [0.1]]),
            np.array([[1.0], [1.0]]),
            0.1,
        )

        self.assertEqual(metrics['mape_percent_per_output'], [None])
        self.assertIsNone(metrics['mape_percent'])
        self.assertIsNone(metrics['mape_percent_macro_mean'])
        self.assertEqual(metrics['mape_coverage_percent_per_output'], [0.0])


class MPCEvaluationContractTest(unittest.TestCase):
    @staticmethod
    def _repeated_action_episode_data():
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
    def _fixed_action_config():
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

    def test_closed_loop_action_history_uses_prior_mpc_actions_after_bootstrap(self):
        """The second decision must not read the future recorded control."""
        encoder = ControlEncoder(
            [ControlVariable('switch', ('OFF', 'ON'), 'OFF')]
        )
        off = encoder.encode_action({'switch': 'OFF'})
        on = encoder.encode_action({'switch': 'ON'})
        row_count = 8
        data = ControlData(
            timestamps=pd.Series(
                pd.date_range('2025-01-01', periods=row_count, freq='min')
            ),
            states=np.zeros((row_count, 1)),
            controls=np.repeat(off[None, :], row_count, axis=0),
            disturbances=np.empty((row_count, 0)),
            powers=np.ones((row_count, 1)),
            state_names=('state',),
            disturbance_names=(),
            power_names=('power_w',),
            encoder=encoder,
        )

        class PowerModel:
            def predict_components(self, states, controls, disturbances):
                return np.zeros((len(np.atleast_2d(states)), 1), dtype=float)

        class HistoryCapturingController:
            prediction_horizon = 1
            action_history_enabled = True
            action_history_lookback_steps = 2
            power_model = PowerModel()
            histories = []

            def solve(
                self,
                initial_state,
                disturbances,
                previous_control,
                action_history=None,
            ):
                del initial_state, disturbances, previous_control
                self.histories.append(np.asarray(action_history, dtype=float).copy())
                return SimpleNamespace(
                    success=True,
                    status=0,
                    message='optimal',
                    objective=0.0,
                    control_vectors=on[None, :].copy(),
                )

            def action_history_diagnostics(
                self,
                applied_control,
                action_history,
                previous_control=None,
            ):
                del applied_control, action_history, previous_control
                return {
                    'enabled': True,
                    'lookback_steps_used': 2,
                    'weighted_distance': 0.0,
                    'historical_change_count': 0,
                    'current_change_count': 0,
                    'rolling_change_count': 0,
                    'rolling_change_budget': 3,
                    'rolling_budget_overflow': 0.0,
                    'distance_by_lag': {'lag_1': 0, 'lag_2': 0},
                }

        class StatelessPlant:
            def step(self, state, control, disturbance):
                del control, disturbance
                return np.asarray(state, dtype=float).copy(), np.array([0.0])

        controller = HistoryCapturingController()
        run_receding_horizon_policy(
            controller,
            StatelessPlant(),
            data,
            start=2,
            duration_steps=2,
            disturbance_forecast_method='persistence',
            training_control_support=data.controls[:2],
        )

        self.assertTrue(np.array_equal(controller.histories[0], np.vstack([off, off])))
        # The first MPC action is ON. It must replace the oldest bootstrap
        # action in the next two-step history, even though the recorded t=2
        # action remains OFF.
        self.assertTrue(np.array_equal(controller.histories[1], np.vstack([off, on])))

    def test_fixed_action_holdout_is_episode_disjoint_and_outcome_blind(self):
        data = self._repeated_action_episode_data()
        holdout = build_fixed_action_episode_holdout(
            data,
            self._fixed_action_config(),
            split_rate=(0.8, 0.1, 0.1),
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
        changed_holdout = build_fixed_action_episode_holdout(
            changed_outcomes,
            self._fixed_action_config(),
            split_rate=(0.8, 0.1, 0.1),
        )

        train = set(holdout.train_rows.tolist())
        validation = set(holdout.validation_rows.tolist())
        test = set(holdout.test_rows.tolist())
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(holdout.metadata['candidate_count'], 2)
        self.assertEqual(
            holdout.metadata['protocol']['outcome_columns_used_for_assignment'],
            [],
        )
        self.assertFalse(
            holdout.metadata['protocol']['temporal_generalization_claim']
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
            episode_sets = [
                set(candidate['support'][name]['episode_ids'])
                for name in ('train', 'validation', 'test')
            ]
            self.assertFalse(episode_sets[0] & episode_sets[1])
            self.assertFalse(episode_sets[0] & episode_sets[2])
            self.assertFalse(episode_sets[1] & episode_sets[2])
        expected_candidate_rows = sum(
            candidate['total_sample_count']
            for candidate in holdout.metadata['candidates']
        )
        self.assertEqual(len(holdout.dataset_rows), expected_candidate_rows)
        for split_name in ('train', 'validation', 'test'):
            expected_split_rows = sum(
                candidate['support'][split_name]['sample_count']
                for candidate in holdout.metadata['candidates']
            )
            actual_split_rows = sum(
                row['split'] == split_name for row in holdout.dataset_rows
            )
            self.assertEqual(actual_split_rows, expected_split_rows)
        dataset_episode_splits = {}
        for row in holdout.dataset_rows:
            dataset_episode_splits.setdefault(row['episode_id'], set()).add(
                row['split']
            )
            self.assertIn('target_power_ventilation_power_w', row)
            self.assertIn('target_next_state_state', row)
        self.assertTrue(
            all(len(splits) == 1 for splits in dataset_episode_splits.values())
        )


    def test_fixed_action_protocol_freezes_selection_and_uses_test_actual_y(self):
        data = self._repeated_action_episode_data()
        holdout = build_fixed_action_episode_holdout(
            data,
            self._fixed_action_config(),
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

        class FrozenReportEvaluator:
            def predict_state(self, states, controls, disturbances):
                return np.zeros((len(states), 1))

            def predict_power(self, states, controls, disturbances):
                return np.tile(np.array([[11.0, 22.0]]), (len(states), 1))

        result, audit_rows = evaluate_fixed_optimal_action_holdout(
            data,
            holdout,
            SelectionEvaluator(),
            FrozenReportEvaluator(),
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
            frozen_report_evaluator_name='frozen-train-validation-reporter',
        )

        self.assertTrue(result['reportable'])
        self.assertEqual(result['selection']['selected_action'], {'control': 'ON'})
        self.assertTrue(result['selection']['frozen_before_test'])
        self.assertTrue(result['protocol']['selected_action_frozen_during_test'])
        self.assertTrue(result['protocol']['report_model_frozen_during_test'])
        self.assertFalse(
            result['protocol']['test_power_and_state_targets_used_for_selection']
        )
        self.assertAlmostEqual(result['primary_metric']['value'], 10.0)
        self.assertGreaterEqual(result['test_evaluation']['exact_match_count'], 8)
        self.assertGreaterEqual(
            result['test_evaluation']['exact_match_episode_count'],
            3,
        )
        self.assertEqual(
            len(audit_rows),
            result['test_evaluation']['exact_match_count'],
        )
        self.assertTrue(
            all(row['mape_denominator_valid'] for row in audit_rows)
        )
        json.dumps(result, allow_nan=False)

    def test_fixed_action_soft_constraint_fallback_keeps_mape_auditable(self):
        data = self._repeated_action_episode_data()
        holdout = build_fixed_action_episode_holdout(
            data,
            self._fixed_action_config(),
            split_rate=(0.8, 0.1, 0.1),
        )

        class Evaluator:
            def __init__(self, report=False):
                self.report = report

            def predict_state(self, states, controls, disturbances):
                return np.zeros((len(states), 1))

            def predict_power(self, states, controls, disturbances):
                if self.report:
                    return np.tile(np.array([[11.0, 22.0]]), (len(states), 1))
                is_on = controls[:, 1] > 0.5
                return np.where(
                    is_on[:, None],
                    np.array([[4.0, 6.0]]),
                    np.array([[20.0, 20.0]]),
                )

        arguments = {
            'state_constraints': [
                {'name': 'state', 'lower': 1.0, 'upper': 2.0}
            ],
            'minimum_predicted_state_satisfaction_percent': 100.0,
            'minimum_test_exact_matches': 8,
            'minimum_test_episodes': 3,
            'minimum_test_mape_samples': 8,
            'mape_min_actual_power_w': 0.1,
            'selection_evaluator_name': 'selector',
            'frozen_report_evaluator_name': 'reporter',
        }
        result, _ = evaluate_fixed_optimal_action_holdout(
            data,
            holdout,
            Evaluator(),
            Evaluator(report=True),
            allow_soft_constraint_fallback=True,
            **arguments,
        )
        strict_result, _ = evaluate_fixed_optimal_action_holdout(
            data,
            holdout,
            Evaluator(),
            Evaluator(report=True),
            allow_soft_constraint_fallback=False,
            **arguments,
        )

        self.assertTrue(result['reportable'])
        self.assertAlmostEqual(result['primary_metric']['value'], 10.0)
        self.assertFalse(
            result['selection']['strict_constraint_candidate_available']
        )
        self.assertTrue(result['selection']['soft_constraint_fallback_used'])
        self.assertEqual(
            result['selection']['selection_mode'],
            'maximum_state_satisfaction_then_minimum_power_fallback',
        )
        self.assertFalse(strict_result['reportable'])
        self.assertIsNone(strict_result['primary_metric']['value'])
        self.assertFalse(strict_result['selection']['soft_constraint_fallback_used'])

    def test_fixed_action_preference_selects_on_within_state_tolerance(self):
        data = self._repeated_action_episode_data()
        holdout = build_fixed_action_episode_holdout(
            data,
            self._fixed_action_config(),
            split_rate=(0.8, 0.1, 0.1),
        )

        class Evaluator:
            def __init__(self, report=False):
                self.report = report

            def predict_state(self, states, controls, disturbances):
                return np.zeros((len(states), 1))

            def predict_power(self, states, controls, disturbances):
                if self.report:
                    return np.tile(np.array([[10.0, 20.0]]), (len(states), 1))
                is_on = controls[:, 1] > 0.5
                return np.where(
                    is_on[:, None],
                    np.array([[20.0, 20.0]]),
                    np.array([[5.0, 5.0]]),
                )

        arguments = {
            'state_constraints': [
                {'name': 'state', 'lower': 1.0, 'upper': 2.0}
            ],
            'minimum_predicted_state_satisfaction_percent': 100.0,
            'allow_soft_constraint_fallback': True,
            'minimum_test_exact_matches': 8,
            'minimum_test_episodes': 3,
            'minimum_test_mape_samples': 8,
            'mape_min_actual_power_w': 0.1,
            'selection_evaluator_name': 'selector',
            'frozen_report_evaluator_name': 'reporter',
        }
        baseline, _ = evaluate_fixed_optimal_action_holdout(
            data, holdout, Evaluator(), Evaluator(report=True), **arguments
        )
        preferred, _ = evaluate_fixed_optimal_action_holdout(
            data,
            holdout,
            Evaluator(),
            Evaluator(report=True),
            selection_preference={
                'enabled': True,
                'preferred_values': {'control': 'ON'},
                'maximum_state_satisfaction_loss_percent': 0.0,
            },
            **arguments,
        )

        self.assertEqual(
            baseline['selection']['selected_action'], {'control': 'OFF'}
        )
        self.assertEqual(
            preferred['selection']['selected_action'], {'control': 'ON'}
        )
        self.assertTrue(
            preferred['selection']['selection_preference_applied']
        )
        self.assertTrue(
            preferred['selection']['selected_validation_score'][
                'matches_all_preferred_values'
            ]
        )

    def test_fixed_action_insufficient_support_is_null_instead_of_crashing(self):
        data = self._repeated_action_episode_data()
        config = self._fixed_action_config()
        config.minimum_training_samples_per_candidate = 10000
        holdout = build_fixed_action_episode_holdout(
            data,
            config,
            split_rate=(0.8, 0.1, 0.1),
        )
        result, rows = evaluate_fixed_optimal_action_holdout(
            data,
            holdout,
            object(),
            object(),
            state_constraints=[
                {'name': 'state', 'lower': -1.0, 'upper': 1.0}
            ],
            minimum_predicted_state_satisfaction_percent=100.0,
            allow_soft_constraint_fallback=True,
            minimum_test_exact_matches=8,
            minimum_test_episodes=3,
            minimum_test_mape_samples=8,
            mape_min_actual_power_w=0.1,
            selection_evaluator_name='unused',
            frozen_report_evaluator_name='unused',
        )

        self.assertEqual(holdout.metadata['candidate_count'], 0)
        self.assertFalse(result['reportable'])
        self.assertIsNone(result['primary_metric']['value'])
        self.assertIn('No joint action satisfies', result['reason'])
        self.assertEqual(rows, [])
        json.dumps(result, allow_nan=False)

    def test_explicit_model_rows_preserve_real_transition_indices(self):
        data = self._repeated_action_episode_data()
        with self.assertRaises(IndexError):
            data.valid_transition_rows(np.array([2, -1, 999]))
        requested = np.array([10, 2, 9, 2])
        valid = data.valid_transition_rows(requested)
        np.testing.assert_array_equal(valid, np.array([2, 9, 10]))
        states, controls, disturbances, next_states, powers = (
            data.model_arrays_for_rows(requested)
        )
        np.testing.assert_array_equal(states, data.states[valid])
        np.testing.assert_array_equal(controls, data.controls[valid])
        np.testing.assert_array_equal(disturbances, data.disturbances[valid])
        np.testing.assert_array_equal(next_states, data.states[valid + 1])
        np.testing.assert_array_equal(powers, data.powers[valid])

    def test_false_safe_and_paired_bootstrap_are_explicit(self):
        constraint_result = constraint_prediction_metrics(
            actual_states=np.array([[6.0], [4.0]]),
            predicted_states=np.array([[4.0], [4.0]]),
            specifications=[
                {'name': 'state', 'lower': None, 'upper': 5.0}
            ],
        )
        self.assertEqual(
            constraint_result['false_safe_state_step_pairs'],
            1,
        )
        self.assertAlmostEqual(
            constraint_result[
                'overall_false_safe_rate_of_actual_violations_percent'
            ],
            100.0,
        )

        bootstrap = paired_bootstrap_energy_reduction(
            baseline_energy_wh=[10.0, 20.0],
            candidate_energy_wh=[5.0, 10.0],
            samples=100,
            confidence=0.95,
            seed=42,
        )
        self.assertAlmostEqual(bootstrap['lower_percent'], 50.0)
        self.assertAlmostEqual(bootstrap['upper_percent'], 50.0)

    def test_observed_combination_mape_uses_validation_selection_and_test_actual_y(
        self,
    ):
        encoder = ControlEncoder(
            [ControlVariable('control', ('OFF', 'ON'), 'OFF')]
        )
        actions = [
            'OFF',
            'OFF',
            'ON',
            'ON',
            'OFF',
            'ON',
            'ON',
            'ON',
            'ON',
            'ON',
            'ON',
            'OFF',
            'OFF',
        ]
        controls = np.vstack(
            [encoder.encode_action({'control': action}) for action in actions]
        )
        powers = np.tile(np.array([[10.0, 20.0]]), (len(actions), 1))
        data = ControlData(
            timestamps=pd.Series(
                pd.date_range('2025-01-01', periods=len(actions), freq='min')
            ),
            states=np.zeros((len(actions), 1)),
            controls=controls,
            disturbances=np.empty((len(actions), 0)),
            powers=powers,
            state_names=('state',),
            disturbance_names=(),
            power_names=('ventilation_power_w', 'hvac_power_w'),
            encoder=encoder,
        )

        class FixedEvaluator:
            def __init__(self, role):
                self.role = role

            def predict_transitions(self, source, start, end):
                row_indices = np.arange(start, end, dtype=int)
                if self.role == 'selector':
                    is_on = source.controls[row_indices, 1] > 0.5
                    predicted_power = np.where(
                        is_on[:, None],
                        np.array([[4.0, 6.0]]),
                        np.array([[20.0, 20.0]]),
                    )
                else:
                    predicted_power = source.powers[row_indices] * 1.1
                return TransitionPredictions(
                    actual_states=source.states[row_indices + 1],
                    predicted_states=source.states[row_indices + 1],
                    actual_powers=source.powers[row_indices],
                    predicted_powers=predicted_power,
                    row_indices=row_indices,
                )

        result, audit_rows = evaluate_observed_control_combinations(
            data,
            FixedEvaluator('selector'),
            FixedEvaluator('reporter'),
            validation_start=4,
            test_start=8,
            test_end=12,
            state_constraints=[
                {'name': 'state', 'lower': -1.0, 'upper': 1.0}
            ],
            minimum_validation_samples=2,
            minimum_predicted_state_satisfaction_percent=100.0,
            minimum_test_exact_matches=2,
            minimum_test_mape_samples=2,
            mape_min_actual_power_w=0.1,
            selection_evaluator_name='train-only-selector',
            frozen_report_evaluator_name='frozen-test-reporter',
        )

        self.assertTrue(result['reportable'])
        self.assertEqual(
            result['selection']['selected_action'],
            {'control': 'ON'},
        )
        self.assertEqual(
            result['test_evaluation']['exact_match_count'],
            3,
        )
        self.assertAlmostEqual(
            result['test_evaluation']['primary_metric']['value'],
            10.0,
        )
        self.assertEqual(len(audit_rows), 3)
        self.assertTrue(
            result['protocol']['report_model_frozen_during_test']
        )
        self.assertFalse(
            result['protocol']['test_power_and_state_targets_used_for_selection']
        )

    def test_mpc_selected_action_crossfit_uses_frequency_weighted_actual_y_mape(
        self,
    ):
        encoder = ControlEncoder(
            [ControlVariable('control', ('OFF', 'ON'), 'OFF')]
        )
        actions = [
            'OFF',
            'ON',
            'OFF',
            'ON',
            'OFF',
            'ON',
            'OFF',
            'ON',
            'OFF',
            'ON',
            'OFF',
        ]
        controls = np.vstack(
            [encoder.encode_action({'control': action}) for action in actions]
        )
        powers = np.asarray(
            [[10.0] if action == 'OFF' else [20.0] for action in actions]
        )
        data = ControlData(
            timestamps=pd.Series(
                pd.date_range('2025-01-01', periods=len(actions), freq='min')
            ),
            states=np.zeros((len(actions), 1)),
            controls=controls,
            disturbances=np.empty((len(actions), 0)),
            powers=powers,
            state_names=('state',),
            disturbance_names=(),
            power_names=('total_power_w',),
            encoder=encoder,
        )

        def fold(fold_number, start, end):
            rows = np.arange(start, end, dtype=int)
            actual = data.powers[rows]
            is_on = data.controls[rows, 1] > 0.5
            predicted = np.where(is_on[:, None], actual * 1.2, actual * 1.1)
            return {
                'fold_number': fold_number,
                'training_end_exclusive': start,
                'evaluation_start': start,
                'evaluation_end_exclusive': end,
                'evaluator_name': f'evaluator-{fold_number}',
                'predictions': TransitionPredictions(
                    actual_states=data.states[rows + 1],
                    predicted_states=data.states[rows + 1],
                    actual_powers=actual,
                    predicted_powers=predicted,
                    row_indices=rows,
                ),
            }

        selected_controls = np.vstack(
            [
                encoder.encode_action({'control': 'OFF'}),
                encoder.encode_action({'control': 'OFF'}),
                encoder.encode_action({'control': 'OFF'}),
                encoder.encode_action({'control': 'ON'}),
            ]
        )
        result, audit_rows = evaluate_mpc_selected_action_crossfit(
            data,
            selected_controls,
            np.zeros((4, 1)),
            np.empty((4, 0)),
            [fold(1, 4, 7), fold(2, 7, 10)],
            initial_training_end_exclusive=4,
            context_distance_threshold=1.0,
            minimum_matched_samples_per_action=3,
            minimum_independent_episodes_per_action=3,
            minimum_fold_training_samples_per_action=1,
            minimum_fold_training_episodes_per_action=1,
            minimum_reportable_mpc_step_coverage_percent=100.0,
            mape_min_actual_power_w=0.1,
            evaluator_family='fixed-test-evaluator',
        )

        self.assertTrue(result['reportable'])
        self.assertAlmostEqual(result['primary_metric']['value'], 12.5)
        self.assertAlmostEqual(
            result['combination_balanced_metric']['value'],
            15.0,
        )
        self.assertEqual(
            result['coverage']['reportable_action_mpc_step_percent'],
            100.0,
        )
        self.assertEqual(result['reportable_action_count'], 2)
        self.assertEqual(len(audit_rows), 6)
        self.assertEqual(
            {action['independent_episode_count'] for action in result['actions']},
            {3},
        )

    def test_mpc_selected_action_crossfit_keeps_primary_null_when_unsupported(
        self,
    ):
        encoder = ControlEncoder(
            [ControlVariable('control', ('OFF', 'ON'), 'OFF')]
        )
        actions = ['OFF', 'OFF', 'OFF', 'OFF', 'OFF', 'OFF']
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
            powers=np.full((len(actions), 1), 10.0),
            state_names=('state',),
            disturbance_names=(),
            power_names=('total_power_w',),
            encoder=encoder,
        )
        rows = np.array([3, 4])
        predictions = TransitionPredictions(
            actual_states=data.states[rows + 1],
            predicted_states=data.states[rows + 1],
            actual_powers=data.powers[rows],
            predicted_powers=data.powers[rows] * 1.1,
            row_indices=rows,
        )
        selected_controls = np.vstack(
            [encoder.encode_action({'control': 'ON'})] * 2
        )

        result, audit_rows = evaluate_mpc_selected_action_crossfit(
            data,
            selected_controls,
            np.zeros((2, 1)),
            np.empty((2, 0)),
            [
                {
                    'fold_number': 1,
                    'training_end_exclusive': 3,
                    'evaluation_start': 3,
                    'evaluation_end_exclusive': 5,
                    'predictions': predictions,
                }
            ],
            initial_training_end_exclusive=3,
            context_distance_threshold=1.0,
            minimum_matched_samples_per_action=1,
            minimum_independent_episodes_per_action=1,
            minimum_fold_training_samples_per_action=1,
            minimum_fold_training_episodes_per_action=1,
            minimum_reportable_mpc_step_coverage_percent=1.0,
            mape_min_actual_power_w=0.1,
            evaluator_family='fixed-test-evaluator',
        )

        self.assertFalse(result['available'])
        self.assertFalse(result['reportable'])
        self.assertIsNone(result['primary_metric']['value'])
        self.assertEqual(
            result['coverage']['exact_action_supported_mpc_step_percent'],
            0.0,
        )
        self.assertEqual(
            result['actions'][0]['nearest_context_distance'],
            {'minimum': None, 'median': None, 'maximum': None},
        )
        json.dumps(result, allow_nan=False)
        self.assertEqual(audit_rows, [])

    def test_crossfit_rejects_action_absent_from_fold_training(self):
        encoder = ControlEncoder(
            [ControlVariable('control', ('OFF', 'ON'), 'OFF')]
        )
        actions = ['OFF', 'OFF', 'OFF', 'ON', 'ON', 'OFF']
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
            powers=np.full((len(actions), 1), 10.0),
            state_names=('state',),
            disturbance_names=(),
            power_names=('total_power_w',),
            encoder=encoder,
        )
        rows = np.array([3, 4])
        predictions = TransitionPredictions(
            actual_states=data.states[rows + 1],
            predicted_states=data.states[rows + 1],
            actual_powers=data.powers[rows],
            predicted_powers=data.powers[rows] * 1.1,
            row_indices=rows,
        )
        result, audit_rows = evaluate_mpc_selected_action_crossfit(
            data,
            np.vstack([encoder.encode_action({'control': 'ON'})] * 2),
            np.zeros((2, 1)),
            np.empty((2, 0)),
            [
                {
                    'fold_number': 1,
                    'training_end_exclusive': 3,
                    'evaluation_start': 3,
                    'evaluation_end_exclusive': 5,
                    'predictions': predictions,
                }
            ],
            initial_training_end_exclusive=3,
            context_distance_threshold=1.0,
            minimum_matched_samples_per_action=1,
            minimum_independent_episodes_per_action=1,
            minimum_fold_training_samples_per_action=1,
            minimum_fold_training_episodes_per_action=1,
            minimum_reportable_mpc_step_coverage_percent=1.0,
            mape_min_actual_power_w=0.1,
            evaluator_family='fixed-test-evaluator',
        )

        action = result['actions'][0]
        self.assertEqual(action['crossfit_raw_exact_action_sample_count'], 2)
        self.assertEqual(action['crossfit_exact_action_sample_count'], 0)
        self.assertIn(
            'action_not_sufficiently_observed_in_fold_training',
            action['unsupported_reasons'],
        )
        self.assertFalse(result['reportable'])
        self.assertEqual(audit_rows, [])

    def test_chronological_action_support_uses_disjoint_complete_episodes(self):
        encoder = ControlEncoder(
            [ControlVariable('control', ('OFF', 'ON'), 'OFF')]
        )
        actions = ['OFF', 'OFF', 'ON', 'ON'] * 3
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
            powers=np.full((len(actions), 1), 10.0),
            state_names=('state',),
            disturbance_names=(),
            power_names=('total_power_w',),
            encoder=encoder,
        )

        aligned = align_boundaries_to_control_episodes(data, [3, 7])
        self.assertEqual(aligned, [2, 6])
        support, audit_rows = chronological_action_split_support(
            data,
            4,
            8,
            minimum_samples_per_split=2,
            minimum_complete_episodes_per_split=1,
        )

        self.assertTrue(support['episode_boundaries_are_disjoint'])
        self.assertEqual(support['cross_split_episode_count'], 0)
        self.assertEqual(support['present_in_all_splits_action_count'], 2)
        self.assertEqual(support['eligible_in_all_splits_action_count'], 2)
        self.assertEqual(len(audit_rows), 2)

        unaligned_support, _ = chronological_action_split_support(
            data,
            3,
            7,
            minimum_samples_per_split=1,
            minimum_complete_episodes_per_split=1,
        )
        self.assertGreater(unaligned_support['cross_split_episode_count'], 0)

    def test_repeated_joint_action_filter_can_be_toggled(self):
        encoder = ControlEncoder(
            [ControlVariable('control', ('OFF', 'ON'), 'OFF')]
        )
        actions = ['OFF', 'OFF', 'OFF', 'ON', 'OFF', 'ON']
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
            powers=np.full((len(actions), 1), 10.0),
            state_names=('state',),
            disturbance_names=(),
            power_names=('power',),
            encoder=encoder,
        )

        allowed_on, forbidden_on, metadata_on = _training_control_support(
            data,
            len(actions),
            (),
            True,
            repeated_action_filter_enabled=True,
            minimum_repeated_samples=3,
            minimum_repeated_episodes=2,
        )
        allowed_off, forbidden_off, _ = _training_control_support(
            data,
            len(actions),
            (),
            True,
            repeated_action_filter_enabled=False,
            minimum_repeated_samples=3,
            minimum_repeated_episodes=2,
        )

        self.assertEqual(len(allowed_on), 1)
        self.assertEqual(
            encoder.decode_action(allowed_on[0]),
            {'control': 'OFF'},
        )
        self.assertEqual(len(forbidden_on), 1)
        self.assertEqual(len(allowed_off), 2)
        self.assertEqual(len(forbidden_off), 0)
        self.assertTrue(
            metadata_on['repeated_joint_action_filter']['enabled']
        )

    def test_category_consolidation_can_be_toggled(self):
        encoder = ControlEncoder(
            [ControlVariable('mode', ('A', 'B', 'C'), 'A')]
        )
        original = pd.DataFrame({'mode': ['A', 'B', 'C', 'B']})
        disabled = original.copy()
        disabled_metadata = consolidate_control_categories(
            disabled,
            encoder,
            OmegaConf.create({'enabled': False, 'mappings': {}}),
        )
        enabled = original.copy()
        enabled_metadata = consolidate_control_categories(
            enabled,
            encoder,
            OmegaConf.create(
                {
                    'enabled': True,
                    'mappings': {'mode': {'B': 'A', 'C': 'A'}},
                }
            ),
        )

        self.assertEqual(disabled['mode'].tolist(), original['mode'].tolist())
        self.assertFalse(disabled_metadata['enabled'])
        self.assertEqual(enabled['mode'].tolist(), ['A', 'A', 'A', 'A'])
        self.assertEqual(enabled_metadata['total_replacement_count'], 3)

        data = ControlData(
            timestamps=pd.Series(
                pd.date_range('2025-01-01', periods=4, freq='min')
            ),
            states=np.zeros((4, 1)),
            controls=encoder.transform(enabled),
            disturbances=np.empty((4, 0)),
            powers=np.full((4, 1), 10.0),
            state_names=('state',),
            disturbance_names=(),
            power_names=('power',),
            encoder=encoder,
            control_preprocessing={
                'category_consolidation': enabled_metadata,
            },
        )
        allowed, forbidden, _ = _training_control_support(
            data,
            4,
            (),
            False,
        )
        self.assertEqual(len(allowed), 1)
        self.assertEqual(len(forbidden), 2)

    def test_official_episode_support_filter_can_be_toggled(self):
        encoder = ControlEncoder(
            [ControlVariable('control', ('OFF', 'ON'), 'OFF')]
        )
        actions = ['ON', 'OFF', 'ON', 'OFF', 'ON', 'ON', 'ON', 'OFF']
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
            powers=np.full((len(actions), 1), 10.0),
            state_names=('state',),
            disturbance_names=(),
            power_names=('power',),
            encoder=encoder,
        )
        rows = np.arange(4, 7)
        predictions = TransitionPredictions(
            actual_states=data.states[rows + 1],
            predicted_states=data.states[rows + 1],
            actual_powers=data.powers[rows],
            predicted_powers=data.powers[rows] * 1.1,
            row_indices=rows,
        )
        fold = {
            'fold_number': 1,
            'training_end_exclusive': 4,
            'evaluation_start': 4,
            'evaluation_end_exclusive': 7,
            'predictions': predictions,
        }

        def evaluate(exclude_insufficient_episodes):
            return evaluate_mpc_selected_action_crossfit(
                data,
                np.vstack([encoder.encode_action({'control': 'ON'})]),
                np.zeros((1, 1)),
                np.empty((1, 0)),
                [fold],
                initial_training_end_exclusive=4,
                context_distance_threshold=1.0,
                minimum_matched_samples_per_action=3,
                minimum_independent_episodes_per_action=2,
                minimum_fold_training_samples_per_action=2,
                minimum_fold_training_episodes_per_action=2,
                minimum_reportable_mpc_step_coverage_percent=100.0,
                mape_min_actual_power_w=0.1,
                evaluator_family='fixed-test-evaluator',
                exclude_insufficient_episode_actions=(
                    exclude_insufficient_episodes
                ),
            )[0]

        strict = evaluate(True)
        relaxed = evaluate(False)
        self.assertFalse(strict['reportable'])
        self.assertIn(
            'insufficient_independent_episodes',
            strict['actions'][0]['unsupported_reasons'],
        )
        self.assertTrue(relaxed['reportable'])
        self.assertAlmostEqual(relaxed['primary_metric']['value'], 10.0)


class ForecastTrainingContractTest(unittest.TestCase):
    def test_plateau_scheduler_reduces_only_after_stagnation_and_honors_floor(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1e-4)
        scheduler = create_lr_scheduler(
            optimizer,
            'reduce_on_plateau',
            {'factor': 0.5, 'patience': 1, 'threshold': 0.0, 'min_lr': 2.5e-5},
        )
        scheduler.step(1.0)
        scheduler.step(1.1)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-4)
        scheduler.step(1.2)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 5e-5)
        for value in (1.3, 1.4, 1.5, 1.6, 1.7, 1.8):
            scheduler.step(value)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 2.5e-5)

    def test_early_stopping_uses_meaningful_delta(self):
        stopping = EarlyStopping(patience=2, min_delta=1e-4)
        self.assertTrue(stopping(1.0))
        self.assertFalse(stopping(0.99995))
        self.assertEqual(stopping.counter, 1)
        self.assertTrue(stopping(0.9998))
        self.assertEqual(stopping.counter, 0)

    def test_power_metrics_are_safe_for_zero_heavy_targets(self):
        true = np.array([0.0, 0.5, 10.0])
        pred = np.array([1.0, 1.0, 8.0])
        metrics = cal_metric(pred, true)
        self.assertIsNotNone(metrics['wape'])
        self.assertGreaterEqual(metrics['smape'], 0.0)
        self.assertLessEqual(metrics['smape'], 200.0)
        self.assertEqual(Kweather_metric(pred, np.zeros_like(true)), (None, None))


if __name__ == '__main__':
    unittest.main()
