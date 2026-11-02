"""Focused safety contracts for first-action model screening and gating."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from controllers.action_model_screen import ActionModelScreen
from controllers.action_selection_gate import ActionSelectionGate
from controllers.action_trust import ActionTrustEligibility
from controllers.evaluation import solve_first_action_with_gate
from controllers.mixed_integer_mpc import MixedIntegerMPC
from data_provider.control_data import ControlData, ControlEncoder, ControlVariable
from dynamics.affine import AffinePowerModel, AffineStateModel


class _AffinePreview:
    """One-component controller proxy: both actions look equally cheap."""

    def predict_components(self, states, controls, disturbances):
        return np.full((len(np.asarray(states)), 1), 2.0, dtype=float)


class _PowerOnlyEvaluator:
    """Independent evaluator that disagrees only for the ON action."""

    def predict_power(self, states, controls, disturbances):
        controls = np.asarray(controls, dtype=float)
        return (2.0 + 4.0 * controls[:, 1])[:, None]


class _CountingPowerOnlyEvaluator(_PowerOnlyEvaluator):
    def __init__(self):
        self.power_calls = 0

    def predict_power(self, states, controls, disturbances):
        self.power_calls += 1
        return super().predict_power(states, controls, disturbances)


class _NonFinitePowerEvaluator:
    def predict_power(self, states, controls, disturbances):
        return np.full((len(np.asarray(states)), 1), np.nan, dtype=float)


class _OnlyOffGate:
    """Minimal gate stub used to test the controller-domain boundary."""

    def __init__(self, off):
        self.off = np.asarray(off, dtype=float)

    def evaluate(self, state, disturbance, *, evaluator):
        decision = SimpleNamespace(
            encoded_control=tuple(np.rint(self.off).astype(int).tolist()),
            eligible=True,
            rejection_reasons=(),
            trust=None,
            model_screen=None,
        )
        return self.off[None, :], [decision]


class ActionModelScreenTest(unittest.TestCase):
    @staticmethod
    def _data():
        encoder = ControlEncoder(
            [ControlVariable('switch', ('OFF', 'ON'), 'OFF')]
        )
        controls = np.vstack(
            [
                encoder.encode_action({'switch': ('OFF' if index % 2 == 0 else 'ON')})
                for index in range(8)
            ]
        )
        return ControlData(
            timestamps=pd.Series(
                pd.date_range('2026-01-01', periods=8, freq='min')
            ),
            states=np.arange(8, dtype=float)[:, None],
            controls=controls,
            disturbances=np.zeros((8, 1), dtype=float),
            powers=np.full((8, 1), 2.0, dtype=float),
            state_names=('comfort',),
            disturbance_names=('outdoor',),
            power_names=('total_power_w',),
            encoder=encoder,
        )

    @staticmethod
    def _screen(data, evaluator):
        return ActionModelScreen.calibrate(
            data,
            validation_rows=np.arange(4, 7),
            affine_power_model=_AffinePreview(),
            validation_evaluator=evaluator,
            state_constraints=[
                {'name': 'comfort', 'lower': None, 'upper': None, 'penalty': 1.0}
            ],
            config={
                'enabled': True,
                'unavailable_model_policy': 'reject',
                'power_disagreement': {
                    'enabled': True,
                    'power_floor_w': 1.0,
                    'maximum_normalized_component_disagreement': 0.2,
                },
                # Deliberately no predict_state in _PowerOnlyEvaluator.  The
                # power-only configuration must remain usable.
                'state_safety': {'enabled': False},
            },
        ).bind_affine_power_model(_AffinePreview())

    def test_power_only_screen_filters_disagreeing_action_and_gate_preserves_audit(self):
        data = self._data()
        evaluator = _PowerOnlyEvaluator()
        screen = self._screen(data, evaluator)
        trust = ActionTrustEligibility.build(
            data,
            training_rows=np.arange(4),
            config={
                'global_support': {'minimum_samples': 1, 'minimum_episodes': 1},
                'local_context_support': {'enabled': False},
            },
            candidate_controls=np.vstack(
                (
                    data.encoder.encode_action({'switch': 'OFF'}),
                    data.encoder.encode_action({'switch': 'ON'}),
                )
            ),
        )
        gate = ActionSelectionGate(trust, screen)

        allowed, decisions = gate.evaluate(
            np.array([0.0]),
            np.array([0.0]),
            evaluator=evaluator,
        )

        self.assertEqual(len(allowed), 1)
        self.assertEqual(
            data.encoder.decode_action(allowed[0]), {'switch': 'OFF'}
        )
        by_action = {
            decision.trust.action['switch']: decision for decision in decisions
        }
        self.assertTrue(by_action['OFF'].eligible)
        self.assertFalse(by_action['ON'].eligible)
        self.assertIn(
            'affine_evaluator_power_disagreement_above_threshold',
            by_action['ON'].rejection_reasons,
        )
        self.assertEqual(
            by_action['ON'].model_screen.state_safety['status'],
            'disabled',
        )

    def test_gate_batches_model_preview_for_all_trusted_actions(self):
        data = self._data()
        evaluator = _CountingPowerOnlyEvaluator()
        screen = self._screen(data, evaluator)
        trust = ActionTrustEligibility.build(
            data,
            training_rows=np.arange(4),
            config={'local_context_support': {'enabled': False}},
            candidate_controls=np.vstack(
                (
                    data.encoder.encode_action({'switch': 'OFF'}),
                    data.encoder.encode_action({'switch': 'ON'}),
                )
            ),
        )
        gate = ActionSelectionGate(trust, screen)
        calls_after_calibration = evaluator.power_calls

        gate.evaluate(
            np.array([0.0]),
            np.array([0.0]),
            evaluator=evaluator,
        )

        # Both candidates are screened, but the independent tree-style model
        # receives one two-row batch rather than one call per action.
        self.assertEqual(evaluator.power_calls, calls_after_calibration + 1)

    def test_nonfinite_power_preview_is_never_silently_accepted(self):
        data = self._data()
        # Calibration appropriately fails fast when a train/validation model
        # emits non-finite values rather than deriving a NaN threshold.
        with self.assertRaisesRegex(ValueError, 'finite'):
            self._screen(data, _NonFinitePowerEvaluator())

    def test_gate_restricts_controller_domain_not_just_inspected_actions(self):
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
        # The unconstrained MILP prefers ON (1 W) to OFF (10 W).
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
        )
        off = encoder.encode_action({'switch': 'OFF'})
        result = solve_first_action_with_gate(
            controller,
            initial_state=np.array([0.0]),
            disturbances=np.empty((1, 0)),
            previous_control=off,
            evaluator=object(),
            action_gate=_OnlyOffGate(off),
        )

        self.assertTrue(result.has_incumbent)
        self.assertFalse(result.used_fallback)
        self.assertEqual(
            encoder.decode_action(result.applied_control), {'switch': 'OFF'}
        )
        self.assertEqual(result.selected_decision.encoded_control, tuple(off.astype(int)))


if __name__ == '__main__':
    unittest.main()


class _StateEvaluator:
    """Next state = state + per-level effect; LOW cools, HIGH heats."""

    EFFECT = np.array([-0.5, 0.0, 0.5])

    def predict_state(self, states, controls, disturbances):
        states = np.asarray(states, dtype=float)
        controls = np.asarray(controls, dtype=float)
        return states + (controls @ self.EFFECT)[:, None]

    def predict_power(self, states, controls, disturbances):
        return np.ones((len(np.asarray(states)), 1), dtype=float)


class OutOfBandStateSafetyPolicyTest(unittest.TestCase):
    """The non-worsening policy relaxes the screen only when no action is in band."""

    @staticmethod
    def _setup(policy):
        from controllers.action_model_screen import ActionModelScreenConfig

        encoder = ControlEncoder(
            [ControlVariable('level', ('LOW', 'MID', 'HIGH'), 'MID')]
        )
        screen = ActionModelScreen(
            config=ActionModelScreenConfig.from_mapping({
                'enabled': True,
                'power_disagreement': {'enabled': False},
                'state_safety': {'enabled': True, 'out_of_band_policy': policy},
            }),
            state_constraints=[
                {'name': 'comfort', 'lower': None, 'upper': 10.0, 'penalty': 1.0}
            ],
            power_names=('total_power_w',),
            state_names=('comfort',),
            disagreement_threshold=None,
            upper_state_margins=np.zeros(1),
            lower_state_margins=np.zeros(1),
            calibration_metadata={'status': 'calibrated'},
        )
        controls = np.vstack(
            [encoder.encode_action({'level': level}) for level in ('LOW', 'MID', 'HIGH')]
        )
        return screen, controls, encoder.encode_action({'level': 'MID'})

    def _eligible(self, policy, state):
        screen, controls, reference = self._setup(policy)
        decisions = screen.evaluate_candidates(
            np.array([state]), np.array([0.0]), controls,
            evaluator=_StateEvaluator(), reference_control=reference,
        )
        return [decision.eligible for decision in decisions], decisions

    def test_default_reject_policy_keeps_original_behavior(self):
        eligible, decisions = self._eligible('reject', 12.0)
        self.assertEqual(eligible, [False, False, False])
        self.assertNotIn('state_safety_relaxed_non_worsening', decisions[0].notes)
        screen, _, _ = self._setup('reject')
        self.assertFalse(screen.requires_reference_control)

    def test_out_of_band_state_accepts_only_non_worsening_actions(self):
        eligible, decisions = self._eligible('non_worsening', 12.0)
        # LOW improves, MID equals holding the reference, HIGH worsens.
        self.assertEqual(eligible, [True, True, False])
        self.assertIn('state_safety_relaxed_non_worsening', decisions[0].notes)
        self.assertIn(
            'evaluator_state_non_worsening_failed:comfort',
            decisions[2].rejection_reasons,
        )

    def test_relaxation_is_not_used_while_any_action_is_in_band(self):
        eligible, decisions = self._eligible('non_worsening', 9.8)
        # LOW and MID stay in band, so the original rule still rejects HIGH.
        self.assertEqual(eligible, [True, True, False])
        self.assertNotIn('state_safety_relaxed_non_worsening', decisions[0].notes)
        self.assertIn(
            'evaluator_state_safety_margin_failed:comfort',
            decisions[2].rejection_reasons,
        )

    def test_invalid_policy_is_rejected(self):
        from controllers.action_model_screen import ActionModelScreenConfig

        with self.assertRaisesRegex(ValueError, 'out_of_band_policy'):
            ActionModelScreenConfig.from_mapping(
                {'state_safety': {'out_of_band_policy': 'ignore'}}
            )

    def test_gate_forwards_previous_action_and_reports_relaxation(self):
        screen, controls, reference = self._setup('non_worsening')
        decisions = screen.evaluate_candidates(
            np.array([12.0]), np.array([0.0]), controls,
            evaluator=_StateEvaluator(), reference_control=None,
        )
        # Without a reference action the policy cannot relax.
        self.assertEqual([d.eligible for d in decisions], [False, False, False])

        trust = SimpleNamespace(candidate_controls=controls)
        gate = ActionSelectionGate(trust, screen)
        self.assertTrue(gate.requires_reference_control)
        trust_decisions = [
            SimpleNamespace(encoded_control=tuple(row.astype(int)), eligible=True,
                            rejection_reasons=())
            for row in controls
        ]
        trust.evaluate_candidates = lambda state, disturbance: trust_decisions
        trust.data = SimpleNamespace(encoder=SimpleNamespace(n_features=3))
        allowed, gate_decisions = gate.evaluate(
            np.array([12.0]), np.array([0.0]),
            evaluator=_StateEvaluator(), reference_control=reference,
        )
        self.assertEqual(len(allowed), 2)
        from controllers.evaluation import GatedMPCSolveResult

        summary = GatedMPCSolveResult(
            solution=None, applied_control=reference, has_incumbent=False,
            used_fallback=False, fallback_reason=None, retry_count=0,
            candidate_decisions=tuple(gate_decisions), selected_decision=None,
        ).summary()
        self.assertTrue(summary['state_safety_relaxed_non_worsening'])
