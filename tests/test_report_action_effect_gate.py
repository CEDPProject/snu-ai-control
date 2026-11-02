"""Contracts for the frozen-report-model first-action gate."""

import unittest

import numpy as np

from controllers.report_action_effect_gate import ReportActionEffectGate
from controllers.total_power_evaluation import ComponentAffineCalibratedReportModel


class _ActionSensitiveReportModel:
    """Two component outputs driven only by the second one-hot feature."""

    def predict_components(self, states, controls, disturbances):
        controls = np.asarray(controls, dtype=float)
        # OFF/reference = (6 W ventilation, 4 W HVAC); ON/candidate adds
        # 3 W to the total, allowing the threshold contract to be visible.
        return np.column_stack((
            np.full(len(controls), 6.0),
            4.0 + 3.0 * controls[:, 1],
        ))

    def to_metadata(self):
        return {'family': 'unit_test_action_sensitive_report'}


class _MetadataReportModel(_ActionSensitiveReportModel):
    family = 'unit_test_base'
    input_feature_metadata = {
        'enabled_groups': ['current_state', 'action'],
        'feature_dimension': 3,
    }
    input_selection = object()

    def to_metadata(self):
        return {
            'model_family': self.family,
            'input_features': dict(self.input_feature_metadata),
        }


class ReportActionEffectGateTest(unittest.TestCase):
    def test_calibration_wrapper_preserves_report_input_interface(self):
        wrapped = ComponentAffineCalibratedReportModel(
            base_model=_MetadataReportModel(),
            slopes=np.array((1.0, 1.0)),
            intercepts=np.array((0.0, 0.0)),
            calibration_metadata={'status': 'unit_test'},
        )
        self.assertEqual(wrapped.family, 'unit_test_base')
        self.assertEqual(
            wrapped.input_feature_metadata['enabled_groups'],
            ['current_state', 'action'],
        )
        self.assertEqual(
            wrapped.to_metadata()['input_features']['feature_dimension'], 3
        )

    def test_total_limit_rejects_large_modelled_change(self):
        gate = ReportActionEffectGate(
            _ActionSensitiveReportModel(),
            ('ventilation_w', 'hvac_w'),
            {
                'enabled': True,
                'maximum_total_relative_change_percent': 20.0,
                'maximum_component_relative_change_percent': None,
            },
        )
        decisions = gate.evaluate_candidates(
            state=np.array([20.0]),
            disturbance=np.array([1.0]),
            candidates=np.array(((1.0, 0.0), (0.0, 1.0))),
            reference_control=np.array((1.0, 0.0)),
        )

        self.assertTrue(decisions[0].eligible)
        self.assertFalse(decisions[1].eligible)
        self.assertIn(
            'report_total_action_effect_above_limit',
            decisions[1].rejection_reasons,
        )
        self.assertAlmostEqual(decisions[1].total_relative_change_percent, 30.0)

    def test_component_limit_can_reject_even_when_total_is_allowed(self):
        gate = ReportActionEffectGate(
            _ActionSensitiveReportModel(),
            ('ventilation_w', 'hvac_w'),
            {
                'enabled': True,
                'maximum_total_relative_change_percent': 40.0,
                'maximum_component_relative_change_percent': 50.0,
            },
        )
        decision = gate.evaluate_candidates(
            state=np.array([20.0]),
            disturbance=np.array([1.0]),
            candidates=np.array(((0.0, 1.0),)),
            reference_control=np.array((1.0, 0.0)),
        )[0]

        self.assertFalse(decision.eligible)
        self.assertIn(
            'report_component_action_effect_above_limit',
            decision.rejection_reasons,
        )
        self.assertEqual(
            decision.component_relative_change_percent,
            (0.0, 75.0),
        )


if __name__ == '__main__':
    unittest.main()
