import unittest

from controllers.candidate_pool_evaluation import (
    _fit_parameters as candidate_pool_fit_parameters,
)
from controllers.total_power_evaluation import (
    _fit_parameters as chronological_fit_parameters,
)


class ComponentPowerReportWiringTest(unittest.TestCase):
    @staticmethod
    def _config():
        return {
            'minimum_action_samples': 5,
            'n_estimators': 20,
            'min_samples_leaf': 2,
            'max_features': 1.0,
            'huber_epsilon': 1.35,
            'huber_alpha': 0.0001,
            'huber_max_iter': 100,
            'n_jobs': 1,
            'hvac_low_power_threshold_w': 3.25,
            'hvac_component_name': '냉난방유효전력(W)',
            'hvac_minimum_active_samples': 7,
        }

    def test_all_report_protocols_forward_two_stage_hvac_parameters(self):
        for parameters in (
            chronological_fit_parameters(self._config()),
            candidate_pool_fit_parameters(self._config()),
        ):
            self.assertEqual(parameters['hvac_low_power_threshold_w'], 3.25)
            self.assertEqual(
                parameters['hvac_component_name'],
                '냉난방유효전력(W)',
            )
            self.assertEqual(parameters['hvac_minimum_active_samples'], 7)

    def test_legacy_report_configuration_gets_safe_two_stage_defaults(self):
        config = self._config()
        config.pop('hvac_low_power_threshold_w')
        config.pop('hvac_component_name')
        config.pop('hvac_minimum_active_samples')
        for parameters in (
            chronological_fit_parameters(config),
            candidate_pool_fit_parameters(config),
        ):
            self.assertEqual(parameters['hvac_low_power_threshold_w'], 0.1)
            self.assertIsNone(parameters['hvac_component_name'])
            self.assertIsNone(parameters['hvac_minimum_active_samples'])


if __name__ == '__main__':
    unittest.main()
