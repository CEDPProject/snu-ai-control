from dataclasses import replace
import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from data_provider.control_data import ControlData, ControlEncoder, ControlVariable
from dynamics.component_power_evaluator import ComponentSeparatedTwoStageHVACEvaluator
from dynamics.evaluator_factory import create_independent_evaluator


def _component_data(rows: int = 180) -> ControlData:
    encoder = ControlEncoder(
        [
            ControlVariable('환기전원', ('OFF', 'ON'), 'OFF'),
            ControlVariable('냉난방전원', ('OFF', 'ON'), 'OFF'),
        ]
    )
    controls = np.empty((rows, encoder.n_features), dtype=float)
    for step in range(rows):
        controls[step] = encoder.encode_action({
            '환기전원': 'ON' if step % 3 else 'OFF',
            '냉난방전원': 'ON' if step % 4 in (1, 2) else 'OFF',
        })
    disturbances = (15.0 + np.sin(np.arange(rows) / 7.0))[:, None]
    states = np.empty((rows, 2), dtype=float)
    powers = np.empty((rows, 2), dtype=float)
    states[0] = [23.0, 650.0]
    for step in range(rows - 1):
        ventilation_on = controls[step, encoder.index('환기전원', 'ON')]
        hvac_on = controls[step, encoder.index('냉난방전원', 'ON')]
        powers[step, 0] = 4.0 + 20.0 * ventilation_on
        powers[step, 1] = 0.0 if not hvac_on else 80.0 + 3.0 * disturbances[step, 0]
        states[step + 1, 0] = (
            states[step, 0]
            + 0.03 * (disturbances[step, 0] - states[step, 0])
            - 0.12 * hvac_on
        )
        states[step + 1, 1] = states[step, 1] + 1.5 - 3.0 * ventilation_on
    ventilation_on = controls[-1, encoder.index('환기전원', 'ON')]
    hvac_on = controls[-1, encoder.index('냉난방전원', 'ON')]
    powers[-1] = [4.0 + 20.0 * ventilation_on, 0.0 if not hvac_on else 125.0]
    return ControlData(
        timestamps=pd.Series(pd.date_range('2025-01-01', periods=rows, freq='min')),
        states=states,
        controls=controls,
        disturbances=disturbances,
        powers=powers,
        state_names=('실내온도(℃)', '실내이산화탄소(ppm)'),
        disturbance_names=('실외온도(℃)',),
        power_names=('환기유효전력(W)', '냉난방유효전력(W)'),
        encoder=encoder,
    )


class ComponentSeparatedTwoStageHVACTest(unittest.TestCase):
    def test_factory_predicts_separate_components_and_hvac_mixture(self):
        data = _component_data()
        config = OmegaConf.create({
            'model_family': 'component_two_stage_hvac',
            'n_estimators': 32,
            'min_samples_leaf': 1,
            'max_features': 1.0,
            'n_jobs': 1,
            'component_power': {
                'hvac_active_threshold_w': 2.5,
            },
        })
        evaluator = create_independent_evaluator(
            config,
            data,
            training_end_exclusive=140,
            random_state=42,
        )
        self.assertIsInstance(evaluator, ComponentSeparatedTwoStageHVACEvaluator)

        rows = np.arange(140, 170)
        states, controls, disturbances, _, _ = data.model_arrays_for_rows(rows)
        breakdown = evaluator.predict_power_breakdown(
            states,
            controls,
            disturbances,
        )
        predicted = breakdown['component_power_w']
        self.assertEqual(predicted.shape, (len(rows), 2))
        self.assertTrue(np.isfinite(predicted).all())
        self.assertTrue((predicted >= 0.0).all())
        self.assertTrue(
            ((breakdown['hvac_active_probability'] >= 0.0)
             & (breakdown['hvac_active_probability'] <= 1.0)).all()
        )

        hvac_on = controls[:, data.encoder.index('냉난방전원', 'ON')] > 0.5
        self.assertGreater(
            predicted[hvac_on, 1].mean(),
            predicted[~hvac_on, 1].mean() + 20.0,
        )
        metadata = evaluator.to_metadata()
        self.assertEqual(metadata['model_family'], 'component_two_stage_hvac')
        self.assertEqual(
            metadata['component_modeling']['hvac_power_name'],
            '냉난방유효전력(W)',
        )
        self.assertEqual(metadata['parameters']['hvac_active_training_rows'], 70)
        self.assertEqual(
            metadata['component_modeling']['hvac_active_threshold_w'],
            2.5,
        )

    def test_factory_respects_explicit_non_token_hvac_meter_name(self):
        """An explicit meter must win even when another name matches a token."""
        data = replace(
            _component_data(),
            power_names=('legacy_hvac_label_w', 'zone_b_compressor_w'),
        )
        config = OmegaConf.create({
            'model_family': 'component_two_stage_hvac',
            'n_estimators': 24,
            'min_samples_leaf': 1,
            'max_features': 1.0,
            'n_jobs': 1,
            'component_power': {
                'hvac_active_threshold_w': 0.1,
                # "zone_b_compressor_w" has no HVAC/냉난방 token.  Without
                # this explicit configuration, automatic inference would pick
                # the first legacy-labelled meter instead.
                'hvac_component_name': 'zone_b_compressor_w',
            },
        })

        evaluator = create_independent_evaluator(
            config,
            data,
            training_end_exclusive=140,
            random_state=7,
        )

        self.assertEqual(evaluator.hvac_power_index, 1)
        metadata = evaluator.to_metadata()
        self.assertEqual(
            metadata['component_modeling']['hvac_power_name'],
            'zone_b_compressor_w',
        )
        self.assertEqual(
            metadata['parameters']['hvac_component_name'],
            'zone_b_compressor_w',
        )
        self.assertEqual(
            metadata['component_modeling']['direct_regression_power_names'],
            ['legacy_hvac_label_w'],
        )


if __name__ == '__main__':
    unittest.main()
