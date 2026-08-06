import unittest

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from controllers.evaluation import evaluate_recursive_rollouts
from controllers.pipeline import _select_evaluation_windows
from data_provider.control_data import (
    ControlData,
    ControlEncoder,
    ControlVariable,
)
from dynamics.evaluator_factory import (
    available_evaluator_families,
    create_independent_evaluator,
)
from models import create_model


def _control_data(rows=120):
    encoder = ControlEncoder(
        [ControlVariable('switch', ('OFF', 'ON'), 'OFF')]
    )
    controls = np.zeros((rows, 2), dtype=float)
    controls[np.arange(rows), np.arange(rows) % 2] = 1.0
    disturbances = (15.0 + np.sin(np.arange(rows) / 10.0))[:, None]
    states = np.empty((rows, 2), dtype=float)
    powers = np.empty((rows, 1), dtype=float)
    states[0] = [22.0, 500.0]
    for step in range(rows - 1):
        enabled = controls[step, 1]
        powers[step, 0] = 20.0 + 80.0 * enabled
        states[step + 1, 0] = (
            states[step, 0]
            + 0.03 * (disturbances[step, 0] - states[step, 0])
            - 0.08 * enabled
        )
        states[step + 1, 1] = states[step, 1] + 1.0 - 2.0 * enabled
    powers[-1, 0] = 20.0 + 80.0 * controls[-1, 1]
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
        power_names=('power',),
        encoder=encoder,
    )


def _evaluator_config():
    return OmegaConf.create(
        {
            'model_family': 'extra_trees_residual_state',
            'validation_model_families': [],
            'n_estimators': 8,
            'min_samples_leaf': 2,
            'max_features': 1.0,
            'max_iter': 8,
            'n_jobs': 1,
            'model_setting_path': None,
            'neural_residual': {
                'ridge_alpha': 1.0,
                'hidden_sizes': [8],
                'activation': 'gelu',
                'dropout': 0.0,
                'epochs': 1,
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 0.0,
                'patience': 1,
                'min_delta': 1e-6,
                'validation_fraction': 0.1,
                'device': 'cpu',
            },
            'forecasting': {
                'model_name': 'DLinear',
                'model_parameters': {
                    'moving_avg': 3,
                    'individual': False,
                    'cat_emb_dim': 4,
                },
                'lookback_steps': 8,
                'include_time_features': True,
                'epochs': 1,
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 0.0,
                'patience': 1,
                'min_delta': 1e-6,
                'validation_fraction': 0.1,
                'device': 'cpu',
            },
        }
    )


class EvaluatorFactoryContractTest(unittest.TestCase):
    def test_sklearn_families_share_prediction_contract(self):
        data = _control_data()
        config = _evaluator_config()
        for family in (
            'extra_trees_residual_state',
            'random_forest_residual_state',
            'hist_gradient_boosting_residual_state',
        ):
            evaluator = create_independent_evaluator(
                config,
                data,
                100,
                random_state=42,
                model_family=family,
            )
            prediction = evaluator.predict_transitions(data, 100, 119)
            self.assertEqual(prediction.predicted_states.shape, (19, 2))
            self.assertEqual(prediction.predicted_powers.shape, (19, 1))
            np.testing.assert_array_equal(
                prediction.row_indices, np.arange(100, 119)
            )

    def test_neural_residual_uses_independent_linear_baseline(self):
        data = _control_data()
        evaluator = create_independent_evaluator(
            _evaluator_config(),
            data,
            100,
            random_state=42,
            model_family='neural_residual',
        )
        metadata = evaluator.to_metadata()
        self.assertEqual(metadata['model_family'], 'neural_residual')
        self.assertIn('Ridge', metadata['state_equation'])
        stability = metadata['parameters']['linear_dynamics_stability']
        self.assertLessEqual(
            stability['final_spectral_radius'],
            stability['maximum_spectral_radius'] + 1e-10,
        )
        prediction = evaluator.predict_transitions(data, 100, 119)
        self.assertTrue(np.isfinite(prediction.predicted_states).all())
        self.assertTrue((prediction.predicted_powers >= 0.0).all())

    def test_forecasting_model_has_history_aware_recursive_contract(self):
        data = _control_data()
        evaluator = create_independent_evaluator(
            _evaluator_config(),
            data,
            100,
            random_state=42,
            model_family='forecasting:DLinear',
        )
        metadata = evaluator.to_metadata()
        self.assertEqual(metadata['forecasting_model_name'], 'DLinear')
        self.assertEqual(metadata['lookback_steps'], 8)
        prediction = evaluator.predict_transitions(data, 100, 119)
        np.testing.assert_array_equal(
            prediction.row_indices, np.arange(100, 119)
        )
        result = evaluate_recursive_rollouts(
            evaluator,
            data,
            starts=[100],
            horizon=10,
            state_constraints=[
                {'name': 'temperature', 'lower': 18.0, 'upper': 27.0},
                {'name': 'co2', 'lower': None, 'upper': 1000.0},
            ],
            mape_min_actual_power_w=0.1,
            lead_time_steps=[1, 10],
        )
        self.assertEqual(result['evaluated_steps'], 10)

    def test_every_forecasting_model_accepts_evaluator_metadata(self):
        settings = OmegaConf.load('src/configs/model_setting.yaml')
        available = available_evaluator_families()['forecasting_sequence']
        self.assertEqual(set(available), set(settings.keys()))
        metadata = {
            'seq_len': 120,
            'pred_len': 1,
            'input_numerical_features': ['state', 'control', 'power'],
            'target_features': ['state', 'power'],
            'feature_types': {'n_cat': 0, 'n_num': 3, 'n_target': 2},
        }
        values = torch.randn(1, 120, 3)
        for model_name in available:
            parameters = OmegaConf.to_container(
                settings[model_name],
                resolve=True,
            )
            parameters['METADATA'] = metadata
            model = create_model(model_name, parameters).eval()
            with torch.no_grad():
                output = model(None, values, None)
            self.assertEqual(tuple(output.shape), (1, 1, 2), model_name)


    def test_validation_windows_never_cross_the_split_boundary(self):
        data = _control_data()
        starts = _select_evaluation_windows(
            data,
            test_start=80,
            duration_steps=10,
            stride=5,
            max_windows=0,
            prediction_horizon=60,
            disturbance_forecast_method='persistence',
            end_exclusive=100,
        )
        self.assertTrue(starts)
        self.assertTrue(
            all(start + 10 < 100 for start in starts),
            starts,
        )


if __name__ == '__main__':
    unittest.main()
