"""Neural residual and forecasting-model evaluators for MPC backtests."""

from copy import deepcopy
from dataclasses import dataclass
import logging
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from dynamics.evaluator_base import (
    RolloutHistory,
    StatelessEvaluatorMixin,
    TransitionPredictions,
)


LOGGER = logging.getLogger('MPC.neural_evaluator')


def _as_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[:, None] if values.ndim == 1 else values


def _resolve_device(value: str) -> torch.device:
    value = str(value).strip().lower()
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA evaluator device requested but unavailable: {value}')
    return device


@dataclass(frozen=True)
class ArrayScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> 'ArrayScaler':
        values = np.asarray(values, dtype=float)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean


def _activation(name: str) -> nn.Module:
    activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'tanh': nn.Tanh,
    }
    key = str(name).lower()
    if key not in activations:
        raise ValueError(f'Unknown neural evaluator activation: {name!r}')
    return activations[key]()


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Sequence[int],
        activation: str,
        dropout: float,
    ):
        super().__init__()
        layers = []
        width = int(input_size)
        for hidden_size in hidden_sizes:
            hidden_size = int(hidden_size)
            if hidden_size <= 0:
                raise ValueError('Every residual MLP hidden size must be positive')
            layers.extend(
                [nn.Linear(width, hidden_size), _activation(activation)]
            )
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            width = hidden_size
        layers.append(nn.Linear(width, int(output_size)))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _train_tensor_model(
    model: nn.Module,
    train_dataset,
    validation_dataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    device: torch.device,
    seed: int,
    sequence_model: bool,
) -> Dict[str, object]:
    if int(epochs) <= 0 or int(batch_size) <= 0:
        raise ValueError('Neural evaluator epochs and batch_size must be positive')
    torch.manual_seed(int(seed))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(int(seed))
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(batch_size),
        shuffle=False,
    )
    model.to(device)
    LOGGER.info(
        'Neural evaluator training started: model=%s | device=%s | train=%d | '
        'validation=%d | epochs=%d | batch_size=%d',
        model.__class__.__name__,
        device,
        len(train_dataset),
        len(validation_dataset),
        int(epochs),
        int(batch_size),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    criterion = nn.MSELoss()
    best_loss = float('inf')
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0
    completed_epochs = 0

    def predict(batch_x):
        if sequence_model:
            return model(None, batch_x, None)[:, -1, :]
        return model(batch_x)

    for epoch in range(int(epochs)):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(predict(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        loss_sum = 0.0
        sample_count = 0
        with torch.no_grad():
            for batch_x, batch_y in validation_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_loss = criterion(predict(batch_x), batch_y)
                loss_sum += float(batch_loss) * len(batch_x)
                sample_count += len(batch_x)
        validation_loss = loss_sum / max(1, sample_count)
        completed_epochs = epoch + 1
        improved = validation_loss < best_loss - float(min_delta)
        if improved:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
        LOGGER.info(
            'Neural evaluator epoch %d/%d: validation_mse_scaled=%.8f | '
            'best=%.8f | stale_epochs=%d',
            epoch + 1,
            int(epochs),
            validation_loss,
            best_loss,
            stale_epochs,
        )
        if not improved and stale_epochs >= int(patience):
            LOGGER.info(
                'Neural evaluator early stopping: completed_epochs=%d | '
                'best_epoch=%d | best_validation_mse_scaled=%.8f',
                completed_epochs,
                best_epoch,
                best_loss,
            )
            break

    model.load_state_dict(best_state)
    model.eval()
    LOGGER.info(
        'Neural evaluator training completed: completed_epochs=%d | '
        'best_epoch=%d | best_validation_mse_scaled=%.8f',
        completed_epochs,
        best_epoch,
        best_loss,
    )
    return {
        'epochs_requested': int(epochs),
        'epochs_completed': int(completed_epochs),
        'best_epoch': int(best_epoch),
        'best_validation_mse_scaled': float(best_loss),
        'train_samples': int(len(train_dataset)),
        'validation_samples': int(len(validation_dataset)),
        'device': str(device),
    }


def _split_count(sample_count: int, validation_fraction: float) -> int:
    if sample_count < 4:
        raise ValueError('Neural evaluator requires at least four training samples')
    if not 0.0 < float(validation_fraction) < 0.5:
        raise ValueError('validation_fraction must be between 0 and 0.5')
    validation_count = max(1, int(round(sample_count * validation_fraction)))
    return min(sample_count - 1, sample_count - validation_count)


def _stabilize_state_ridge(
    estimator: Ridge,
    state_count: int,
    maximum_spectral_radius: float,
    enabled: bool,
) -> Dict[str, object]:
    """Bound the autonomous linear-state radius before residual fitting."""

    maximum = float(maximum_spectral_radius)
    if maximum <= 0.0 or maximum > 1.0:
        raise ValueError('maximum_spectral_radius must be in (0, 1]')
    coefficients = np.asarray(estimator.coef_, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[0] != int(state_count):
        raise ValueError('State Ridge coefficients do not match state dimensions')
    transition = np.eye(state_count) + coefficients[:, :state_count]
    original_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    adjusted = bool(enabled) and original_radius > maximum
    if adjusted:
        transition *= maximum / original_radius
        coefficients[:, :state_count] = transition - np.eye(state_count)
        estimator.coef_ = coefficients
    final_transition = np.eye(state_count) + np.asarray(
        estimator.coef_,
        dtype=float,
    )[:, :state_count]
    final_radius = float(np.max(np.abs(np.linalg.eigvals(final_transition))))
    return {
        'enabled': bool(enabled),
        'maximum_spectral_radius': maximum,
        'original_spectral_radius': original_radius,
        'adjusted': adjusted,
        'final_spectral_radius': final_radius,
    }


@dataclass
class NeuralResidualEvaluator(StatelessEvaluatorMixin):
    """Independent Ridge dynamics with a neural residual correction."""

    base_state_estimator: Ridge
    base_power_estimator: Ridge
    residual_model: ResidualMLP
    input_scaler: ArrayScaler
    residual_scaler: ArrayScaler
    device: torch.device
    state_names: tuple[str, ...]
    control_feature_names: tuple[str, ...]
    disturbance_names: tuple[str, ...]
    power_names: tuple[str, ...]
    training_transition_count: int
    parameters: Dict[str, object]
    training_summary: Dict[str, object]

    @classmethod
    def fit(
        cls,
        states: np.ndarray,
        controls: np.ndarray,
        disturbances: np.ndarray,
        next_states: np.ndarray,
        powers: np.ndarray,
        state_names: Sequence[str],
        control_feature_names: Sequence[str],
        disturbance_names: Sequence[str],
        power_names: Sequence[str],
        *,
        ridge_alpha: float = 1.0,
        stabilize_linear_dynamics: bool = True,
        maximum_spectral_radius: float = 0.999,
        hidden_sizes: Sequence[int] = (128, 128),
        activation: str = 'gelu',
        dropout: float = 0.1,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5,
        min_delta: float = 1e-5,
        validation_fraction: float = 0.1,
        device: str = 'auto',
        random_state: int = 42,
    ) -> 'NeuralResidualEvaluator':
        arrays = [
            np.asarray(values, dtype=float)
            for values in (states, controls, disturbances, next_states, powers)
        ]
        states, controls, disturbances, next_states, powers = arrays
        row_counts = {len(values) for values in arrays}
        if len(row_counts) != 1 or not row_counts or len(states) == 0:
            raise ValueError('Residual evaluator arrays must have equal nonzero rows')
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError('Residual evaluator arrays must be finite')

        design = np.hstack([states, controls, disturbances])
        state_delta = next_states - states
        split = _split_count(len(design), validation_fraction)
        base_state = Ridge(alpha=float(ridge_alpha)).fit(
            design[:split],
            state_delta[:split],
        )
        stability_summary = _stabilize_state_ridge(
            base_state,
            len(state_names),
            maximum_spectral_radius,
            stabilize_linear_dynamics,
        )
        base_power = Ridge(alpha=float(ridge_alpha)).fit(
            design[:split],
            powers[:split],
        )
        base_state_prediction = _as_2d(base_state.predict(design))
        base_power_prediction = _as_2d(base_power.predict(design))
        residual_target = np.hstack(
            [
                state_delta - base_state_prediction,
                powers - base_power_prediction,
            ]
        )
        input_scaler = ArrayScaler.fit(design[:split])
        residual_scaler = ArrayScaler.fit(residual_target[:split])
        scaled_design = input_scaler.transform(design).astype(np.float32)
        scaled_target = residual_scaler.transform(residual_target).astype(np.float32)
        train_dataset = TensorDataset(
            torch.from_numpy(scaled_design[:split]),
            torch.from_numpy(scaled_target[:split]),
        )
        validation_dataset = TensorDataset(
            torch.from_numpy(scaled_design[split:]),
            torch.from_numpy(scaled_target[split:]),
        )
        torch.manual_seed(int(random_state))
        residual_model = ResidualMLP(
            input_size=design.shape[1],
            output_size=residual_target.shape[1],
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
        )
        resolved_device = _resolve_device(device)
        training_summary = _train_tensor_model(
            residual_model,
            train_dataset,
            validation_dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            device=resolved_device,
            seed=random_state,
            sequence_model=False,
        )
        return cls(
            base_state_estimator=base_state,
            base_power_estimator=base_power,
            residual_model=residual_model,
            input_scaler=input_scaler,
            residual_scaler=residual_scaler,
            device=resolved_device,
            state_names=tuple(state_names),
            control_feature_names=tuple(control_feature_names),
            disturbance_names=tuple(disturbance_names),
            power_names=tuple(power_names),
            training_transition_count=int(len(states)),
            parameters={
                'ridge_alpha': float(ridge_alpha),
                'linear_dynamics_stability': stability_summary,
                'stabilize_linear_dynamics': bool(stabilize_linear_dynamics),
                'maximum_spectral_radius': float(maximum_spectral_radius),
                'hidden_sizes': [int(value) for value in hidden_sizes],
                'activation': str(activation),
                'dropout': float(dropout),
                'epochs': int(epochs),
                'batch_size': int(batch_size),
                'learning_rate': float(learning_rate),
                'weight_decay': float(weight_decay),
                'patience': int(patience),
                'min_delta': float(min_delta),
                'validation_fraction': float(validation_fraction),
                'random_state': int(random_state),
            },
            training_summary=training_summary,
        )

    def _design(self, states, controls, disturbances) -> tuple[np.ndarray, np.ndarray]:
        states = np.atleast_2d(np.asarray(states, dtype=float))
        controls = np.atleast_2d(np.asarray(controls, dtype=float))
        disturbances = np.atleast_2d(np.asarray(disturbances, dtype=float))
        if not (len(states) == len(controls) == len(disturbances)):
            raise ValueError('Evaluator inputs must have the same row count')
        return states, np.hstack([states, controls, disturbances])

    def _residual(self, design: np.ndarray) -> np.ndarray:
        scaled = self.input_scaler.transform(design).astype(np.float32)
        with torch.no_grad():
            prediction = self.residual_model(
                torch.from_numpy(scaled).to(self.device)
            ).cpu().numpy()
        return self.residual_scaler.inverse_transform(prediction)

    def _predictions(self, states, controls, disturbances):
        states, design = self._design(states, controls, disturbances)
        residual = self._residual(design)
        state_count = len(self.state_names)
        state_delta = _as_2d(self.base_state_estimator.predict(design))
        power = _as_2d(self.base_power_estimator.predict(design))
        next_state = states + state_delta + residual[:, :state_count]
        power = np.maximum(power + residual[:, state_count:], 0.0)
        if not (np.isfinite(next_state).all() and np.isfinite(power).all()):
            raise FloatingPointError('Neural residual evaluator produced non-finite output')
        return next_state, power

    def predict_state(self, states, controls, disturbances) -> np.ndarray:
        return self._predictions(states, controls, disturbances)[0]

    def predict_power(self, states, controls, disturbances) -> np.ndarray:
        return self._predictions(states, controls, disturbances)[1]

    def step(self, state, control, disturbance) -> tuple[np.ndarray, np.ndarray]:
        next_state, power = self._predictions(state, control, disturbance)
        return next_state[0], power[0]

    def to_metadata(self) -> Dict[str, object]:
        return {
            'role': 'independent nonlinear plant surrogate for offline evaluation',
            'model_family': 'neural_residual',
            'state_equation': 'x_next = x + Ridge([x,z,d]) + MLP_residual([x,z,d])',
            'power_equation': 'p = max(Ridge([x,z,d]) + MLP_residual([x,z,d]), 0)',
            'trained_on': 'chronological rows strictly before the configured training boundary',
            'training_transition_count': self.training_transition_count,
            'state_names': list(self.state_names),
            'control_feature_names': list(self.control_feature_names),
            'disturbance_names': list(self.disturbance_names),
            'power_names': list(self.power_names),
            'parameters': self.parameters,
            'training_summary': self.training_summary,
        }


class _SequenceDataset(Dataset):
    def __init__(
        self,
        scaled_features: np.ndarray,
        scaled_targets: np.ndarray,
        endpoints: np.ndarray,
        lookback_steps: int,
    ):
        self.features = torch.from_numpy(
            np.asarray(scaled_features, dtype=np.float32)
        )
        self.targets = torch.from_numpy(
            np.asarray(scaled_targets, dtype=np.float32)
        )
        self.endpoints = np.asarray(endpoints, dtype=int)
        self.lookback_steps = int(lookback_steps)

    def __len__(self) -> int:
        return int(len(self.endpoints))

    def __getitem__(self, position: int):
        endpoint = int(self.endpoints[position])
        start = endpoint - self.lookback_steps + 1
        return self.features[start : endpoint + 1], self.targets[endpoint]


def _time_features(timestamps) -> np.ndarray:
    index = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps)))
    minute = index.hour.to_numpy() * 60 + index.minute.to_numpy()
    weekday = index.dayofweek.to_numpy()
    return np.column_stack(
        [
            np.sin(2.0 * np.pi * minute / 1440.0),
            np.cos(2.0 * np.pi * minute / 1440.0),
            np.sin(2.0 * np.pi * weekday / 7.0),
            np.cos(2.0 * np.pi * weekday / 7.0),
        ]
    )


def _valid_sequence_endpoints(data, start: int, end: int, lookback_steps: int):
    timestamps = pd.DatetimeIndex(pd.to_datetime(data.timestamps))
    contiguous = np.asarray(
        (timestamps[1:] - timestamps[:-1]) == pd.Timedelta(minutes=1),
        dtype=bool,
    )
    bad_prefix = np.concatenate([[0], np.cumsum(~contiguous)])
    endpoints = []
    first = max(int(start), int(lookback_steps) - 1)
    last = min(int(end), len(data.states) - 1)
    for endpoint in range(first, last):
        left = endpoint - int(lookback_steps) + 1
        # Includes look-back continuity and endpoint -> endpoint+1 target.
        if bad_prefix[endpoint + 1] - bad_prefix[left] == 0:
            endpoints.append(endpoint)
    return np.asarray(endpoints, dtype=int)


def _raw_sequence_features(data, include_time_features: bool):
    lag_power = np.zeros_like(data.powers, dtype=float)
    timestamps = pd.DatetimeIndex(pd.to_datetime(data.timestamps))
    if len(lag_power) > 1:
        contiguous = np.asarray(
            (timestamps[1:] - timestamps[:-1]) == pd.Timedelta(minutes=1),
            dtype=bool,
        )
        lag_power[1:][contiguous] = data.powers[:-1][contiguous]
    arrays = [data.states, data.controls, data.disturbances, lag_power]
    names = [
        *data.state_names,
        *data.encoder.feature_names,
        *data.disturbance_names,
        *data.power_names,
    ]
    if include_time_features:
        arrays.append(_time_features(data.timestamps))
        names.extend(
            ['minute_sin', 'minute_cos', 'weekday_sin', 'weekday_cos']
        )
    return np.hstack(arrays), tuple(names)


class ForecastingRolloutSession:
    def __init__(
        self,
        evaluator: 'ForecastingSequenceEvaluator',
        history: Optional[RolloutHistory],
    ):
        self.evaluator = evaluator
        self.rows = []
        self.previous_power = np.zeros(len(evaluator.power_names), dtype=float)
        self.last_timestamp = None
        if history is not None and len(history.states):
            self.previous_power = history.powers[-1].copy()
            self.last_timestamp = pd.Timestamp(history.timestamps[-1])
            # Starting at one gives every historical row a genuine lagged power.
            for position in range(1, len(history.states)):
                self.rows.append(
                    evaluator.feature_row(
                        history.states[position],
                        history.controls[position],
                        history.disturbances[position],
                        history.powers[position - 1],
                        history.timestamps[position],
                    )
                )
        self.rows = self.rows[-(evaluator.lookback_steps - 1) :]

    def step(self, state, control, disturbance, *, timestamp=None):
        if timestamp is None:
            timestamp = (
                self.last_timestamp + pd.Timedelta(minutes=1)
                if self.last_timestamp is not None
                else pd.Timestamp('1970-01-01')
            )
        timestamp = pd.Timestamp(timestamp)
        row = self.evaluator.feature_row(
            state,
            control,
            disturbance,
            self.previous_power,
            timestamp,
        )
        self.rows.append(row)
        self.rows = self.rows[-self.evaluator.lookback_steps :]
        sequence = np.asarray(self.rows, dtype=float)
        if len(sequence) < self.evaluator.lookback_steps:
            padding = np.repeat(
                sequence[0][None, :],
                self.evaluator.lookback_steps - len(sequence),
                axis=0,
            )
            sequence = np.vstack([padding, sequence])
        prediction = self.evaluator.predict_sequence(sequence)[0]
        state_count = len(self.evaluator.state_names)
        next_state = prediction[:state_count]
        power = np.maximum(prediction[state_count:], 0.0)
        if not (np.isfinite(next_state).all() and np.isfinite(power).all()):
            raise FloatingPointError('Forecasting evaluator produced non-finite output')
        self.previous_power = power.copy()
        self.last_timestamp = timestamp
        return next_state, power


@dataclass
class ForecastingSequenceEvaluator:
    """Use any repository forecasting model as a one-step plant surrogate.

    The input window contains state, one-hot control, disturbance, and lagged
    power.  Targets are the next physical state and current power.  Lagging the
    power channel avoids target leakage while retaining the channel-preserving
    contract expected by DLinear, PatchTST, iTransformer, and related models.
    """

    model: nn.Module
    model_name: str
    model_parameters: Dict[str, object]
    feature_scaler: ArrayScaler
    target_indices: tuple[int, ...]
    lookback_steps: int
    include_time_features: bool
    inference_batch_size: int
    device: torch.device
    state_names: tuple[str, ...]
    control_feature_names: tuple[str, ...]
    disturbance_names: tuple[str, ...]
    power_names: tuple[str, ...]
    input_feature_names: tuple[str, ...]
    training_transition_count: int
    training_summary: Dict[str, object]

    @property
    def required_history_steps(self) -> int:
        return int(self.lookback_steps)

    @classmethod
    def fit(
        cls,
        data,
        start: int,
        end: int,
        *,
        model_name: str,
        model_parameters: Mapping[str, object],
        lookback_steps: int = 120,
        include_time_features: bool = True,
        epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-4,
        patience: int = 5,
        min_delta: float = 1e-5,
        validation_fraction: float = 0.1,
        device: str = 'auto',
        random_state: int = 42,
        eligible_endpoints: Optional[np.ndarray] = None,
    ) -> 'ForecastingSequenceEvaluator':
        if int(lookback_steps) <= 1:
            raise ValueError('Forecasting evaluator lookback_steps must exceed one')
        features, feature_names = _raw_sequence_features(
            data,
            bool(include_time_features),
        )
        endpoints = _valid_sequence_endpoints(
            data,
            start,
            end,
            lookback_steps,
        )
        if eligible_endpoints is not None:
            requested = np.unique(
                np.asarray(eligible_endpoints, dtype=int).reshape(-1)
            )
            eligible_row = np.zeros(len(data.states), dtype=bool)
            eligible_row[requested] = True
            endpoints = np.asarray(
                [
                    endpoint
                    for endpoint in endpoints[np.isin(endpoints, requested)]
                    if eligible_row[
                        endpoint - int(lookback_steps) + 1 : endpoint + 1
                    ].all()
                ],
                dtype=int,
            )
            if not len(endpoints):
                raise ValueError(
                    'No forecasting sequence has both its endpoint and complete '
                    'lookback history inside the explicit evaluator training rows'
                )
        split = _split_count(len(endpoints), validation_fraction)
        train_endpoints = endpoints[:split]
        validation_endpoints = endpoints[split:]
        if eligible_endpoints is None:
            scaler_rows = np.arange(int(train_endpoints[-1]) + 1, dtype=int)
        else:
            scaler_rows = np.unique(
                np.concatenate(
                    [
                        np.arange(
                            int(endpoint) - int(lookback_steps) + 1,
                            int(endpoint) + 1,
                            dtype=int,
                        )
                        for endpoint in train_endpoints
                    ]
                )
            )
        feature_scaler = ArrayScaler.fit(features[scaler_rows])
        scaled_features = feature_scaler.transform(features).astype(np.float32)
        targets = np.hstack([data.states[1:], data.powers[:-1]])
        state_indices = tuple(range(len(data.state_names)))
        power_offset = (
            len(data.state_names)
            + data.encoder.n_features
            + len(data.disturbance_names)
        )
        power_indices = tuple(
            range(power_offset, power_offset + len(data.power_names))
        )
        target_indices = state_indices + power_indices
        target_mean = feature_scaler.mean[list(target_indices)]
        target_scale = feature_scaler.scale[list(target_indices)]
        scaled_targets = (targets - target_mean) / target_scale

        metadata = {
            'seq_len': int(lookback_steps),
            'pred_len': 1,
            'input_numerical_features': list(feature_names),
            'target_features': [*data.state_names, *data.power_names],
            'feature_types': {
                'n_cat': 0,
                'n_num': int(features.shape[1]),
                'n_target': int(targets.shape[1]),
            },
        }
        parameters = deepcopy(dict(model_parameters))
        parameters['METADATA'] = metadata
        from models import create_model

        torch.manual_seed(int(random_state))
        model = create_model(modelname=model_name, params=parameters)
        train_dataset = _SequenceDataset(
            scaled_features,
            scaled_targets,
            train_endpoints,
            lookback_steps,
        )
        validation_dataset = _SequenceDataset(
            scaled_features,
            scaled_targets,
            validation_endpoints,
            lookback_steps,
        )
        resolved_device = _resolve_device(device)
        training_summary = _train_tensor_model(
            model,
            train_dataset,
            validation_dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            device=resolved_device,
            seed=random_state,
            sequence_model=True,
        )
        training_summary['eligible_contiguous_sequences'] = int(len(endpoints))
        return cls(
            model=model,
            model_name=str(model_name),
            model_parameters=parameters,
            feature_scaler=feature_scaler,
            target_indices=target_indices,
            lookback_steps=int(lookback_steps),
            include_time_features=bool(include_time_features),
            inference_batch_size=int(batch_size),
            device=resolved_device,
            state_names=tuple(data.state_names),
            control_feature_names=tuple(data.encoder.feature_names),
            disturbance_names=tuple(data.disturbance_names),
            power_names=tuple(data.power_names),
            input_feature_names=feature_names,
            training_transition_count=int(len(endpoints)),
            training_summary=training_summary,
        )

    def _inverse_targets(self, values: np.ndarray) -> np.ndarray:
        mean = self.feature_scaler.mean[list(self.target_indices)]
        scale = self.feature_scaler.scale[list(self.target_indices)]
        return np.asarray(values, dtype=float) * scale + mean

    def _predict_scaled_sequences(self, sequences: np.ndarray) -> np.ndarray:
        sequences = np.asarray(sequences, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(sequences))
        loader = DataLoader(
            dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
        )
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for (batch_x,) in loader:
                output = self.model(None, batch_x.to(self.device), None)
                outputs.append(output[:, -1, :].cpu().numpy())
        return np.concatenate(outputs)

    def predict_sequence(self, raw_sequence: np.ndarray) -> np.ndarray:
        raw_sequence = np.asarray(raw_sequence, dtype=float)
        if raw_sequence.ndim == 2:
            raw_sequence = raw_sequence[None, :, :]
        if raw_sequence.shape[1:] != (
            self.lookback_steps,
            len(self.input_feature_names),
        ):
            raise ValueError(
                'Forecasting evaluator sequence shape must be '
                f'(batch, {self.lookback_steps}, {len(self.input_feature_names)})'
            )
        scaled = self.feature_scaler.transform(raw_sequence)
        return self._inverse_targets(self._predict_scaled_sequences(scaled))

    def feature_row(self, state, control, disturbance, lag_power, timestamp):
        arrays = [
            np.asarray(state, dtype=float).reshape(-1),
            np.asarray(control, dtype=float).reshape(-1),
            np.asarray(disturbance, dtype=float).reshape(-1),
            np.asarray(lag_power, dtype=float).reshape(-1),
        ]
        if self.include_time_features:
            arrays.append(_time_features([timestamp])[0])
        row = np.concatenate(arrays)
        if len(row) != len(self.input_feature_names):
            raise ValueError('Forecasting evaluator feature width mismatch')
        return row

    def predict_transitions(self, data, start: int, end: int):
        features, feature_names = _raw_sequence_features(
            data,
            self.include_time_features,
        )
        if feature_names != self.input_feature_names:
            raise ValueError('Evaluation data feature names differ from training')
        endpoints = _valid_sequence_endpoints(
            data,
            start,
            end,
            self.lookback_steps,
        )
        if not len(endpoints):
            raise ValueError('No contiguous sequence evaluator samples in this split')
        raw_sequences = np.stack(
            [
                features[
                    endpoint - self.lookback_steps + 1 : endpoint + 1
                ]
                for endpoint in endpoints
            ]
        )
        prediction = self.predict_sequence(raw_sequences)
        state_count = len(self.state_names)
        return TransitionPredictions(
            actual_states=data.states[endpoints + 1],
            predicted_states=prediction[:, :state_count],
            actual_powers=data.powers[endpoints],
            predicted_powers=np.maximum(prediction[:, state_count:], 0.0),
            row_indices=endpoints,
        )

    def predict_control_rows(self, data, row_indices, controls):
        """Predict explicit rows after replacing only the endpoint control.

        Past observations in each look-back window remain measured history;
        the candidate joint action is injected at the prediction endpoint.  This
        is the sequence-model equivalent of ``predict_power(x, action, d)``.
        """
        requested = np.unique(
            np.asarray(row_indices, dtype=int).reshape(-1)
        )
        endpoints = _valid_sequence_endpoints(
            data,
            0,
            len(data.states) - 1,
            self.lookback_steps,
        )
        endpoints = endpoints[np.isin(endpoints, requested)]
        if not len(endpoints):
            raise ValueError(
                'No contiguous sequence evaluator samples in requested rows'
            )
        candidate_controls = np.asarray(controls, dtype=float)
        if candidate_controls.ndim == 1:
            candidate_controls = np.repeat(
                candidate_controls[None, :],
                len(endpoints),
                axis=0,
            )
        elif len(candidate_controls) == len(requested):
            positions = {int(row): position for position, row in enumerate(requested)}
            candidate_controls = np.asarray(
                [candidate_controls[positions[int(row)]] for row in endpoints],
                dtype=float,
            )
        if candidate_controls.shape != (
            len(endpoints),
            data.encoder.n_features,
        ):
            raise ValueError(
                'Candidate controls must match requested sequence rows and '
                'the encoded control width'
            )

        features, feature_names = _raw_sequence_features(
            data,
            self.include_time_features,
        )
        if feature_names != self.input_feature_names:
            raise ValueError('Evaluation data feature names differ from training')
        sequences = []
        control_start = len(data.state_names)
        control_end = control_start + data.encoder.n_features
        for endpoint, control in zip(endpoints, candidate_controls):
            sequence = features[
                endpoint - self.lookback_steps + 1 : endpoint + 1
            ].copy()
            sequence[-1, control_start:control_end] = control
            sequences.append(sequence)
        prediction = self.predict_sequence(np.stack(sequences))
        state_count = len(self.state_names)
        return TransitionPredictions(
            actual_states=data.states[endpoints + 1],
            predicted_states=prediction[:, :state_count],
            actual_powers=data.powers[endpoints],
            predicted_powers=np.maximum(prediction[:, state_count:], 0.0),
            row_indices=endpoints,
        )

    def start_rollout(self, history: Optional[RolloutHistory] = None):
        return ForecastingRolloutSession(self, history)

    def to_metadata(self) -> Dict[str, object]:
        return {
            'role': 'independent history-dependent plant surrogate for offline evaluation',
            'model_family': 'forecasting_sequence',
            'forecasting_model_name': self.model_name,
            'state_target': 'absolute next state',
            'power_target': 'current power with previous-minute power as input',
            'lookback_steps': self.lookback_steps,
            'include_time_features': self.include_time_features,
            'trained_on': (
                'chronological sequences strictly before the configured '
                'training boundary'
            ),
            'training_transition_count': self.training_transition_count,
            'state_names': list(self.state_names),
            'control_feature_names': list(self.control_feature_names),
            'disturbance_names': list(self.disturbance_names),
            'power_names': list(self.power_names),
            'input_feature_names': list(self.input_feature_names),
            'model_parameters': self.model_parameters,
            'training_summary': self.training_summary,
        }
