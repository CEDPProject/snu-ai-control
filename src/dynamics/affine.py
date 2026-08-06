"""Affine models that keep mixed-integer MPC linear and auditable."""

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def _fit_physical_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on standardized inputs and return coefficients in physical units."""
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    estimator = Ridge(alpha=float(alpha))
    estimator.fit(standardized, targets)

    coefficients = np.atleast_2d(estimator.coef_) / scaler.scale_[None, :]
    intercept = np.atleast_1d(estimator.intercept_) - coefficients @ scaler.mean_
    return coefficients, intercept


@dataclass(frozen=True)
class AffineStateModel:
    """x(t+1) = A x(t) + B z(t) + E d(t) + c."""

    A: np.ndarray
    B: np.ndarray
    E: np.ndarray
    c: np.ndarray
    state_names: tuple[str, ...]
    control_feature_names: tuple[str, ...]
    disturbance_names: tuple[str, ...]

    def predict(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        disturbances: np.ndarray,
    ) -> np.ndarray:
        states = np.atleast_2d(states)
        controls = np.atleast_2d(controls)
        disturbances = np.atleast_2d(disturbances)
        return (
            states @ self.A.T
            + controls @ self.B.T
            + disturbances @ self.E.T
            + self.c
        )

    def to_dict(self) -> Dict:
        return {
            'equation': 'x_next = A @ x + B @ z + E @ d + c',
            'A': self.A.tolist(),
            'B': self.B.tolist(),
            'E': self.E.tolist(),
            'c': self.c.tolist(),
            'state_names': list(self.state_names),
            'control_feature_names': list(self.control_feature_names),
            'disturbance_names': list(self.disturbance_names),
        }


@dataclass(frozen=True)
class AffinePowerModel:
    """p(t) = F x(t) + G z(t) + H d(t) + b, one row per meter."""

    F: np.ndarray
    G: np.ndarray
    H: np.ndarray
    b: np.ndarray
    power_names: tuple[str, ...]
    state_names: tuple[str, ...]
    control_feature_names: tuple[str, ...]
    disturbance_names: tuple[str, ...]

    def predict_components(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        disturbances: np.ndarray,
        clip_nonnegative: bool = True,
    ) -> np.ndarray:
        states = np.atleast_2d(states)
        controls = np.atleast_2d(controls)
        disturbances = np.atleast_2d(disturbances)
        prediction = (
            states @ self.F.T
            + controls @ self.G.T
            + disturbances @ self.H.T
            + self.b
        )
        return np.maximum(prediction, 0.0) if clip_nonnegative else prediction

    def predict_total(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        disturbances: np.ndarray,
        clip_nonnegative: bool = True,
    ) -> np.ndarray:
        return self.predict_components(
            states, controls, disturbances, clip_nonnegative
        ).sum(axis=1)

    @property
    def total_coefficients(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        return (
            self.F.sum(axis=0),
            self.G.sum(axis=0),
            self.H.sum(axis=0),
            float(self.b.sum()),
        )

    def to_dict(self) -> Dict:
        return {
            'equation': 'power_components = F @ x + G @ z + H @ d + b',
            'F': self.F.tolist(),
            'G': self.G.tolist(),
            'H': self.H.tolist(),
            'b': self.b.tolist(),
            'power_names': list(self.power_names),
            'state_names': list(self.state_names),
            'control_feature_names': list(self.control_feature_names),
            'disturbance_names': list(self.disturbance_names),
        }


def fit_affine_models(
    x: np.ndarray,
    z: np.ndarray,
    d: np.ndarray,
    x_next: np.ndarray,
    power: np.ndarray,
    state_names: Sequence[str],
    control_feature_names: Sequence[str],
    disturbance_names: Sequence[str],
    power_names: Sequence[str],
    state_alpha: float = 1.0,
    power_alpha: float = 1.0,
) -> tuple[AffineStateModel, AffinePowerModel]:
    if not (len(x) == len(z) == len(d) == len(x_next) == len(power)):
        raise ValueError('Affine training arrays must contain the same number of rows')
    if len(x) == 0:
        raise ValueError('No one-minute transitions are available for affine training')

    design = np.hstack([x, z, d])
    state_coef, state_intercept = _fit_physical_ridge(
        design, x_next, state_alpha
    )
    power_coef, power_intercept = _fit_physical_ridge(
        design, power, power_alpha
    )

    nx = x.shape[1]
    nz = z.shape[1]
    state_model = AffineStateModel(
        A=state_coef[:, :nx],
        B=state_coef[:, nx : nx + nz],
        E=state_coef[:, nx + nz :],
        c=state_intercept,
        state_names=tuple(state_names),
        control_feature_names=tuple(control_feature_names),
        disturbance_names=tuple(disturbance_names),
    )
    power_model = AffinePowerModel(
        F=power_coef[:, :nx],
        G=power_coef[:, nx : nx + nz],
        H=power_coef[:, nx + nz :],
        b=power_intercept,
        power_names=tuple(power_names),
        state_names=tuple(state_names),
        control_feature_names=tuple(control_feature_names),
        disturbance_names=tuple(disturbance_names),
    )
    return state_model, power_model


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, object]:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return {
        'mae_per_output': np.atleast_1d(
            mean_absolute_error(actual, predicted, multioutput='raw_values')
        ).tolist(),
        'r2_per_output': np.atleast_1d(
            r2_score(actual, predicted, multioutput='raw_values')
        ).tolist(),
        'mae_mean': float(mean_absolute_error(actual, predicted)),
        'r2_variance_weighted': float(
            r2_score(actual, predicted, multioutput='variance_weighted')
        ),
    }


def power_mape_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    minimum_actual_power_w: float,
) -> Dict[str, object]:
    """Compute power MAPE while making near-zero exclusions auditable.

    A percentage error is defined only where ``abs(actual)`` is strictly
    greater than ``minimum_actual_power_w``. Per-output and overall values
    are both reported because each power meter can have a different number of
    valid (device-on) samples.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    threshold = float(minimum_actual_power_w)

    if actual.shape != predicted.shape:
        raise ValueError(
            'Power MAPE actual and predicted arrays must have the same shape'
        )
    if actual.ndim == 1:
        actual = actual[:, None]
        predicted = predicted[:, None]
    if actual.ndim != 2:
        raise ValueError('Power MAPE arrays must be one- or two-dimensional')
    if threshold < 0.0 or not np.isfinite(threshold):
        raise ValueError('minimum_actual_power_w must be finite and nonnegative')
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError('Power MAPE arrays must contain only finite values')

    valid_mask = np.abs(actual) > threshold
    percentage_errors = np.full(actual.shape, np.nan, dtype=float)
    percentage_errors[valid_mask] = (
        np.abs(predicted[valid_mask] - actual[valid_mask])
        / np.abs(actual[valid_mask])
        * 100.0
    )

    valid_counts = valid_mask.sum(axis=0)
    sample_count = int(actual.shape[0])
    per_output = [
        float(percentage_errors[:, index][valid_mask[:, index]].mean())
        if valid_counts[index] > 0
        else None
        for index in range(actual.shape[1])
    ]
    defined_per_output = [value for value in per_output if value is not None]
    all_valid_errors = percentage_errors[valid_mask]
    overall = float(all_valid_errors.mean()) if all_valid_errors.size else None
    coverage = (
        (valid_counts.astype(float) / sample_count * 100.0).tolist()
        if sample_count
        else [0.0] * actual.shape[1]
    )

    return {
        'mape_definition': (
            'mean(abs(predicted - actual) / abs(actual)) * 100, '
            'using only abs(actual) > minimum_actual_power_w'
        ),
        'mape_unit': 'percent',
        'minimum_actual_power_w': threshold,
        'mape_percent': overall,
        'mape_percent_per_output': per_output,
        'mape_percent_macro_mean': (
            float(np.mean(defined_per_output)) if defined_per_output else None
        ),
        'mape_valid_samples_per_output': valid_counts.astype(int).tolist(),
        'mape_total_samples_per_output': sample_count,
        'mape_coverage_percent_per_output': coverage,
    }
