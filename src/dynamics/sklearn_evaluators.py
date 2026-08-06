"""Interchangeable scikit-learn plant evaluators for MPC backtests."""

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.multioutput import MultiOutputRegressor

from dynamics.evaluator_base import StatelessEvaluatorMixin


def _as_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[:, None] if values.ndim == 1 else values


def _tree_estimator(
    model_family: str,
    *,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: float,
    max_iter: int,
    n_jobs: int,
    random_state: int,
):
    if model_family == 'extra_trees_residual_state':
        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if model_family == 'random_forest_residual_state':
        return RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if model_family == 'hist_gradient_boosting_residual_state':
        base = HistGradientBoostingRegressor(
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        return MultiOutputRegressor(base, n_jobs=n_jobs)
    raise ValueError(
        f'Unsupported sklearn evaluator family: {model_family!r}'
    )


@dataclass
class IndependentSklearnEvaluator(StatelessEvaluatorMixin):
    """Independent residual-state evaluator backed by a sklearn ensemble."""

    state_estimator: object
    power_estimator: object
    model_family: str
    state_names: tuple[str, ...]
    control_feature_names: tuple[str, ...]
    disturbance_names: tuple[str, ...]
    power_names: tuple[str, ...]
    training_transition_count: int
    parameters: Dict[str, object]

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
        model_family: str,
        n_estimators: int = 100,
        min_samples_leaf: int = 5,
        max_features: float = 1.0,
        max_iter: int = 100,
        n_jobs: int = 1,
        random_state: int = 42,
    ) -> 'IndependentSklearnEvaluator':
        arrays = [
            np.asarray(values, dtype=float)
            for values in (states, controls, disturbances, next_states, powers)
        ]
        states, controls, disturbances, next_states, powers = arrays
        row_counts = {len(values) for values in arrays}
        if len(row_counts) != 1 or not row_counts or len(states) == 0:
            raise ValueError('Evaluator arrays must have the same nonzero row count')
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError('Evaluator training arrays must be finite')
        if int(n_estimators) <= 0 or int(max_iter) <= 0:
            raise ValueError('n_estimators and max_iter must be positive')
        if int(min_samples_leaf) <= 0:
            raise ValueError('min_samples_leaf must be positive')

        design = np.hstack([states, controls, disturbances])
        common = {
            'n_estimators': int(n_estimators),
            'min_samples_leaf': int(min_samples_leaf),
            'max_features': float(max_features),
            'max_iter': int(max_iter),
            'n_jobs': int(n_jobs),
        }
        state_estimator = _tree_estimator(
            model_family,
            **common,
            random_state=int(random_state),
        )
        power_estimator = _tree_estimator(
            model_family,
            **common,
            random_state=int(random_state) + 1,
        )
        state_target = next_states - states
        power_target = powers
        if state_target.shape[1] == 1 and not isinstance(
            state_estimator,
            MultiOutputRegressor,
        ):
            state_target = state_target[:, 0]
        if power_target.shape[1] == 1 and not isinstance(
            power_estimator,
            MultiOutputRegressor,
        ):
            power_target = power_target[:, 0]
        state_estimator.fit(design, state_target)
        power_estimator.fit(design, power_target)

        return cls(
            state_estimator=state_estimator,
            power_estimator=power_estimator,
            model_family=model_family,
            state_names=tuple(state_names),
            control_feature_names=tuple(control_feature_names),
            disturbance_names=tuple(disturbance_names),
            power_names=tuple(power_names),
            training_transition_count=int(len(states)),
            parameters={**common, 'random_state': int(random_state)},
        )

    def _design(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        disturbances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        states = np.atleast_2d(np.asarray(states, dtype=float))
        controls = np.atleast_2d(np.asarray(controls, dtype=float))
        disturbances = np.atleast_2d(np.asarray(disturbances, dtype=float))
        if not (len(states) == len(controls) == len(disturbances)):
            raise ValueError('Evaluator inputs must have the same row count')
        expected = (
            len(self.state_names),
            len(self.control_feature_names),
            len(self.disturbance_names),
        )
        actual = (states.shape[1], controls.shape[1], disturbances.shape[1])
        if actual != expected:
            raise ValueError(f'Evaluator input widths must be {expected}, got {actual}')
        return states, np.hstack([states, controls, disturbances])

    def predict_state(self, states, controls, disturbances) -> np.ndarray:
        states, design = self._design(states, controls, disturbances)
        return states + _as_2d(self.state_estimator.predict(design))

    def predict_power(self, states, controls, disturbances) -> np.ndarray:
        _, design = self._design(states, controls, disturbances)
        prediction = _as_2d(self.power_estimator.predict(design))
        return np.maximum(prediction, 0.0)

    def step(self, state, control, disturbance) -> tuple[np.ndarray, np.ndarray]:
        power = self.predict_power(state, control, disturbance)[0]
        next_state = self.predict_state(state, control, disturbance)[0]
        return next_state, power

    def to_metadata(self) -> Dict[str, object]:
        return {
            'role': 'independent nonlinear plant surrogate for offline evaluation',
            'model_family': self.model_family,
            'state_equation': 'x_next = x + ensemble([x, z, d])',
            'power_equation': 'p = max(ensemble([x, z, d]), 0)',
            'trained_on': 'chronological rows strictly before the configured training boundary',
            'training_transition_count': self.training_transition_count,
            'state_names': list(self.state_names),
            'control_feature_names': list(self.control_feature_names),
            'disturbance_names': list(self.disturbance_names),
            'power_names': list(self.power_names),
            'parameters': self.parameters,
        }
