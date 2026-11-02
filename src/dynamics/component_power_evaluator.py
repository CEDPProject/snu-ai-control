"""Component-separated plant evaluator with a two-stage HVAC power model.

The evaluator deliberately keeps one joint next-state model because ventilation
and HVAC actions affect the same indoor state.  It predicts power components
independently, however: every non-HVAC meter has its own regressor and the HVAC
meter is modelled as ``P(active) * E[power | active]``.  This prevents a single
continuous regressor from treating frequent OFF/near-zero HVAC observations as
ordinary low-power values.
"""

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from dynamics.evaluator_base import StatelessEvaluatorMixin


def _as_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[:, None] if values.ndim == 1 else values


def _hvac_power_index(
    power_names: Sequence[str],
    requested_name: str | None = None,
) -> int:
    """Resolve the HVAC component, preferring an explicit meter name."""
    names = tuple(str(name) for name in power_names)
    if requested_name is not None:
        matches = [
            index
            for index, name in enumerate(names)
            if name == str(requested_name)
        ]
        if len(matches) != 1:
            raise ValueError(
                'component_two_stage_hvac.hvac_component_name must exactly '
                f'match one power output; requested={requested_name!r}, '
                f'outputs={list(names)!r}'
            )
        return int(matches[0])
    markers = ('냉난방', 'hvac', 'heating', 'cooling', 'air_condition')
    matches = [
        index
        for index, name in enumerate(names)
        if any(marker in str(name).lower().replace(' ', '') for marker in markers)
    ]
    if len(matches) != 1:
        raise ValueError(
            'component_two_stage_hvac requires exactly one HVAC power meter; '
            f'could not uniquely resolve it from {list(names)!r}; set '
            'component_power.hvac_component_name explicitly'
        )
    return int(matches[0])


def _validate_fit_arrays(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    converted = tuple(np.asarray(values, dtype=float) for values in arrays)
    row_counts = {len(values) for values in converted}
    if len(row_counts) != 1 or not row_counts or len(converted[0]) == 0:
        raise ValueError('Evaluator arrays must have the same nonzero row count')
    if not all(np.isfinite(values).all() for values in converted):
        raise ValueError('Evaluator arrays must contain only finite values')
    return converted


def _active_probability(classifier, design: np.ndarray) -> np.ndarray:
    """Return P(active=1), including the one-class training edge cases."""
    probabilities = np.asarray(classifier.predict_proba(design), dtype=float)
    classes = np.asarray(classifier.classes_)
    active_positions = np.flatnonzero(classes == 1)
    if not len(active_positions):
        return np.zeros(len(design), dtype=float)
    return probabilities[:, int(active_positions[0])]


@dataclass
class ComponentSeparatedTwoStageHVACEvaluator(StatelessEvaluatorMixin):
    """Independent component-power evaluator for the MPC plant backtest.

    State dynamics are learned jointly as a residual model.  The HVAC power
    component uses a binary active/low-power classifier followed by a regressor
    trained only on active rows; all other power components use independent
    regressors.  No controller affine coefficient is reused.
    """

    state_estimator: object
    direct_power_estimators: tuple[object | None, ...]
    hvac_active_classifier: object
    hvac_active_regressor: object
    hvac_power_index: int
    hvac_active_threshold_w: float
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
        n_estimators: int = 100,
        min_samples_leaf: int = 5,
        max_features: float = 1.0,
        n_jobs: int = 1,
        random_state: int = 42,
        hvac_active_threshold_w: float = 0.1,
        hvac_component_name: str | None = None,
    ) -> 'ComponentSeparatedTwoStageHVACEvaluator':
        (
            states,
            controls,
            disturbances,
            next_states,
            powers,
        ) = _validate_fit_arrays(
            states,
            controls,
            disturbances,
            next_states,
            powers,
        )
        if states.ndim != 2 or controls.ndim != 2 or disturbances.ndim != 2:
            raise ValueError('Evaluator inputs must be two-dimensional')
        if next_states.ndim != 2 or powers.ndim != 2:
            raise ValueError('Evaluator targets must be two-dimensional')
        if next_states.shape[1] != len(state_names):
            raise ValueError('next_states width does not match state_names')
        if powers.shape[1] != len(power_names):
            raise ValueError('powers width does not match power_names')
        if int(n_estimators) <= 0:
            raise ValueError('n_estimators must be positive')
        if int(min_samples_leaf) <= 0:
            raise ValueError('min_samples_leaf must be positive')
        if float(hvac_active_threshold_w) < 0.0:
            raise ValueError('hvac_active_threshold_w must be nonnegative')

        hvac_index = _hvac_power_index(power_names, hvac_component_name)
        design = np.hstack([states, controls, disturbances])
        common = {
            'n_estimators': int(n_estimators),
            'min_samples_leaf': int(min_samples_leaf),
            'max_features': float(max_features),
            'n_jobs': int(n_jobs),
        }
        state_estimator = ExtraTreesRegressor(
            **common,
            random_state=int(random_state),
        )
        state_estimator.fit(design, next_states - states)

        direct_estimators: list[object | None] = []
        for index, _ in enumerate(power_names):
            if index == hvac_index:
                direct_estimators.append(None)
                continue
            estimator = ExtraTreesRegressor(
                **common,
                random_state=int(random_state) + 10 + index,
            )
            estimator.fit(design, powers[:, index])
            direct_estimators.append(estimator)

        active = powers[:, hvac_index] > float(hvac_active_threshold_w)
        active_classifier = ExtraTreesClassifier(
            **common,
            random_state=int(random_state) + 100,
        )
        active_classifier.fit(design, active.astype(int))
        if np.any(active):
            active_regressor = ExtraTreesRegressor(
                **common,
                random_state=int(random_state) + 101,
            )
            active_regressor.fit(design[active], powers[active, hvac_index])
            active_regressor_kind = 'extra_trees_active_conditional_regressor'
        else:
            # A valid all-OFF training segment should produce a deterministic,
            # auditable zero expectation instead of an estimator-fit failure.
            active_regressor = DummyRegressor(strategy='constant', constant=0.0)
            active_regressor.fit(design, np.zeros(len(design), dtype=float))
            active_regressor_kind = 'constant_zero_no_active_training_rows'

        return cls(
            state_estimator=state_estimator,
            direct_power_estimators=tuple(direct_estimators),
            hvac_active_classifier=active_classifier,
            hvac_active_regressor=active_regressor,
            hvac_power_index=hvac_index,
            hvac_active_threshold_w=float(hvac_active_threshold_w),
            state_names=tuple(state_names),
            control_feature_names=tuple(control_feature_names),
            disturbance_names=tuple(disturbance_names),
            power_names=tuple(power_names),
            training_transition_count=int(len(states)),
            parameters={
                **common,
                'random_state': int(random_state),
                'hvac_active_threshold_w': float(hvac_active_threshold_w),
                'hvac_component_name': str(power_names[hvac_index]),
                'hvac_active_training_rows': int(active.sum()),
                'hvac_low_or_off_training_rows': int((~active).sum()),
                'hvac_active_regressor': active_regressor_kind,
            },
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
        if not (
            np.isfinite(states).all()
            and np.isfinite(controls).all()
            and np.isfinite(disturbances).all()
        ):
            raise ValueError('Evaluator inputs must be finite')
        return states, np.hstack([states, controls, disturbances])

    def predict_state(self, states, controls, disturbances) -> np.ndarray:
        states, design = self._design(states, controls, disturbances)
        return states + _as_2d(self.state_estimator.predict(design))

    def predict_power_breakdown(self, states, controls, disturbances) -> Dict[str, np.ndarray]:
        """Return separate component predictions and HVAC mixture terms."""
        _, design = self._design(states, controls, disturbances)
        components = np.empty((len(design), len(self.power_names)), dtype=float)
        for index, estimator in enumerate(self.direct_power_estimators):
            if index == self.hvac_power_index:
                continue
            if estimator is None:  # Defensive contract check for corrupted state.
                raise RuntimeError('Missing direct estimator for a non-HVAC meter')
            components[:, index] = np.asarray(estimator.predict(design), dtype=float)

        active_probability = _active_probability(self.hvac_active_classifier, design)
        active_conditional_power = np.asarray(
            self.hvac_active_regressor.predict(design),
            dtype=float,
        ).reshape(-1)
        components[:, self.hvac_power_index] = (
            active_probability * active_conditional_power
        )
        components = np.maximum(components, 0.0)
        return {
            'component_power_w': components,
            'hvac_active_probability': active_probability,
            'hvac_active_conditional_power_w': np.maximum(
                active_conditional_power,
                0.0,
            ),
        }

    def predict_power(self, states, controls, disturbances) -> np.ndarray:
        return self.predict_power_breakdown(
            states,
            controls,
            disturbances,
        )['component_power_w']

    def step(self, state, control, disturbance) -> tuple[np.ndarray, np.ndarray]:
        power = self.predict_power(state, control, disturbance)[0]
        next_state = self.predict_state(state, control, disturbance)[0]
        return next_state, power

    def to_metadata(self) -> Dict[str, object]:
        direct_names = [
            name
            for index, name in enumerate(self.power_names)
            if index != self.hvac_power_index
        ]
        return {
            'role': 'independent nonlinear plant surrogate for offline evaluation',
            'model_family': 'component_two_stage_hvac',
            'state_equation': (
                'x_next = x + joint ExtraTrees([x, z, d]) residual prediction'
            ),
            'power_equation': (
                'non-HVAC components = independent ExtraTrees([x, z, d]); '
                'HVAC = P(active | x, z, d) * E[power | active, x, z, d]'
            ),
            'trained_on': 'chronological rows strictly before the configured training boundary',
            'training_transition_count': self.training_transition_count,
            'state_names': list(self.state_names),
            'control_feature_names': list(self.control_feature_names),
            'disturbance_names': list(self.disturbance_names),
            'power_names': list(self.power_names),
            'component_modeling': {
                'direct_regression_power_names': direct_names,
                'hvac_power_name': self.power_names[self.hvac_power_index],
                'hvac_active_threshold_w': self.hvac_active_threshold_w,
                'hvac_mixture': 'active_probability_times_active_conditional_power',
            },
            'parameters': self.parameters,
        }
