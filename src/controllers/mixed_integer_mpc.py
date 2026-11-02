"""Exact categorical MPC formulated as a mixed-integer linear program."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import Bounds, LinearConstraint

from controllers.guarded_milp import guarded_milp
from scipy.sparse import coo_matrix

from data_provider.control_data import ControlEncoder
from dynamics.affine import AffinePowerModel, AffineStateModel


@dataclass(frozen=True)
class MPCSolution:
    success: bool
    status: int
    message: str
    objective: float
    actions: tuple[Dict[str, str], ...]
    control_vectors: np.ndarray
    state_trajectory: np.ndarray
    power_components_w: np.ndarray
    lower_slack: np.ndarray
    upper_slack: np.ndarray

    @property
    def total_power_w(self) -> np.ndarray:
        return self.power_components_w.sum(axis=1)

    @property
    def energy_wh(self) -> float:
        return float(self.total_power_w.sum() / 60.0)


class _VariableLayout:
    def __init__(self):
        self.size = 0

    def reserve(self, shape):
        count = int(np.prod(shape))
        indices = np.arange(self.size, self.size + count, dtype=int).reshape(shape)
        self.size += count
        return indices


class _ConstraintBuilder:
    def __init__(self):
        self.rows = []
        self.cols = []
        self.values = []
        self.lower = []
        self.upper = []

    def add(self, coefficients: Mapping[int, float], lower=-np.inf, upper=np.inf):
        row = len(self.lower)
        for column, value in coefficients.items():
            if value != 0.0:
                self.rows.append(row)
                self.cols.append(int(column))
                self.values.append(float(value))
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def build(self, n_variables: int) -> LinearConstraint:
        matrix = coo_matrix(
            (self.values, (self.rows, self.cols)),
            shape=(len(self.lower), n_variables),
        ).tocsr()
        return LinearConstraint(
            matrix,
            np.asarray(self.lower, dtype=float),
            np.asarray(self.upper, dtype=float),
        )


class MixedIntegerMPC:
    """MILP controller with exact one-hot controls and soft state constraints."""

    def __init__(
        self,
        state_model: AffineStateModel,
        power_model: AffinePowerModel,
        encoder: ControlEncoder,
        state_constraints: Sequence[Mapping],
        logical_rules: Sequence[Mapping],
        prediction_horizon: int = 60,
        control_horizon: int = 10,
        switch_penalty: float = 0.05,
        maximum_changed_control_variables_per_step: int | None = None,
        action_history_enabled: bool = False,
        action_history_lookback_steps: int = 0,
        action_history_penalty: float = 0.0,
        action_history_recency_weights: Sequence[float] | None = None,
        action_history_group_definitions: Sequence[Mapping] | None = None,
        action_history_rolling_change_budget: int | None = None,
        action_history_per_group_rolling_change_budgets: Mapping[str, int] | None = None,
        action_history_budget_slack_penalty: float = 1000.0,
        time_limit_seconds: float = 30.0,
        mip_relative_gap: float = 0.01,
        forbidden_control_vectors: np.ndarray | None = None,
    ):
        self.state_model = state_model
        self.power_model = power_model
        self.encoder = encoder
        self.prediction_horizon = int(prediction_horizon)
        self.control_horizon = int(control_horizon)
        self.switch_penalty = float(switch_penalty)
        self.maximum_changed_control_variables_per_step = (
            None
            if maximum_changed_control_variables_per_step is None
            else int(maximum_changed_control_variables_per_step)
        )
        self.action_history_enabled = bool(action_history_enabled)
        self.action_history_lookback_steps = int(action_history_lookback_steps)
        self.action_history_penalty = float(action_history_penalty)
        self.action_history_recency_weights = (
            None
            if action_history_recency_weights is None
            else tuple(float(value) for value in action_history_recency_weights)
        )
        self.action_history_group_definitions = action_history_group_definitions
        self.action_history_rolling_change_budget = (
            None
            if action_history_rolling_change_budget is None
            else int(action_history_rolling_change_budget)
        )
        self.action_history_per_group_rolling_change_budgets = (
            action_history_per_group_rolling_change_budgets
        )
        self.action_history_budget_slack_penalty = float(
            action_history_budget_slack_penalty
        )
        self.time_limit_seconds = float(time_limit_seconds)
        self.mip_relative_gap = float(mip_relative_gap)
        self.state_constraints = tuple(
            OmegaConf.to_container(item, resolve=True)
            if not isinstance(item, dict)
            else dict(item)
            for item in state_constraints
        )
        self.logical_rules = tuple(
            OmegaConf.to_container(item, resolve=True)
            if not isinstance(item, dict)
            else dict(item)
            for item in logical_rules
        )
        if forbidden_control_vectors is None:
            forbidden_control_vectors = np.empty((0, encoder.n_features))
        forbidden_control_vectors = np.asarray(
            forbidden_control_vectors,
            dtype=float,
        )
        if forbidden_control_vectors.size == 0:
            forbidden_control_vectors = np.empty((0, encoder.n_features))
        self.forbidden_control_vectors = forbidden_control_vectors
        self.action_history_groups = self._resolve_action_history_groups()
        self.action_history_per_group_rolling_change_budgets = (
            self._resolve_per_group_history_budgets()
        )
        self._validate_contract()

    def _resolve_action_history_groups(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Resolve auditable action-distance groups from configured variables.

        A raw categorical OFF transition can change several one-hot control
        variables because canonical OFF sub-settings are required by logical
        rules.  History smoothing therefore supports equipment-level groups
        (for example, the three ventilation variables as one group) in
        addition to one-control-variable-per-group semantics.
        """
        raw_definitions = self.action_history_group_definitions
        if raw_definitions is None:
            return tuple(
                (name, tuple(features))
                for name, features in self.encoder.groups.items()
            )
        if OmegaConf.is_config(raw_definitions):
            raw_definitions = OmegaConf.to_container(
                raw_definitions,
                resolve=True,
            )
        if not isinstance(raw_definitions, Sequence) or isinstance(
            raw_definitions,
            (str, bytes),
        ):
            raise TypeError('action_history_group_definitions must be a sequence')

        groups = []
        assigned_variables = set()
        known_variables = set(self.encoder.groups)
        for raw_definition in raw_definitions:
            definition = (
                OmegaConf.to_container(raw_definition, resolve=True)
                if OmegaConf.is_config(raw_definition)
                else dict(raw_definition)
            )
            name = str(definition.get('name', '')).strip()
            variables = definition.get('variables')
            if not name:
                raise ValueError('Every action history group requires a name')
            if not isinstance(variables, Sequence) or isinstance(
                variables,
                (str, bytes),
            ) or not variables:
                raise ValueError(
                    f'Action history group {name!r} requires nonempty variables'
                )
            variable_names = tuple(str(variable) for variable in variables)
            unknown = sorted(set(variable_names) - known_variables)
            if unknown:
                raise ValueError(
                    f'Action history group {name!r} has unknown variables: {unknown}'
                )
            duplicated = sorted(set(variable_names) & assigned_variables)
            if duplicated:
                raise ValueError(
                    'Action history groups must form a non-overlapping partition; '
                    f'{name!r} repeats {duplicated}'
                )
            assigned_variables.update(variable_names)
            features = tuple(
                feature
                for variable in variable_names
                for feature in self.encoder.groups[variable]
            )
            groups.append((name, features))
        if assigned_variables != known_variables:
            missing = sorted(known_variables - assigned_variables)
            raise ValueError(
                'Action history groups must cover every control variable; '
                f'missing={missing}'
            )
        names = [name for name, _ in groups]
        if len(names) != len(set(names)):
            raise ValueError('Action history group names must be unique')
        return tuple(groups)

    def _resolve_per_group_history_budgets(self) -> dict[str, int] | None:
        """Validate optional independent budgets for configured history groups.

        ``rolling_change_budget`` remains the aggregate budget across all
        history groups.  This optional mapping adds separate upper bounds for
        named groups, so a frequently changing variable cannot consume the
        entire aggregate allowance.  A mapping may intentionally cover only a
        subset of groups; unlisted groups are governed by the aggregate budget
        and the soft history penalty only.
        """
        raw_budgets = self.action_history_per_group_rolling_change_budgets
        if raw_budgets is None:
            return None
        if OmegaConf.is_config(raw_budgets):
            raw_budgets = OmegaConf.to_container(raw_budgets, resolve=True)
        if not isinstance(raw_budgets, Mapping):
            raise TypeError(
                'action_history_per_group_rolling_change_budgets must be a mapping'
            )
        if not raw_budgets:
            raise ValueError(
                'action_history_per_group_rolling_change_budgets must not be empty'
            )

        known_names = {name for name, _ in self.action_history_groups}
        unknown = sorted(set(raw_budgets) - known_names)
        if unknown:
            raise ValueError(
                'Per-group history budgets reference unknown groups: '
                f'{unknown}; known={sorted(known_names)}'
            )

        budgets: dict[str, int] = {}
        for name, _ in self.action_history_groups:
            if name not in raw_budgets:
                continue
            value = raw_budgets[name]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(
                    f'Per-group history budget for {name!r} must be a nonnegative integer'
                )
            if int(value) < 0:
                raise ValueError(
                    f'Per-group history budget for {name!r} must be nonnegative'
                )
            budgets[name] = int(value)
        return budgets

    def _normalize_action_history(
        self,
        action_history: np.ndarray | None,
        previous_control: np.ndarray,
    ) -> np.ndarray:
        """Return chronological history ending with the latest applied action."""
        if not self.action_history_enabled:
            return np.empty((0, self.encoder.n_features), dtype=float)
        if action_history is None:
            history = np.asarray(previous_control, dtype=float)[None, :]
        else:
            history = np.asarray(action_history, dtype=float)
            if history.size == 0:
                history = np.asarray(previous_control, dtype=float)[None, :]
            if history.ndim != 2 or history.shape[1] != self.encoder.n_features:
                raise ValueError(
                    'action_history must have shape '
                    f'(steps, {self.encoder.n_features})'
                )
        history = history[-self.action_history_lookback_steps :].copy()
        canonical = self._canonicalize_categorical_controls(history)
        if canonical is None:
            raise ValueError('action_history vectors must be binary one-hot controls')
        return canonical

    def _canonicalize_categorical_controls(
        self,
        controls: np.ndarray,
        *,
        tolerance: float = 1e-6,
    ) -> np.ndarray | None:
        """Return exact binary one-hot controls, or reject a relaxed MILP point.

        HiGHS may return ``result.x`` even when it stopped before finding an
        integer incumbent.  Such a vector can be a fractional LP relaxation;
        it is not a realizable categorical action and must never be applied or
        appended to action history.  Conversely, a genuine MILP incumbent can
        contain harmless floating-point residue such as ``0.999999999``.

        This boundary performs no argmax projection: values are accepted only
        when every element is already within ``tolerance`` of 0 or 1 and every
        categorical group has exactly one selected value.  Accepted vectors
        are then made *exactly* binary for histories, equality cuts, CSV audit
        trails, and downstream evaluator calls.
        """
        values = np.asarray(controls, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.encoder.n_features:
            raise ValueError(
                'categorical controls must have shape '
                f'(steps, {self.encoder.n_features})'
            )
        if not np.isfinite(values).all():
            return None
        rounded = np.rint(values)
        if not np.allclose(values, rounded, rtol=0.0, atol=tolerance):
            return None
        if not np.isin(rounded, [0.0, 1.0]).all():
            return None
        for features in self.encoder.groups.values():
            if not np.all(rounded[:, list(features)].sum(axis=1) == 1.0):
                return None
        return rounded.astype(float, copy=False)

    def _action_history_group_distance(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> int:
        """Count configured action groups whose categorical setting differs."""
        return int(self._action_history_group_changes(left, right).sum())

    def _action_history_group_changes(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        """Return one change indicator per configured history group."""
        left = np.rint(np.asarray(left, dtype=float)).astype(int)
        right = np.rint(np.asarray(right, dtype=float)).astype(int)
        return np.asarray([
            int(not np.array_equal(left[list(features)], right[list(features)]))
            for _, features in self.action_history_groups
        ], dtype=int)

    def _historical_change_count(self, history: np.ndarray) -> int:
        return int(self._historical_change_counts(history).sum())

    def _historical_change_counts(self, history: np.ndarray) -> np.ndarray:
        """Count prior transitions separately for every history group."""
        counts = np.zeros(len(self.action_history_groups), dtype=int)
        for position in range(1, len(history)):
            counts += self._action_history_group_changes(
                history[position - 1], history[position]
            )
        return counts

    def action_history_diagnostics(
        self,
        applied_control: np.ndarray,
        action_history: np.ndarray | None,
        previous_control: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        """Describe the first-action history regularization used in one solve."""
        if not self.action_history_enabled:
            return {'enabled': False}
        if previous_control is None:
            previous_control = self.encoder.encode_action()
        history = self._normalize_action_history(action_history, previous_control)
        distances = [
            self._action_history_group_distance(applied_control, history[-lag])
            for lag in range(1, len(history) + 1)
        ]
        weighted_distance = float(sum(
            self.action_history_recency_weights[position] * distance
            for position, distance in enumerate(distances)
        ))
        historical_change_counts = self._historical_change_counts(history)
        historical_change_count = int(historical_change_counts.sum())
        current_changes = (
            self._action_history_group_changes(applied_control, history[-1])
            if len(history)
            else np.zeros(len(self.action_history_groups), dtype=int)
        )
        current_change_count = int(current_changes.sum())
        rolling_change_count = historical_change_count + current_change_count
        budget = self.action_history_rolling_change_budget
        group_names = [name for name, _ in self.action_history_groups]
        per_group_budgets = self.action_history_per_group_rolling_change_budgets
        rolling_by_group = historical_change_counts + current_changes
        return {
            'enabled': True,
            'lookback_steps_configured': self.action_history_lookback_steps,
            'lookback_steps_used': int(len(history)),
            'group_names': group_names,
            'distance_by_lag': {
                f'lag_{position + 1}': int(distance)
                for position, distance in enumerate(distances)
            },
            'weighted_distance': weighted_distance,
            'historical_change_count': int(historical_change_count),
            'current_change_count': int(current_change_count),
            'rolling_change_count': int(rolling_change_count),
            'rolling_change_budget': budget,
            'rolling_budget_overflow': (
                None
                if budget is None
                else float(max(0, rolling_change_count - budget))
            ),
            'historical_change_count_by_group': {
                name: int(historical_change_counts[position])
                for position, name in enumerate(group_names)
            },
            'current_change_by_group': {
                name: int(current_changes[position])
                for position, name in enumerate(group_names)
            },
            'rolling_change_count_by_group': {
                name: int(rolling_by_group[position])
                for position, name in enumerate(group_names)
            },
            'per_group_rolling_change_budgets': per_group_budgets,
            'per_group_rolling_budget_overflow': (
                None
                if per_group_budgets is None
                else {
                    name: float(max(0, rolling_by_group[position] - budget))
                    for position, name in enumerate(group_names)
                    if name in per_group_budgets
                    for budget in (per_group_budgets[name],)
                }
            ),
        }

    def _validate_contract(self):
        if self.prediction_horizon <= 0:
            raise ValueError('prediction_horizon must be positive')
        if not 0 < self.control_horizon <= self.prediction_horizon:
            raise ValueError('control_horizon must be in [1, prediction_horizon]')
        if (
            self.maximum_changed_control_variables_per_step is not None
            and not 0
            <= self.maximum_changed_control_variables_per_step
            <= len(self.encoder.variables)
        ):
            raise ValueError(
                'maximum_changed_control_variables_per_step must be null or '
                'an integer in [0, number of control variables]'
            )
        if self.action_history_enabled:
            if self.action_history_lookback_steps <= 0:
                raise ValueError(
                    'action_history_lookback_steps must be positive when '
                    'action history is enabled'
                )
            if self.action_history_penalty < 0.0:
                raise ValueError('action_history_penalty must be non-negative')
            if self.action_history_recency_weights is None:
                self.action_history_recency_weights = tuple(
                    1.0
                    for _ in range(self.action_history_lookback_steps)
                )
            if len(self.action_history_recency_weights) != (
                self.action_history_lookback_steps
            ):
                raise ValueError(
                    'action_history_recency_weights must contain exactly '
                    'action_history_lookback_steps values'
                )
            if any(value < 0.0 for value in self.action_history_recency_weights):
                raise ValueError('action_history_recency_weights must be non-negative')
            if (
                self.action_history_penalty == 0.0
                and self.action_history_rolling_change_budget is None
                and self.action_history_per_group_rolling_change_budgets is None
            ):
                raise ValueError(
                    'Enabled action history requires a positive penalty or a '
                    'rolling change budget'
                )
            if (
                self.action_history_rolling_change_budget is not None
                and self.action_history_rolling_change_budget < 0
            ):
                raise ValueError(
                    'action_history_rolling_change_budget must be null or non-negative'
                )
            if (
                (
                    self.action_history_rolling_change_budget is not None
                    or self.action_history_per_group_rolling_change_budgets is not None
                )
                and self.action_history_budget_slack_penalty <= 0.0
            ):
                raise ValueError(
                    'action_history_budget_slack_penalty must be positive when '
                    'a rolling change budget is configured'
                )
        if tuple(item['name'] for item in self.state_constraints) != (
            self.state_model.state_names
        ):
            raise ValueError('State constraint order must match the affine state model')
        if self.state_model.control_feature_names != self.encoder.feature_names:
            raise ValueError('Control encoder does not match the affine state model')
        if self.power_model.control_feature_names != self.encoder.feature_names:
            raise ValueError('Control encoder does not match the affine power model')
        expected_shape = (len(self.forbidden_control_vectors), self.encoder.n_features)
        if self.forbidden_control_vectors.shape != expected_shape:
            raise ValueError(
                'forbidden_control_vectors must be a two-dimensional control matrix'
            )
        for vector in self.forbidden_control_vectors:
            rounded = np.rint(vector)
            if not np.allclose(vector, rounded) or not np.isin(rounded, [0, 1]).all():
                raise ValueError('Forbidden control vectors must be binary')
            for group in self.encoder.groups.values():
                if int(rounded[list(group)].sum()) != 1:
                    raise ValueError(
                        'Each forbidden control vector must select one category '
                        'per control variable'
                    )

        # Validate rule references before a solve is attempted.
        for rule in self.logical_rules:
            when = rule['when']
            self.encoder.index(when['variable'], when['value'])
            for variable, value in rule.get('require', {}).items():
                self.encoder.index(variable, value)
            for variable, values in rule.get('forbid', {}).items():
                for value in values:
                    self.encoder.index(variable, value)

    def _validate_dynamic_forbidden_vectors(
        self,
        values: np.ndarray | None,
    ) -> np.ndarray:
        """Validate context-local first-action no-good cuts.

        Static cuts exclude globally invalid or unsupported actions.  These
        additional cuts are supplied on each receding-horizon solve after a
        train-only local-support and evaluator screen.  They intentionally
        constrain only the first action: later actions are never applied
        without a new measured state and a new screen one minute later.
        """
        if values is None:
            return np.empty((0, self.encoder.n_features), dtype=float)
        vectors = np.asarray(values, dtype=float)
        if vectors.size == 0:
            return np.empty((0, self.encoder.n_features), dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != self.encoder.n_features:
            raise ValueError(
                'first_step_forbidden_control_vectors must be a two-dimensional '
                'control matrix'
            )
        for vector in vectors:
            rounded = np.rint(vector)
            if not np.allclose(vector, rounded) or not np.isin(rounded, [0, 1]).all():
                raise ValueError('First-step forbidden control vectors must be binary')
            for group in self.encoder.groups.values():
                if int(rounded[list(group)].sum()) != 1:
                    raise ValueError(
                        'Each first-step forbidden control vector must select one '
                        'category per control variable'
                    )
        return vectors

    def solve(
        self,
        initial_state: np.ndarray,
        disturbances: np.ndarray,
        previous_control: np.ndarray | None = None,
        action_history: np.ndarray | None = None,
        first_step_forbidden_control_vectors: np.ndarray | None = None,
    ) -> MPCSolution:
        n = self.prediction_horizon
        m = self.control_horizon
        nx = len(self.state_model.state_names)
        nz = self.encoder.n_features
        npower = len(self.power_model.power_names)

        initial_state = np.asarray(initial_state, dtype=float).reshape(nx)
        disturbances = np.asarray(disturbances, dtype=float)
        expected_shape = (n, len(self.state_model.disturbance_names))
        if disturbances.shape != expected_shape:
            raise ValueError(
                f'disturbances must have shape {expected_shape}, got {disturbances.shape}'
            )
        if previous_control is None:
            previous_control = self.encoder.encode_action()
        previous_control = np.asarray(previous_control, dtype=float).reshape(nz)
        normalized_action_history = self._normalize_action_history(
            action_history,
            previous_control,
        )
        dynamic_forbidden = self._validate_dynamic_forbidden_vectors(
            first_step_forbidden_control_vectors,
        )

        layout = _VariableLayout()
        x_index = layout.reserve((n + 1, nx))
        z_index = layout.reserve((m, nz))
        power_index = layout.reserve((n, npower))
        lower_slack_index = layout.reserve((n, nx))
        upper_slack_index = layout.reserve((n, nx))
        delta_index = layout.reserve((m, nz))
        changed_group_index = layout.reserve((m, len(self.encoder.variables)))
        history_changed_group_index = layout.reserve(
            (len(normalized_action_history), len(self.action_history_groups))
        )
        history_budget_slack_index = (
            layout.reserve((1,))
            if (
                self.action_history_enabled
                and self.action_history_rolling_change_budget is not None
            )
            else None
        )
        per_group_history_budget_slack_index = (
            layout.reserve((len(self.action_history_groups),))
            if (
                self.action_history_enabled
                and self.action_history_per_group_rolling_change_budgets is not None
            )
            else None
        )

        objective = np.zeros(layout.size, dtype=float)
        # A one-minute sample of power in W contributes W / 60 Wh.
        objective[power_index.ravel()] = 1.0 / 60.0
        objective[delta_index.ravel()] = self.switch_penalty
        if self.action_history_enabled:
            for position in range(len(normalized_action_history)):
                objective[history_changed_group_index[position]] = (
                    self.action_history_penalty
                    * self.action_history_recency_weights[position]
                )
        if history_budget_slack_index is not None:
            objective[history_budget_slack_index] = (
                self.action_history_budget_slack_penalty
            )
        if per_group_history_budget_slack_index is not None:
            objective[per_group_history_budget_slack_index] = (
                self.action_history_budget_slack_penalty
            )
        for state_position, constraint in enumerate(self.state_constraints):
            penalty = float(constraint.get('penalty', 1000.0))
            objective[lower_slack_index[:, state_position]] = penalty
            objective[upper_slack_index[:, state_position]] = penalty

        variable_lower = np.full(layout.size, -np.inf, dtype=float)
        variable_upper = np.full(layout.size, np.inf, dtype=float)
        integrality = np.zeros(layout.size, dtype=int)

        variable_lower[z_index.ravel()] = 0.0
        variable_upper[z_index.ravel()] = 1.0
        integrality[z_index.ravel()] = 1
        variable_lower[power_index.ravel()] = 0.0
        variable_lower[lower_slack_index.ravel()] = 0.0
        variable_lower[upper_slack_index.ravel()] = 0.0
        variable_lower[delta_index.ravel()] = 0.0
        variable_lower[changed_group_index.ravel()] = 0.0
        variable_upper[changed_group_index.ravel()] = 1.0
        if len(normalized_action_history):
            variable_lower[history_changed_group_index.ravel()] = 0.0
            variable_upper[history_changed_group_index.ravel()] = 1.0
            integrality[history_changed_group_index.ravel()] = 1
        if history_budget_slack_index is not None:
            variable_lower[history_budget_slack_index] = 0.0
        if per_group_history_budget_slack_index is not None:
            variable_lower[per_group_history_budget_slack_index] = 0.0
            per_group_budgets = self.action_history_per_group_rolling_change_budgets
            for group_position, (name, _) in enumerate(self.action_history_groups):
                if name not in per_group_budgets:
                    variable_upper[per_group_history_budget_slack_index[group_position]] = 0.0

        for state_position, constraint in enumerate(self.state_constraints):
            if constraint.get('lower') is None:
                variable_upper[lower_slack_index[:, state_position]] = 0.0
            if constraint.get('upper') is None:
                variable_upper[upper_slack_index[:, state_position]] = 0.0

        constraints = _ConstraintBuilder()

        for state_position in range(nx):
            constraints.add(
                {x_index[0, state_position]: 1.0},
                lower=initial_state[state_position],
                upper=initial_state[state_position],
            )

        for step in range(m):
            for group in self.encoder.groups.values():
                constraints.add(
                    {z_index[step, idx]: 1.0 for idx in group},
                    lower=1.0,
                    upper=1.0,
                )

        for step in range(m):
            for rule in self.logical_rules:
                when = rule['when']
                antecedent = z_index[
                    step, self.encoder.index(when['variable'], when['value'])
                ]
                for variable, value in rule.get('require', {}).items():
                    consequent = z_index[step, self.encoder.index(variable, value)]
                    constraints.add(
                        {antecedent: 1.0, consequent: -1.0},
                        upper=0.0,
                    )
                for variable, values in rule.get('forbid', {}).items():
                    for value in values:
                        forbidden = z_index[step, self.encoder.index(variable, value)]
                        constraints.add(
                            {antecedent: 1.0, forbidden: 1.0},
                            upper=1.0,
                        )

        # A no-good cut excludes each complete categorical combination that
        # was logically valid but absent from the chronological training data.
        category_count = len(self.encoder.variables)
        for step in range(m):
            forbidden_vectors = self.forbidden_control_vectors
            if step == 0 and len(dynamic_forbidden):
                forbidden_vectors = np.vstack(
                    [forbidden_vectors, dynamic_forbidden]
                )
            for forbidden_vector in forbidden_vectors:
                selected = np.flatnonzero(forbidden_vector > 0.5)
                constraints.add(
                    {z_index[step, feature]: 1.0 for feature in selected},
                    upper=float(category_count - 1),
                )

        A, B, E, c = (
            self.state_model.A,
            self.state_model.B,
            self.state_model.E,
            self.state_model.c,
        )
        for step in range(n):
            control_step = min(step, m - 1)
            for output in range(nx):
                row = {x_index[step + 1, output]: 1.0}
                for feature in range(nx):
                    row[x_index[step, feature]] = -A[output, feature]
                for feature in range(nz):
                    row[z_index[control_step, feature]] = -B[output, feature]
                right_hand_side = float(E[output] @ disturbances[step] + c[output])
                constraints.add(row, lower=right_hand_side, upper=right_hand_side)

        F, G, H, b = (
            self.power_model.F,
            self.power_model.G,
            self.power_model.H,
            self.power_model.b,
        )
        for step in range(n):
            control_step = min(step, m - 1)
            for meter in range(npower):
                # Nonnegative epigraph: max(affine component prediction, 0).
                row = {power_index[step, meter]: 1.0}
                for feature in range(nx):
                    row[x_index[step, feature]] = -F[meter, feature]
                for feature in range(nz):
                    row[z_index[control_step, feature]] = -G[meter, feature]
                right_hand_side = float(H[meter] @ disturbances[step] + b[meter])
                constraints.add(row, lower=right_hand_side)

        for step in range(n):
            for state_position, specification in enumerate(self.state_constraints):
                state_variable = x_index[step + 1, state_position]
                lower = specification.get('lower')
                upper = specification.get('upper')
                if lower is not None:
                    constraints.add(
                        {
                            state_variable: 1.0,
                            lower_slack_index[step, state_position]: 1.0,
                        },
                        lower=float(lower),
                    )
                if upper is not None:
                    constraints.add(
                        {
                            state_variable: 1.0,
                            upper_slack_index[step, state_position]: -1.0,
                        },
                        upper=float(upper),
                    )

        for step in range(m):
            for feature in range(nz):
                delta = delta_index[step, feature]
                current = z_index[step, feature]
                if step == 0:
                    constraints.add(
                        {delta: 1.0, current: -1.0},
                        lower=-previous_control[feature],
                    )
                    constraints.add(
                        {delta: 1.0, current: 1.0},
                        lower=previous_control[feature],
                    )
                else:
                    prior = z_index[step - 1, feature]
                    constraints.add(
                        {delta: 1.0, current: -1.0, prior: 1.0},
                        lower=0.0,
                    )
                    constraints.add(
                        {delta: 1.0, current: 1.0, prior: -1.0},
                        lower=0.0,
                    )

            # ``delta`` tracks changed one-hot bits, so one categorical
            # category change produces two nonzero bits.  Group indicators
            # provide a clear count of changed *control variables* and allow
            # a hard switching cap without changing the legacy penalty.
            for group_position, group in enumerate(self.encoder.groups.values()):
                changed = changed_group_index[step, group_position]
                for feature in group:
                    current = z_index[step, feature]
                    if step == 0:
                        reference = float(previous_control[feature])
                        constraints.add(
                            {changed: 1.0, current: -1.0},
                            lower=-reference,
                        )
                        constraints.add(
                            {changed: 1.0, current: 1.0},
                            lower=reference,
                        )
                    else:
                        prior = z_index[step - 1, feature]
                        constraints.add(
                            {changed: 1.0, current: -1.0, prior: 1.0},
                            lower=0.0,
                        )
                        constraints.add(
                            {changed: 1.0, current: 1.0, prior: -1.0},
                            lower=0.0,
                        )
            if self.maximum_changed_control_variables_per_step is not None:
                constraints.add(
                    {
                        changed_group_index[step, group_position]: 1.0
                        for group_position in range(len(self.encoder.variables))
                    },
                    upper=float(self.maximum_changed_control_variables_per_step),
                )

        # At the receding-horizon boundary only z_0 is physically applied. The
        # recent-action regularizer therefore compares z_0 to each actually
        # applied control in the supplied chronological history. Future plan
        # steps retain the standard adjacent-step switching penalty and will be
        # re-solved with an updated history on the next real minute.
        if self.action_history_enabled:
            for position, lag in enumerate(
                range(1, len(normalized_action_history) + 1)
            ):
                reference_control = normalized_action_history[-lag]
                for group_position, (_, features) in enumerate(
                    self.action_history_groups
                ):
                    changed = history_changed_group_index[position, group_position]
                    for feature in features:
                        reference = float(reference_control[feature])
                        constraints.add(
                            {changed: 1.0, z_index[0, feature]: -1.0},
                            lower=-reference,
                        )
                        constraints.add(
                            {changed: 1.0, z_index[0, feature]: 1.0},
                            lower=reference,
                        )
            if history_budget_slack_index is not None:
                historical_changes = self._historical_change_count(
                    normalized_action_history
                )
                coefficients = {
                    history_changed_group_index[0, group_position]: 1.0
                    for group_position in range(len(self.action_history_groups))
                }
                coefficients[history_budget_slack_index[0]] = -1.0
                constraints.add(
                    coefficients,
                    upper=float(
                        self.action_history_rolling_change_budget
                        - historical_changes
                    ),
                )
            if per_group_history_budget_slack_index is not None:
                historical_changes_by_group = self._historical_change_counts(
                    normalized_action_history
                )
                per_group_budgets = self.action_history_per_group_rolling_change_budgets
                for group_position, (name, _) in enumerate(self.action_history_groups):
                    if name not in per_group_budgets:
                        continue
                    constraints.add(
                        {
                            history_changed_group_index[0, group_position]: 1.0,
                            per_group_history_budget_slack_index[group_position]: -1.0,
                        },
                        upper=float(
                            per_group_budgets[name]
                            - historical_changes_by_group[group_position]
                        ),
                    )

        result = guarded_milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(variable_lower, variable_upper),
            constraints=constraints.build(layout.size),
            options={
                'time_limit': self.time_limit_seconds,
                'mip_rel_gap': self.mip_relative_gap,
                'presolve': True,
            },
        )
        if result.x is None:
            return MPCSolution(
                success=False,
                status=int(result.status),
                message=str(result.message),
                objective=float('inf'),
                actions=tuple(),
                control_vectors=np.empty((0, nz)),
                state_trajectory=np.empty((0, nx)),
                power_components_w=np.empty((0, npower)),
                lower_slack=np.empty((0, nx)),
                upper_slack=np.empty((0, nx)),
            )

        control_vectors = self._canonicalize_categorical_controls(result.x[z_index])
        if control_vectors is None:
            # ``result.x`` can describe the continuous relaxation after a
            # time/deadline termination.  Returning it as an action would
            # violate the categorical MPC contract and corrupt future history.
            return MPCSolution(
                success=False,
                status=int(result.status),
                message=(
                    f'{result.message}; rejected nonintegral categorical '
                    'solver output without a feasible integer incumbent'
                ),
                objective=float('inf'),
                actions=tuple(),
                control_vectors=np.empty((0, nz)),
                state_trajectory=np.empty((0, nx)),
                power_components_w=np.empty((0, npower)),
                lower_slack=np.empty((0, nx)),
                upper_slack=np.empty((0, nx)),
            )
        actions = tuple(
            self.encoder.decode_action(vector) for vector in control_vectors
        )
        return MPCSolution(
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            objective=float(result.fun),
            actions=actions,
            control_vectors=control_vectors,
            state_trajectory=result.x[x_index],
            power_components_w=result.x[power_index],
            lower_slack=result.x[lower_slack_index],
            upper_slack=result.x[upper_slack_index],
        )
