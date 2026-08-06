"""Exact categorical MPC formulated as a mixed-integer linear program."""

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import Bounds, LinearConstraint, milp
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
        self._validate_contract()

    def _validate_contract(self):
        if self.prediction_horizon <= 0:
            raise ValueError('prediction_horizon must be positive')
        if not 0 < self.control_horizon <= self.prediction_horizon:
            raise ValueError('control_horizon must be in [1, prediction_horizon]')
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

    def solve(
        self,
        initial_state: np.ndarray,
        disturbances: np.ndarray,
        previous_control: np.ndarray | None = None,
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

        layout = _VariableLayout()
        x_index = layout.reserve((n + 1, nx))
        z_index = layout.reserve((m, nz))
        power_index = layout.reserve((n, npower))
        lower_slack_index = layout.reserve((n, nx))
        upper_slack_index = layout.reserve((n, nx))
        delta_index = layout.reserve((m, nz))

        objective = np.zeros(layout.size, dtype=float)
        # A one-minute sample of power in W contributes W / 60 Wh.
        objective[power_index.ravel()] = 1.0 / 60.0
        objective[delta_index.ravel()] = self.switch_penalty
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
            for forbidden_vector in self.forbidden_control_vectors:
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

        result = milp(
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

        control_vectors = result.x[z_index]
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
