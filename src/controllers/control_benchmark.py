"""Interchangeable baseline, joint-scenario MILP, and categorical CEM planners."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds

from controllers.guarded_milp import guarded_milp

from controllers.mixed_integer_mpc import MixedIntegerMPC, _VariableLayout, _ConstraintBuilder
from models.control.base import categorical_changes, stage_cost


@dataclass
class ControlPlan:
    controls: np.ndarray
    objective: float | None
    status: str
    elapsed_seconds: float
    optimal: bool = False
    mip_gap: float | None = None


class BaselinePlanner:
    """The existing affine MILP, not a reimplementation of its baseline."""

    def __init__(self, model, actions, constraints, logical_rules, options, training_contexts):
        del training_contexts
        spec = model.spec
        valid = spec.encoder.valid_control_vectors(logical_rules)
        keys = {tuple(a.astype(int)) for a in actions}
        forbidden = np.asarray([a for a in valid if tuple(a.astype(int)) not in keys])
        self.actions = actions
        self.solver = MixedIntegerMPC(
            model.state_model, model.power_model, spec.encoder, constraints, logical_rules,
            prediction_horizon=int(options['prediction_horizon']),
            control_horizon=int(options['control_horizon']),
            switch_penalty=float(options['switch_penalty']),
            maximum_changed_control_variables_per_step=int(options['maximum_changed_variables']),
            time_limit_seconds=float(options['time_limit_seconds']),
            mip_relative_gap=float(options['mip_relative_gap']),
            forbidden_control_vectors=forbidden,
        )

    def solve(self, context, disturbances, previous_control, first_allowed):
        start = perf_counter()
        forbidden = self.actions[~np.asarray(first_allowed)]
        result = self.solver.solve(context[:len(self.solver.state_model.state_names)],
                                   disturbances, previous_control,
                                   first_step_forbidden_control_vectors=forbidden)
        return ControlPlan(result.control_vectors, result.objective if len(result.control_vectors) else None,
                           f'highs_status_{result.status}', perf_counter()-start,
                           optimal=result.status == 0)


class JointScenarioMILP:
    """Exact finite-action disjunctions for bounded affine scenario models.

    Model scenarios share actions but have separate state trajectories. Big-M
    constants are computed from explicit train-derived state boxes, not magic
    global constants. Box restrictions are disclosed in experiment metadata.
    """

    def __init__(self, model, actions, constraints, logical_rules, options, training_contexts):
        del logical_rules
        if not model.supports_milp:
            raise TypeError(f'{model.family} is not affine-representable')
        self.model, self.actions, self.constraints, self.options = model, actions, constraints, options
        self.members = getattr(model, 'members', [model])
        spec = model.spec
        x = training_contexts[:, :spec.nx]
        span = np.maximum(x.max(0)-x.min(0), 1.0)
        padding = float(options.get('state_box_padding', 2.0))
        self.state_lower, self.state_upper = x.min(0)-padding*span, x.max(0)+padding*span
        self.transitions = categorical_changes(actions[:, None], actions[None]) <= int(options['maximum_changed_variables'])
        self.mode_matrix_strategy = str(options.get('mode_matrix_strategy', 'exact_disjunction'))
        if self.mode_matrix_strategy not in ('nominal_linearization', 'exact_disjunction'):
            raise ValueError('Unknown mode_matrix_strategy')
        self.warm = None

    def solve(self, context, disturbances, previous_control, first_allowed):
        start = perf_counter()
        s = self.model.spec
        n, m, k, q = len(disturbances), int(self.options['control_horizon']), len(self.actions), len(self.members)
        layout = _VariableLayout()
        cx = layout.reserve((q, n+1, s.width))
        u = layout.reserve((m, k))
        categorical = layout.reserve((m, s.nz))
        power = layout.reserve((q, n, s.np))
        lo = layout.reserve((q, n, s.nx))
        hi = layout.reserve((q, n, s.nx))
        delta = layout.reserve((m, s.nz))
        risk = layout.reserve((1,))
        c = np.zeros(layout.size)
        lower, upper = np.full(layout.size, -np.inf), np.full(layout.size, np.inf)
        integer = np.zeros(layout.size, dtype=int)
        xl = np.minimum(self.state_lower, np.minimum(context[:s.nx], context[s.nx:2*s.nx]))
        xu = np.maximum(self.state_upper, np.maximum(context[:s.nx], context[s.nx:2*s.nx]))
        cl = np.concatenate([xl, xl, np.zeros(2*s.nz)])
        cu = np.concatenate([xu, xu, np.ones(2*s.nz)])
        lower[cx] = cl
        upper[cx] = cu
        # Use only sum(category counts)*m integer variables. Nonnegative joint
        # weights linked to integral one-hot marginals must select exactly one
        # distinct joint action; K*m extra binaries are unnecessary.
        lower[u], upper[u] = 0, 1
        lower[categorical], upper[categorical], integer[categorical] = 0, 1, 1
        upper[u[0, ~np.asarray(first_allowed)]] = 0
        for indices in (power, lo, hi, delta):
            lower[indices] = 0
        lower[risk] = 0
        constraints = _ConstraintBuilder()
        scenario_cost = []
        risk_weight = float(self.options.get('risk_weight', 0.5)) if q > 1 else 0.0
        c[delta] = float(self.options['switch_penalty'])
        for t in range(m):
            constraints.add({j: 1.0 for j in u[t]}, lower=1, upper=1)
            for j in range(s.nz):
                row = {categorical[t, j]: 1}
                row.update({u[t, a]: -self.actions[a, j] for a in range(k)})
                constraints.add(row, lower=0, upper=0)
        # Shared per-variable switching cap, including within the planned sequence.
        cap = int(self.options['maximum_changed_variables'])
        for i, action in enumerate(self.actions):
            if categorical_changes(action, previous_control) > cap:
                upper[u[0, i]] = 0
        # Exact L1 switching epigraph, consistent with the legacy solver.
        for t in range(m):
            constraints.add({index: 1 for index in delta[t]}, upper=2*cap)
            for j in range(s.nz):
                row = {delta[t, j]: 1}
                row.update({u[t, a]: -self.actions[a, j] for a in range(k)})
                rhs = -previous_control[j] if t == 0 else 0
                if t:
                    row.update({u[t-1, a]: self.actions[a, j] for a in range(k)})
                constraints.add(row, lower=rhs)
                row = {index: -value for index, value in row.items() if index != delta[t, j]}
                row[delta[t, j]] = 1
                constraints.add(row, lower=-rhs)
        for scenario, member in enumerate(self.members):
            cost = {}
            nominal_context = context.copy()
            for j in range(s.width):
                constraints.add({cx[scenario, 0, j]: 1}, lower=context[j], upper=context[j])
            for t in range(n):
                step = min(t, m-1)
                A, b, F, g = member.linear_coefficients(self.actions, disturbances[t])
                if self.mode_matrix_strategy == 'nominal_linearization':
                    # First-order local model: exact for every candidate at
                    # the reference context, but not globally exact after a
                    # different trajectory. Refresh each receding-horizon solve.
                    nominal_action = previous_control if self.warm is None else self.warm[step]
                    An, _, Fn, _ = member.linear_coefficients(nominal_action[None], disturbances[t])
                    b = b + np.einsum('kij,j->ki', A-An, nominal_context)
                    g = g + np.einsum('kij,j->ki', F-Fn, nominal_context)
                    for j in range(s.nx):
                        row = {cx[scenario, t+1, j]: 1}
                        row.update({cx[scenario, t, v]: -An[0, j, v]
                                    for v in np.flatnonzero(An[0, j])})
                        row.update({u[step, a]: -b[a, j] for a in range(k)})
                        constraints.add(row, lower=0, upper=0)
                    for j in range(s.np):
                        row = {power[scenario, t, j]: 1}
                        row.update({cx[scenario, t, v]: -Fn[0, j, v]
                                    for v in np.flatnonzero(Fn[0, j])})
                        row.update({u[step, a]: -g[a, j] for a in range(k)})
                        constraints.add(row, lower=0)
                    nx_nominal, _ = member.predict(nominal_context, nominal_action, disturbances[t])
                    nominal_context = np.clip(s.advance(nominal_context, nominal_action, nx_nominal[0]), cl, cu)
                # Actions with the same mode-dependent state matrix share one
                # disjunction. Offsets remain exact linear mixtures of the
                # shared joint-action indicators (not an approximation).
                signatures = np.concatenate([A.reshape(k, -1), F.reshape(k, -1)], axis=1)
                _, representatives, memberships = np.unique(
                    signatures, axis=0, return_index=True, return_inverse=True)
                if self.mode_matrix_strategy == 'nominal_linearization':
                    representatives = []
                for group, action in enumerate(representatives):
                    active_indices = u[step, memberships == group]
                    for j in range(s.nx):
                        co = A[action, j]
                        row = {cx[scenario, t+1, j]: 1}
                        row.update({cx[scenario, t, v]: -co[v] for v in np.flatnonzero(co)})
                        row.update({u[step, a]: -b[a, j] for a in range(k)})
                        expr_min = xl[j] - (np.maximum(co, 0) @ cu + np.minimum(co, 0) @ cl) - b[:, j].max()
                        expr_max = xu[j] - (np.maximum(co, 0) @ cl + np.minimum(co, 0) @ cu) - b[:, j].min()
                        ml, mu = max(float(-expr_min), 0)+1e-7, max(float(expr_max), 0)+1e-7
                        low_row, high_row = dict(row), dict(row)
                        for index in active_indices:
                            low_row[index] = low_row.get(index, 0)-ml
                            high_row[index] = high_row.get(index, 0)+mu
                        constraints.add(low_row, lower=-ml)
                        constraints.add(high_row, upper=mu)
                    for j in range(s.np):
                        co = F[action, j]
                        # power >= max(affine selected component, 0).
                        maximum = float(np.maximum(co, 0) @ cu + np.minimum(co, 0) @ cl + g[:, j].max())
                        big_m = max(maximum, 0) + 1e-7
                        row = {power[scenario, t, j]: 1}
                        row.update({cx[scenario, t, v]: -co[v] for v in np.flatnonzero(co)})
                        row.update({u[step, a]: -g[a, j] for a in range(k)})
                        for index in active_indices:
                            row[index] = row.get(index, 0)-big_m
                        constraints.add(row, lower=-big_m)
                # Shift previous state, previous action and exponentially weighted action memory.
                for j in range(s.nx):
                    constraints.add({cx[scenario, t+1, s.nx+j]: 1,
                                     cx[scenario, t, j]: -1}, lower=0, upper=0)
                for j in range(s.nz):
                    action_row = {u[step, a]: -self.actions[a, j] for a in range(k)}
                    constraints.add({**action_row, cx[scenario, t+1, 2*s.nx+j]: 1}, lower=0, upper=0)
                    memory_row = {u[step, a]: -(1-s.decay)*self.actions[a, j] for a in range(k)}
                    memory_row.update({cx[scenario, t+1, 2*s.nx+s.nz+j]: 1,
                                       cx[scenario, t, 2*s.nx+s.nz+j]: -s.decay})
                    constraints.add(memory_row, lower=0, upper=0)
                for j, item in enumerate(self.constraints):
                    penalty = float(item.get('penalty', 1000))
                    for idx in (lo[scenario, t, j], hi[scenario, t, j]):
                        cost[idx] = penalty
                    if item.get('lower') is not None:
                        constraints.add({cx[scenario, t+1, j]: 1, lo[scenario, t, j]: 1},
                                        lower=float(item['lower']))
                    else:
                        upper[lo[scenario, t, j]] = 0
                    if item.get('upper') is not None:
                        constraints.add({cx[scenario, t+1, j]: 1, hi[scenario, t, j]: -1},
                                        upper=float(item['upper']))
                    else:
                        upper[hi[scenario, t, j]] = 0
                for index in power[scenario, t]:
                    cost[index] = 1/60
            scenario_cost.append(cost)
            for index, value in cost.items():
                c[index] += (1-risk_weight)*value/q
            constraints.add({risk[0]: 1, **{index: -v for index, v in cost.items()}}, lower=0)
        c[risk] = risk_weight
        matrix = constraints.build(layout.size)
        result = guarded_milp(c, integrality=integer, bounds=Bounds(lower, upper), constraints=matrix,
                      options={'time_limit': float(self.options['time_limit_seconds']),
                               'mip_rel_gap': float(self.options['mip_relative_gap']),
                               'presolve': True})
        controls = np.empty((0, s.nz))
        accepted = False
        if result.x is not None and np.isfinite(result.x).all():
            weights = result.x[u]
            selected = np.argmax(weights, axis=1)
            integral = np.allclose(weights, np.eye(k)[selected], atol=1e-5)
            lhs = matrix.A @ result.x
            feasible = (np.all(lhs >= matrix.lb - 1e-4 - 1e-7*np.abs(np.where(np.isfinite(matrix.lb), matrix.lb, 0)))
                        and np.all(lhs <= matrix.ub + 1e-4 + 1e-7*np.abs(np.where(np.isfinite(matrix.ub), matrix.ub, 0))))
            if integral and feasible:
                controls, accepted = self.actions[selected], True
                self.warm = np.concatenate([controls[1:], controls[-1:]], axis=0)
        gap = getattr(result, 'mip_gap', None)
        return ControlPlan(controls, float(result.fun) if accepted else None,
                           f'{self.mode_matrix_strategy}_highs_status_{result.status}' + ('' if accepted else '_no_feasible_incumbent'),
                           perf_counter()-start, bool(result.status == 0 and accepted),
                           float(gap) if gap is not None and np.isfinite(gap) else None)


class CategoricalCEM:
    """Finite joint-action CEM; no continuous-action rounding."""

    def __init__(self, model, actions, constraints, logical_rules, options, training_contexts):
        del logical_rules, training_contexts
        if int(options.get('cem_population', 64)) < 1 or int(options.get('cem_iterations', 3)) < 1:
            raise ValueError('CEM population and iterations must be positive')
        if not 0 < float(options.get('cem_elite_fraction', 0.2)) <= 1:
            raise ValueError('CEM elite fraction must be in (0,1]')
        self.model, self.actions, self.constraints, self.options = model, actions, constraints, options
        self.rng = np.random.default_rng(int(options.get('seed', 42)))
        self.transitions = categorical_changes(actions[:, None], actions[None]) <= int(options['maximum_changed_variables'])
        self.warm = None

    def _costs(self, context, disturbances, sequences, previous_control):
        count, horizon = len(sequences), len(disturbances)
        members = getattr(self.model, 'members', [self.model])
        totals = []
        for member in members:
            ctx = np.repeat(context[None], count, axis=0)
            cost = np.zeros(count)
            for t in range(horizon):
                z = self.actions[sequences[:, min(t, sequences.shape[1]-1)]]
                d = np.repeat(disturbances[t][None], count, axis=0)
                with np.errstate(over='ignore', invalid='ignore'):
                    nx, p = member.predict(ctx, z, d)
                    cost += stage_cost(nx, p, self.constraints)
                    ctx = self.model.spec.advance(ctx, z, nx)
            totals.append(cost)
        value = np.mean(totals, axis=0)
        prev = np.broadcast_to(previous_control, (count, len(previous_control)))
        for t in range(sequences.shape[1]):
            z = self.actions[sequences[:, t]]
            value += 2*float(self.options['switch_penalty'])*categorical_changes(z, prev)
            prev = z
        return np.where(np.isfinite(value), value, np.inf)

    def solve(self, context, disturbances, previous_control, first_allowed):
        start = perf_counter()
        m, k = int(self.options['control_horizon']), len(self.actions)
        first = np.asarray(first_allowed) & (
            categorical_changes(self.actions, previous_control) <= int(self.options['maximum_changed_variables']))
        if not first.any():
            return ControlPlan(np.empty((0, self.model.spec.nz)), None, 'no_eligible_action', perf_counter()-start)
        probability = np.full((m, k), 1/k)
        if self.warm is not None:
            probability *= 0.5
            probability[np.arange(m), self.warm] += 0.5
        best, best_value = None, np.inf
        population = int(self.options.get('cem_population', 64))
        elite_count = max(1, int(population*float(self.options.get('cem_elite_fraction', 0.2))))
        for iteration in range(int(self.options.get('cem_iterations', 3))):
            sequences = np.empty((population, m), dtype=int)
            for t in range(m):
                allowed = (np.broadcast_to(first, (population, k)) if t == 0
                           else self.transitions[sequences[:, t-1]])
                probs = probability[t][None] * allowed
                probs /= probs.sum(1, keepdims=True)
                cumulative = probs.cumsum(1)
                cumulative[:, -1] = 1.0
                sequences[:, t] = (self.rng.random(population)[:, None] > cumulative).sum(1)
            # Evaluate every eligible constant-action sequence as anchors.
            anchors = np.repeat(np.flatnonzero(first)[:, None], m, axis=1)
            sequences = np.concatenate([sequences, anchors], axis=0)
            if best is not None:
                sequences = np.concatenate([sequences, best[None]], axis=0)
            values = self._costs(context, disturbances, sequences, previous_control)
            order = np.argsort(values, kind='stable')
            if values[order[0]] < best_value:
                best, best_value = sequences[order[0]].copy(), float(values[order[0]])
            elite = sequences[order[:elite_count]]
            frequencies = np.stack([np.bincount(elite[:, t], minlength=k) for t in range(m)]).astype(float)
            frequencies = (frequencies+0.1)/(elite_count+0.1*k)
            probability = 0.2*probability + 0.8*frequencies
        if best is None:
            return ControlPlan(np.empty((0, self.model.spec.nz)), None, 'nonfinite_rollouts', perf_counter()-start)
        self.warm = np.concatenate([best[1:], best[-1:]])
        return ControlPlan(self.actions[best], best_value, 'cem_feasible_not_globally_optimal',
                           perf_counter()-start)


def create_control_planner(family, model, actions, constraints, logical_rules, options, training_contexts):
    registry = {'baseline_milp': BaselinePlanner, 'joint_milp': JointScenarioMILP,
                'scenario_milp': JointScenarioMILP, 'categorical_cem': CategoricalCEM}
    if family not in registry:
        raise ValueError(f'Unknown control planner {family!r}: {tuple(registry)}')
    if family == 'baseline_milp' and model.family != 'affine':
        raise TypeError('baseline_milp requires the affine model; use joint_milp or categorical_cem')
    if family == 'scenario_milp' and not getattr(model, 'members', None):
        raise ValueError('scenario_milp requires an ensemble')
    return registry[family](model, actions, constraints, logical_rules, options, training_contexts)
