"""절감 확대 ablation 전용 CEM. 기본 categorical_cem의 동작은 바꾸지 않는다."""

from time import perf_counter

import numpy as np

from controllers.control_benchmark import CategoricalCEM, ControlPlan
from models.control.base import categorical_changes, stage_cost


def physical_changes(encoder, a, b):
    """OFF와 연결된 부설정 차이를 무시한다. 실제 장비 사양 보증은 아니다.

    전원 전환은 항상 센다. 양쪽 ON일 때만 해당 장비의 부설정 전환을 센다.
    따라서 기존 raw categorical cap보다 느슨한 별도 실험 조건이다.
    """
    a, b = np.broadcast_arrays(a, b)
    result = np.zeros(a.shape[:-1], dtype=float)
    for variable in encoder.variables:
        indices = list(encoder.groups[variable.name])
        changed = np.argmax(a[..., indices], -1) != np.argmax(b[..., indices], -1)
        power_name = next((prefix+'전원' for prefix in ('환기', '냉난방')
                           if variable.name.startswith(prefix)
                           and variable.name != prefix+'전원'), None)
        if power_name in encoder.groups:
            on = encoder.index(power_name, 'ON')
            changed &= (a[..., on] > .5) & (b[..., on] > .5)
        result += changed
    return result


def violation_magnitudes(states, constraints):
    value = np.zeros_like(states, dtype=float)
    for j, item in enumerate(constraints):
        if item.get('lower') is not None:
            value[..., j] += np.maximum(float(item['lower'])-states[..., j], 0)
        if item.get('upper') is not None:
            value[..., j] += np.maximum(states[..., j]-float(item['upper']), 0)
    return value


class EnergyCategoricalCEM(CategoricalCEM):
    """에너지계수/품질예산/시간블록/명시warm/첫action균등/전환의 개별 ablation."""

    def __init__(self, model, actions, constraints, logical_rules, options, training_contexts):
        super().__init__(model, actions, constraints, logical_rules, options, training_contexts)
        self.energy_weight = float(options.get('energy_weight', 1))
        self.block = int(options.get('cem_block_steps', 1))
        self.quality_budget = bool(options.get('quality_budget', False))
        if not np.isfinite(self.energy_weight) or self.energy_weight <= 0 or self.block < 1:
            raise ValueError('Energy weight and CEM block must be positive')
        self.power_weights = np.asarray(options.get(
            'power_weights', [1.]*model.spec.np), dtype=float)
        if (self.power_weights.shape != (model.spec.np,)
                or not np.isfinite(self.power_weights).all()
                or (self.power_weights <= 0).any()):
            raise ValueError('power_weights must be finite positive per-component values')
        self.change = (lambda a, b: physical_changes(model.spec.encoder, a, b)
                       ) if options.get('physical_transition_count', False) else categorical_changes
        self.transitions = self.change(actions[:, None], actions[None]) <= int(
            options['maximum_changed_variables'])
        self.dwell = int(options.get('assumed_minimum_onoff_steps', 0))
        self.power_indices = [model.spec.encoder.index(n, 'ON')
                              for n in ('환기전원', '냉난방전원')
                              if n in model.spec.encoder.groups]
        if self.dwell < 0 or (self.dwell and not self.power_indices):
            raise ValueError('Dwell sensitivity needs nonnegative duration and known power variables')
        self.remaining = np.zeros(len(self.power_indices), dtype=int)
        self.last_diagnostics = {}

    def reset(self):
        self.warm = None
        self.last_diagnostics = {}
        # Initial lockout history is unknown: explicit simulation assumption.
        self.remaining[:] = 0

    def observe_applied(self, previous, applied):
        changed = np.asarray(previous)[self.power_indices] != np.asarray(applied)[self.power_indices]
        self.remaining = np.where(changed, max(0, self.dwell-1), np.maximum(self.remaining-1, 0))

    def _dwell_allowed(self, candidates, previous, remaining):
        if not self.dwell:
            return np.ones(np.broadcast_shapes(
                np.shape(candidates)[:-1], np.shape(previous)[:-1]), dtype=bool)
        changed = np.asarray(candidates)[..., self.power_indices] != np.asarray(previous)[..., self.power_indices]
        return ~((remaining > 0) & changed).any(-1)

    def _valid_warm(self, first, previous):
        if self.warm is None or not first[self.warm[0]]:
            return False
        remaining = self.remaining.copy()
        for i, index in enumerate(self.warm):
            action = self.actions[index]
            if i and not self.transitions[self.warm[i-1], index]:
                return False
            if not self._dwell_allowed(action, previous, remaining):
                return False
            changed = action[self.power_indices] != previous[self.power_indices]
            remaining = np.where(changed, max(0, self.dwell-1), np.maximum(remaining-1, 0))
            previous = action
        return True

    def _rollouts(self, context, disturbances, sequences, previous_control):
        """성분별 energy와 상태별 누적/peak를 유지해 평균으로 위반을 숨기지 않는다."""
        count = len(sequences)
        members = getattr(self.model, 'members', [self.model])
        energies, softs, sums, peaks, conventional = [], [], [], [], []
        penalties = np.asarray([float(c.get('penalty', 1000)) for c in self.constraints])
        for member in members:
            ctx = np.repeat(context[None], count, axis=0)
            energy = np.zeros((count, self.model.spec.np))
            cost = np.zeros(count)
            cumulative = np.zeros((count, self.model.spec.nx))
            peak = np.zeros_like(cumulative)
            for t, disturbance in enumerate(disturbances):
                z = self.actions[sequences[:, min(t, sequences.shape[1]-1)]]
                with np.errstate(over='ignore', invalid='ignore'):
                    nx, power = member.predict(
                        ctx, z, np.broadcast_to(disturbance, (count, len(disturbance))))
                    cost += stage_cost(nx, power, self.constraints)
                    energy += np.maximum(power, 0)/60
                    v = violation_magnitudes(nx, self.constraints)
                    cumulative += v
                    peak = np.maximum(peak, v)
                    ctx = self.model.spec.advance(ctx, z, nx)
            energies.append(energy)
            softs.append(cumulative @ penalties)
            sums.append(cumulative)
            peaks.append(peak)
            conventional.append(cost)
        switching = np.zeros(count)
        conventional = np.mean(conventional, axis=0)
        previous = np.broadcast_to(previous_control, (count, len(previous_control)))
        for t in range(sequences.shape[1]):
            z = self.actions[sequences[:, t]]
            # Objective remains raw one-hot switching; only feasibility cap can differ.
            increment = 2*float(self.options['switch_penalty'])*categorical_changes(z, previous)
            switching += increment
            conventional += increment
            previous = z
        return (np.mean(energies, axis=0), np.mean(softs, axis=0), switching,
                np.stack(sums), np.stack(peaks), conventional)

    def _hold_budgets(self, context, disturbances, previous_control):
        # Reference need not belong to the train-supported candidate set.
        # Roll it out separately rather than silently replacing it with a candidate.
        sums, peaks = [], []
        for member in getattr(self.model, 'members', [self.model]):
            ctx = context.copy()
            total, peak = np.zeros(self.model.spec.nx), np.zeros(self.model.spec.nx)
            for disturbance in disturbances:
                nx, _ = member.predict(ctx, previous_control, disturbance)
                v = violation_magnitudes(nx[0], self.constraints)
                total += v
                peak = np.maximum(peak, v)
                ctx = self.model.spec.advance(ctx, previous_control, nx[0])
            sums.append(total)
            peaks.append(peak)
        return np.asarray(sums), np.asarray(peaks)

    def solve(self, context, disturbances, previous_control, first_allowed):
        start = perf_counter()
        self.last_diagnostics = {}
        m, k = int(self.options['control_horizon']), len(self.actions)
        cap = int(self.options['maximum_changed_variables'])
        first = np.asarray(first_allowed, dtype=bool) & (self.change(
            self.actions, previous_control) <= cap)
        first &= self._dwell_allowed(self.actions, previous_control, self.remaining)
        if not first.any():
            return ControlPlan(np.empty((0, self.model.spec.nz)), None,
                               'no_eligible_action', perf_counter()-start)
        probability = np.full((m, k), 1/k)
        if self.warm is not None:
            probability *= .5
            probability[np.arange(m), self.warm] += .5
        budgets = self._hold_budgets(context, disturbances, previous_control) if self.quality_budget else None
        best, best_key, best_objective, best_diag = None, None, None, None
        population = int(self.options.get('cem_population', 64))
        elite_count = max(1, int(population*float(self.options.get('cem_elite_fraction', .2))))
        evaluations = 0
        for iteration in range(int(self.options.get('cem_iterations', 3))):
            sequences = np.empty((population, m), dtype=int)
            remaining = np.broadcast_to(self.remaining, (population, len(self.remaining))).copy()
            preceding = np.broadcast_to(previous_control, (population, len(previous_control)))
            for t in range(m):
                if t and t % self.block:
                    sequences[:, t] = sequences[:, t-1]
                else:
                    allowed = (np.broadcast_to(first, (population, k)).copy() if t == 0
                               else self.transitions[sequences[:, t-1]].copy())
                    if self.dwell:
                        allowed &= self._dwell_allowed(
                            self.actions[None], preceding[:, None], remaining[:, None])
                    probs = probability[t][None]*allowed
                    probs /= probs.sum(1, keepdims=True)
                    cumulative = probs.cumsum(1)
                    cumulative[:, -1] = 1
                    sequences[:, t] = (self.rng.random(population)[:, None] > cumulative).sum(1)
                    if t == 0 and self.options.get('stratified_first_actions', False):
                        eligible = self.rng.permutation(np.flatnonzero(first))
                        sequences[:, 0] = np.resize(eligible, population)
                applied = self.actions[sequences[:, t]]
                changed = applied[:, self.power_indices] != preceding[:, self.power_indices]
                remaining = np.where(changed, max(0, self.dwell-1), np.maximum(remaining-1, 0))
                preceding = applied
            anchors = np.repeat(np.flatnonzero(first)[:, None], m, axis=1)
            sequences = np.concatenate([sequences, anchors])
            if self.options.get('explicit_warm_candidate', False) and self.warm is not None:
                if self._valid_warm(first, previous_control):
                    # Replace a random sample: warm candidate adds no evaluations.
                    sequences[0] = self.warm
            if best is not None:
                sequences = np.concatenate([sequences, best[None]])
            energy, soft, switch, sums, peaks, conventional = self._rollouts(
                context, disturbances, sequences, previous_control)
            # Exact original accumulation order when all energy coefficients are1.
            objective = conventional+energy @ (self.energy_weight*self.power_weights-1)
            finite = (np.isfinite(objective) & np.isfinite(sums).all((0, 2))
                      & np.isfinite(peaks).all((0, 2)))
            if budgets is None:
                keys = [(0, float(v), 0.) if ok else (2, np.inf, np.inf)
                        for v, ok in zip(objective, finite)]
            else:
                cumulative_budget, peak_budget = budgets
                if not np.isfinite(cumulative_budget).all() or not np.isfinite(peak_budget).all():
                    return ControlPlan(np.empty((0, self.model.spec.nz)), None,
                                       'nonfinite_quality_reference', perf_counter()-start)
                excess_sum = np.maximum(sums-cumulative_budget[:, None]-1e-8, 0)
                excess_peak = np.maximum(peaks-peak_budget[:, None]-1e-8, 0)
                feasible = (excess_sum <= 0).all((0, 2)) & (excess_peak <= 0).all((0, 2))
                recovery = (excess_sum/(1+cumulative_budget[:, None])).sum((0, 2))
                recovery += (excess_peak/(1+peak_budget[:, None])).sum((0, 2))
                energy_objective = self.energy_weight*(energy @ self.power_weights)+switch
                keys = [
                    ((0, float(e), 0.) if safe else (1, float(r), float(e))) if ok
                    else (2, np.inf, np.inf)
                    for safe, r, e, ok in zip(feasible, recovery, energy_objective, finite)]
            order = sorted(range(len(keys)), key=keys.__getitem__)
            evaluations += len(sequences)
            winner = order[0]
            if keys[winner][0] < 2 and (best_key is None or keys[winner] < best_key):
                best, best_key = sequences[winner].copy(), keys[winner]
                best_objective = float(objective[winner])
                best_diag = {
                    'energy_wh': energy[winner].tolist(),
                    'weighted_state_violation': float(soft[winner]),
                    'raw_switching_cost': float(switch[winner]),
                    'quality_budget_feasible': bool(best_key[0] == 0) if budgets is not None else None,
                    'quality_recovery_mode': bool(best_key[0] == 1) if budgets is not None else False,
                    'raw_first_changes': float(categorical_changes(
                        self.actions[best[0]], previous_control)),
                    'effective_first_changes': float(self.change(
                        self.actions[best[0]], previous_control)),
                }
            elite = sequences[order[:elite_count]]
            frequency = np.stack([np.bincount(elite[:, t], minlength=k)
                                  for t in range(m)]).astype(float)
            frequency = (frequency+.1)/(elite_count+.1*k)
            probability = .2*probability+.8*frequency
        if best is None:
            return ControlPlan(np.empty((0, self.model.spec.nz)), None,
                               'nonfinite_rollouts', perf_counter()-start)
        self.warm = np.concatenate([best[1:], best[-1:]])
        self.last_diagnostics = {**best_diag, 'candidate_rollout_count': evaluations,
                                 'ranking_key': list(best_key),
                                 'assumed_minimum_onoff_steps': self.dwell,
                                 'initial_remaining_onoff_steps': self.remaining.tolist()}
        status = 'energy_cem_' + ('recovery' if best_key[0] == 1 else 'feasible')
        return ControlPlan(self.actions[best], best_objective, status, perf_counter()-start)
