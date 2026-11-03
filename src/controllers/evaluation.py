"""Evaluation utilities for independent, receding-horizon MPC backtests."""

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Dict, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf

from dynamics.affine import power_mape_metrics
from dynamics.evaluator_base import history_for_rollout, start_rollout_session


LOGGER = logging.getLogger('MPC.evaluation')

REPORT_MODEL_REFIT_SCOPES = ('train_validation', 'train_validation_test')


def resolve_report_model_refit_rows(
    data,
    training_rows,
    validation_rows,
    *,
    test_rows=None,
    scope: str = 'train_validation',
) -> np.ndarray:
    """frozen report model을 다시 학습할 행 집합을 refit scope 설정에 따라 만든다.

    - ``train_validation`` (기본): train + validation 만 사용하고 test는 채점 전용으로
      남긴다(=held-out test 오차 추정). 지금까지의 동작과 동일하다.
    - ``train_validation_test``: 배포용 instrument 목적으로 test까지 포함해 학습한다.
      **이 경우 test 실제값으로 학습하므로, 그 뒤 test에서 잰 MAPE/WAPE는 held-out
      오차가 아니라 누수된 값**이 된다. KPI(일반화 성능) 주장에는 쓸 수 없다.
    """
    scope = str(scope)
    if scope not in REPORT_MODEL_REFIT_SCOPES:
        raise ValueError(
            f'report_model_refit_scope must be one of {REPORT_MODEL_REFIT_SCOPES}; '
            f'got {scope!r}'
        )
    parts = [
        data.valid_transition_rows(training_rows),
        data.valid_transition_rows(validation_rows),
    ]
    if scope == 'train_validation_test':
        if test_rows is None:
            raise ValueError(
                "report_model_refit_scope='train_validation_test' requires test_rows"
            )
        parts.append(data.valid_transition_rows(test_rows))
    return np.unique(np.concatenate(parts))


@dataclass(frozen=True)
class PlantRollout:
    states: np.ndarray
    powers: np.ndarray


@dataclass(frozen=True)
class GatedMPCSolveResult:
    """First-action solve outcome with a train-only eligibility audit.

    ``solution`` is ``None`` only when the candidate gate leaves no action.
    In that case the caller must hold the previous applied action rather than
    inventing a test-recorded action as a fallback.
    """

    solution: object | None
    applied_control: np.ndarray
    has_incumbent: bool
    used_fallback: bool
    fallback_reason: str | None
    retry_count: int
    candidate_decisions: tuple[object, ...]
    selected_decision: object | None

    def summary(self) -> Dict[str, object]:
        candidates = self.candidate_decisions
        trust_rejected = 0
        model_rejected = 0
        report_action_effect_rejected = 0
        state_safety_relaxed = False
        for decision in candidates:
            screen = getattr(decision, 'model_screen', None)
            if screen is not None and 'state_safety_relaxed_non_worsening' in tuple(
                getattr(screen, 'notes', ())
            ):
                state_safety_relaxed = True
            reasons = tuple(getattr(decision, 'rejection_reasons', ()))
            trust = getattr(decision, 'trust', None)
            if trust is not None and not bool(getattr(trust, 'eligible', False)):
                trust_rejected += 1
            if any(
                reason.startswith('affine_evaluator_')
                or reason.startswith('evaluator_state_')
                or reason.startswith('independent_model_')
                for reason in reasons
            ):
                model_rejected += 1
            if any(
                reason.startswith('report_total_action_effect_')
                or reason.startswith('report_component_action_effect_')
                or reason.startswith('report_action_effect_')
                for reason in reasons
            ):
                report_action_effect_rejected += 1
        return {
            'candidate_count': int(len(candidates)),
            'eligible_candidate_count': int(
                sum(bool(getattr(item, 'eligible', False)) for item in candidates)
            ),
            'trust_rejected_candidate_count': int(trust_rejected),
            'model_rejected_candidate_count': int(model_rejected),
            'report_action_effect_rejected_candidate_count': int(
                report_action_effect_rejected
            ),
            'used_action_gate_fallback': bool(self.used_fallback),
            'action_gate_fallback_reason': self.fallback_reason,
            'action_gate_retry_count': int(self.retry_count),
            'state_safety_relaxed_non_worsening': bool(state_safety_relaxed),
        }


def solve_first_action_with_gate(
    controller,
    *,
    initial_state: np.ndarray,
    disturbances: np.ndarray,
    previous_control: np.ndarray,
    evaluator,
    action_gate=None,
    action_history: np.ndarray | None = None,
    maximum_retries: int = 1,
) -> GatedMPCSolveResult:
    """Solve one MPC step after optional local-support/model screening.

    The dynamic exclusions apply only to ``z_0``.  Later actions in the 60
    minute plan are provisional; at the next real minute a measured state is
    available and this function builds a fresh first-action screen again.
    """
    previous_control = np.asarray(previous_control, dtype=float).copy()
    retries = int(maximum_retries)
    if retries < 0:
        raise ValueError('maximum_retries must be non-negative')
    if action_gate is None:
        solve_kwargs = {
            'initial_state': initial_state,
            'disturbances': disturbances,
            'previous_control': previous_control,
        }
        if action_history is not None:
            solve_kwargs['action_history'] = action_history
        solution = controller.solve(
            **solve_kwargs,
        )
        has_incumbent = bool(len(solution.control_vectors))
        return GatedMPCSolveResult(
            solution=solution,
            applied_control=(
                solution.control_vectors[0].copy()
                if has_incumbent
                else previous_control
            ),
            has_incumbent=has_incumbent,
            used_fallback=not has_incumbent,
            fallback_reason=(None if has_incumbent else 'solver_no_incumbent'),
            retry_count=0,
            candidate_decisions=(),
            selected_decision=None,
        )

    gate_kwargs = {'evaluator': evaluator}
    # Preserve the lightweight gate protocol used by diagnostic tests and
    # external callers.  Only the concrete report-action-effect extension
    # needs an explicit reference action.
    if (
        getattr(action_gate, 'report_action_effect_gate', None) is not None
        or getattr(action_gate, 'requires_reference_control', False)
    ):
        gate_kwargs['reference_control'] = previous_control
    allowed, decisions = action_gate.evaluate(
        initial_state,
        np.asarray(disturbances, dtype=float)[0],
        **gate_kwargs,
    )
    decisions = tuple(decisions)
    if not len(allowed):
        return GatedMPCSolveResult(
            solution=None,
            applied_control=previous_control,
            has_incumbent=False,
            used_fallback=True,
            fallback_reason='no_trusted_first_action',
            retry_count=0,
            candidate_decisions=decisions,
            selected_decision=None,
        )

    def control_key(vector) -> tuple[int, ...]:
        return tuple(
            np.rint(np.asarray(vector, dtype=float)).astype(int).tolist()
        )

    # The gate is a *candidate-pool restriction*, not merely a way to reject
    # the candidates it happens to inspect.  Without these first-step cuts a
    # controller with a broader static domain could select an action absent
    # from the gate pool and escape the trust/screen policy.  Static forbidden
    # actions are already handled for every planned step by the controller;
    # only otherwise-valid first actions need to be added here.
    eligible_keys = {control_key(vector) for vector in np.asarray(allowed, dtype=float)}
    static_forbidden_keys = {
        control_key(vector)
        for vector in np.asarray(controller.forbidden_control_vectors, dtype=float)
    }
    forbidden_by_key = {
        control_key(decision.encoded_control): np.asarray(
            decision.encoded_control,
            dtype=float,
        )
        for decision in decisions
        if (
            decision.encoded_control is not None
            and not bool(decision.eligible)
        )
    }
    for vector in controller.encoder.valid_control_vectors(controller.logical_rules):
        key = control_key(vector)
        if key not in eligible_keys and key not in static_forbidden_keys:
            forbidden_by_key.setdefault(key, np.asarray(vector, dtype=float))
    forbidden = list(forbidden_by_key.values())
    decision_by_key = {
        control_key(decision.encoded_control): decision
        for decision in decisions
        if decision.encoded_control is not None
    }
    for retry in range(retries + 1):
        forbidden_matrix = (
            np.asarray(forbidden, dtype=float)
            if forbidden
            else np.empty((0, previous_control.size), dtype=float)
        )
        solve_kwargs = {
            'initial_state': initial_state,
            'disturbances': disturbances,
            'previous_control': previous_control,
            'first_step_forbidden_control_vectors': forbidden_matrix,
        }
        if action_history is not None:
            solve_kwargs['action_history'] = action_history
        solution = controller.solve(
            **solve_kwargs,
        )
        has_incumbent = bool(len(solution.control_vectors))
        if not has_incumbent:
            return GatedMPCSolveResult(
                solution=solution,
                applied_control=previous_control,
                has_incumbent=False,
                used_fallback=True,
                fallback_reason='solver_no_incumbent_after_action_gate',
                retry_count=retry,
                candidate_decisions=decisions,
                selected_decision=None,
            )
        selected = solution.control_vectors[0].copy()
        selected_decision = decision_by_key.get(control_key(selected))
        if selected_decision is not None and bool(selected_decision.eligible):
            return GatedMPCSolveResult(
                solution=solution,
                applied_control=selected,
                has_incumbent=True,
                used_fallback=False,
                fallback_reason=None,
                retry_count=retry,
                candidate_decisions=decisions,
                selected_decision=selected_decision,
            )
        # Defensive retry: a caller can configure a controller whose static
        # candidate set is wider than the gate candidate set.  Never apply the
        # escaped action; add a no-good cut and resolve once more.
        forbidden_by_key.setdefault(control_key(selected), selected)
        forbidden = list(forbidden_by_key.values())
    return GatedMPCSolveResult(
        solution=None,
        applied_control=previous_control,
        has_incumbent=False,
        used_fallback=True,
        fallback_reason='selected_action_failed_gate_after_retry',
        retry_count=retries,
        candidate_decisions=decisions,
        selected_decision=None,
    )


def recorded_action_history(
    data,
    end_exclusive: int,
    lookback_steps: int,
) -> np.ndarray:
    """Return recent recorded controls without crossing a source segment.

    This function is only a bootstrap for the first control decision of a
    closed-loop window or for a row-wise counterfactual diagnostic. Once the
    policy starts, ``run_receding_horizon_policy`` appends its own applied MPC
    actions instead of continuing to read recorded future actions.
    """
    end_exclusive = int(end_exclusive)
    lookback_steps = int(lookback_steps)
    if lookback_steps <= 0 or end_exclusive <= 0:
        return np.empty((0, data.encoder.n_features), dtype=float)
    if end_exclusive > len(data.controls):
        raise IndexError('end_exclusive exceeds available control rows')
    start = max(0, end_exclusive - lookback_steps)
    if data.segment_ids is not None:
        segment_ids = np.asarray(data.segment_ids)
        segment = segment_ids[end_exclusive - 1]
        while start < end_exclusive and segment_ids[start] != segment:
            start += 1
    return np.asarray(data.controls[start:end_exclusive], dtype=float).copy()


class AffinePlantAdapter:
    """Expose controller affine models through the evaluator interface."""

    def __init__(self, state_model, power_model):
        self.state_model = state_model
        self.power_model = power_model
        self.state_names = state_model.state_names
        self.power_names = power_model.power_names

    def predict_state(self, states, controls, disturbances) -> np.ndarray:
        return self.state_model.predict(states, controls, disturbances)

    def predict_power(self, states, controls, disturbances) -> np.ndarray:
        return self.power_model.predict_components(states, controls, disturbances)

    def step(self, state, control, disturbance) -> tuple[np.ndarray, np.ndarray]:
        power = self.predict_power(state, control, disturbance)[0]
        next_state = self.predict_state(state, control, disturbance)[0]
        return next_state, power


def energy_wh(power_components_w: np.ndarray, sample_interval_minutes: int = 1) -> float:
    power_components_w = np.asarray(power_components_w, dtype=float)
    return float(
        power_components_w.sum() * float(sample_interval_minutes) / 60.0
    )


def _plain_specification(raw_specification) -> Dict[str, object]:
    if isinstance(raw_specification, dict):
        return dict(raw_specification)
    return OmegaConf.to_container(raw_specification, resolve=True)


def _violation_values(values: np.ndarray, specification: Mapping) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    violation = np.zeros_like(values)
    lower = specification.get('lower')
    upper = specification.get('upper')
    if lower is not None:
        violation += np.maximum(float(lower) - values, 0.0)
    if upper is not None:
        violation += np.maximum(values - float(upper), 0.0)
    return violation


def constraint_metrics(
    states: np.ndarray,
    specifications: Sequence[Mapping],
    sample_interval_minutes: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Measure constraint severity and duration in physical state units."""
    states = np.asarray(states, dtype=float)
    result = {}
    for position, raw_specification in enumerate(specifications):
        specification = _plain_specification(raw_specification)
        violation = _violation_values(states[:, position], specification)
        violating_steps = int(np.count_nonzero(violation > 1e-8))
        result[specification['name']] = {
            'violation_magnitude_minutes': float(
                violation.sum() * sample_interval_minutes
            ),
            'maximum_violation': float(violation.max(initial=0.0)),
            'violating_steps': violating_steps,
            'violating_minutes': int(violating_steps * sample_interval_minutes),
        }
    return result


def constraint_prediction_metrics(
    actual_states: np.ndarray,
    predicted_states: np.ndarray,
    specifications: Sequence[Mapping],
) -> Dict[str, object]:
    """Report false-safe errors where prediction hides a real violation."""
    actual_states = np.asarray(actual_states, dtype=float)
    predicted_states = np.asarray(predicted_states, dtype=float)
    if actual_states.shape != predicted_states.shape:
        raise ValueError('Actual and predicted state arrays must have the same shape')

    by_state = {}
    total_actual_violations = 0
    total_false_safe = 0
    total_pairs = int(actual_states.shape[0] * actual_states.shape[1])
    rates = []
    for position, raw_specification in enumerate(specifications):
        specification = _plain_specification(raw_specification)
        actual_violation = (
            _violation_values(actual_states[:, position], specification) > 1e-8
        )
        predicted_violation = (
            _violation_values(predicted_states[:, position], specification) > 1e-8
        )
        false_safe = actual_violation & ~predicted_violation
        false_alarm = ~actual_violation & predicted_violation
        actual_count = int(actual_violation.sum())
        false_safe_count = int(false_safe.sum())
        rate = (
            100.0 * false_safe_count / actual_count
            if actual_count > 0
            else None
        )
        if rate is not None:
            rates.append(rate)
        by_state[specification['name']] = {
            'actual_violating_steps': actual_count,
            'predicted_violating_steps': int(predicted_violation.sum()),
            'false_safe_steps': false_safe_count,
            'false_alarm_steps': int(false_alarm.sum()),
            'false_safe_rate_of_actual_violations_percent': rate,
        }
        total_actual_violations += actual_count
        total_false_safe += false_safe_count

    return {
        'definition': (
            'false-safe means actual state violates a configured bound while '
            'the predicted state is inside that bound'
        ),
        'evaluated_state_step_pairs': total_pairs,
        'actual_violation_state_step_pairs': total_actual_violations,
        'false_safe_state_step_pairs': total_false_safe,
        'overall_false_safe_rate_of_actual_violations_percent': (
            100.0 * total_false_safe / total_actual_violations
            if total_actual_violations > 0
            else None
        ),
        'maximum_state_false_safe_rate_percent': (
            float(max(rates)) if rates else None
        ),
        'by_state': by_state,
    }


def _safe_r2_per_output(actual: np.ndarray, predicted: np.ndarray) -> list:
    values = []
    for position in range(actual.shape[1]):
        target = actual[:, position]
        if len(target) < 2 or np.var(target) <= 1e-12:
            values.append(None)
            continue
        residual = np.square(target - predicted[:, position]).sum()
        total = np.square(target - target.mean()).sum()
        values.append(float(1.0 - residual / total))
    return values


def power_prediction_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_names: Sequence[str],
    mape_min_actual_power_w: float,
) -> Dict[str, object]:
    """Power errors with explicit units, near-zero coverage, and name mappings."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim == 1:
        actual = actual[:, None]
        predicted = predicted[:, None]
    if actual.shape != predicted.shape:
        raise ValueError('Actual and predicted power arrays must have the same shape')
    names = tuple(output_names)
    if actual.shape[1] != len(names):
        raise ValueError('Power output_names do not match array width')

    absolute_error = np.abs(predicted - actual)
    mae = absolute_error.mean(axis=0)
    rmse = np.sqrt(np.square(predicted - actual).mean(axis=0))
    denominators = np.abs(actual).sum(axis=0)
    wape = [
        float(100.0 * absolute_error[:, position].sum() / denominators[position])
        if denominators[position] > 1e-12
        else None
        for position in range(actual.shape[1])
    ]
    overall_denominator = float(np.abs(actual).sum())
    mape = power_mape_metrics(
        actual,
        predicted,
        mape_min_actual_power_w,
    )
    mape['mape_percent_by_output'] = dict(
        zip(names, mape['mape_percent_per_output'])
    )
    mape['mape_valid_samples_by_output'] = dict(
        zip(names, mape['mape_valid_samples_per_output'])
    )
    mape['mape_coverage_percent_by_output'] = dict(
        zip(names, mape['mape_coverage_percent_per_output'])
    )

    return {
        'output_names': list(names),
        'sample_count': int(len(actual)),
        'mae_w_per_output': mae.tolist(),
        'mae_w_by_output': dict(zip(names, mae.tolist())),
        'rmse_w_per_output': rmse.tolist(),
        'rmse_w_by_output': dict(zip(names, rmse.tolist())),
        'r2_per_output': _safe_r2_per_output(actual, predicted),
        'wape_percent_per_output': wape,
        'wape_percent_by_output': dict(zip(names, wape)),
        'wape_percent': (
            float(100.0 * absolute_error.sum() / overall_denominator)
            if overall_denominator > 1e-12
            else None
        ),
        **mape,
    }


def _joint_constraint_satisfaction_mask(
    states: np.ndarray,
    specifications: Sequence[Mapping],
) -> np.ndarray:
    """Return rows satisfying every configured state bound."""
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != len(specifications):
        raise ValueError(
            'State samples must be two-dimensional and match state constraints'
        )
    satisfied = np.ones(len(states), dtype=bool)
    for position, raw_specification in enumerate(specifications):
        specification = _plain_specification(raw_specification)
        satisfied &= (
            _violation_values(states[:, position], specification) <= 1e-8
        )
    return satisfied


def _prediction_row_indices(predictions, split_name: str) -> np.ndarray:
    row_indices = getattr(predictions, 'row_indices', None)
    if row_indices is None:
        raise ValueError(
            f'{split_name} evaluator predictions do not expose source row indices'
        )
    return np.asarray(row_indices, dtype=int)


def evaluate_observed_control_combinations(
    data,
    selection_evaluator,
    frozen_report_evaluator,
    *,
    validation_start: int,
    test_start: int,
    test_end: int,
    state_constraints: Sequence[Mapping],
    minimum_validation_samples: int,
    minimum_predicted_state_satisfaction_percent: float,
    minimum_test_exact_matches: int,
    minimum_test_mape_samples: int,
    mape_min_actual_power_w: float,
    selection_evaluator_name: str,
    frozen_report_evaluator_name: str,
) -> tuple[Dict[str, object], list[Dict[str, object]]]:
    """Evaluate one validation-selected observed action against held-out actual Y.

    Only exact joint control combinations observed in validation are candidates.
    A train-only evaluator scores them, and a predeclared test-control overlap
    gate ensures actual Y exists without inspecting test targets. The selected
    combination is evaluated with a train+validation model whose parameters remain
    frozen throughout test evaluation.
    """
    minimum_validation_samples = int(minimum_validation_samples)
    minimum_test_exact_matches = int(minimum_test_exact_matches)
    minimum_test_mape_samples = int(minimum_test_mape_samples)
    minimum_satisfaction = float(
        minimum_predicted_state_satisfaction_percent
    )
    if minimum_validation_samples <= 0:
        raise ValueError('minimum_validation_samples must be positive')
    if minimum_test_exact_matches <= 0:
        raise ValueError('minimum_test_exact_matches must be positive')
    if minimum_test_mape_samples <= 0:
        raise ValueError('minimum_test_mape_samples must be positive')
    if not 0.0 <= minimum_satisfaction <= 100.0:
        raise ValueError(
            'minimum_predicted_state_satisfaction_percent must be in [0, 100]'
        )
    if not validation_start < test_start < test_end:
        raise ValueError(
            'Observed-combination evaluation requires validation before test'
        )

    protocol = {
        'name': 'observed_control_combination_holdout_mape',
        'purpose': (
            'Evaluate the best eligible observed joint categorical control '
            'combination without applying a new action to real equipment'
        ),
        'selection_definition': (
            'Among exact joint control combinations with sufficient validation '
            'support, predicted state-constraint satisfaction, and '
            'preconfigured minimum exact test-control support, choose the '
            'combination with the lowest validation mean predicted total HVAC power'
        ),
        'selection_model': selection_evaluator_name,
        'selection_model_training_scope': 'train only',
        'selection_score_data_scope': 'validation only',
        'overlap_eligibility_data_scope': 'test controls only',
        'report_model': frozen_report_evaluator_name,
        'report_model_training_scope': 'train + validation',
        'report_model_frozen_during_test': True,
        'test_matching': 'exact joint categorical control combination',
        'test_controls_used_only_for_overlap_eligibility': True,
        'test_power_and_state_targets_used_for_selection': False,
        'primary_target': 'total HVAC power = sum of configured power outputs',
        'primary_metric_formula': (
            'MAPE(%) = 100/n * sum(abs(actual_y - predicted_y) / abs(actual_y))'
        ),
        'mape_zero_denominator_policy': (
            'Rows with absolute actual power at or below the configured '
            'minimum are excluded and coverage is reported'
        ),
        'minimum_actual_power_w': float(mape_min_actual_power_w),
        'minimum_validation_samples_per_combination': minimum_validation_samples,
        'minimum_predicted_state_satisfaction_percent': minimum_satisfaction,
        'minimum_test_exact_control_matches': minimum_test_exact_matches,
        'minimum_test_mape_samples_for_reporting': minimum_test_mape_samples,
        'claim_boundary': (
            'This is an observational holdout prediction evaluation, not a '
            'causal estimate of counterfactual MPC savings or real closed-loop control'
        ),
    }

    selection_predictions = selection_evaluator.predict_transitions(
        data,
        int(validation_start),
        int(test_start) - 1,
    )
    selection_rows = _prediction_row_indices(
        selection_predictions,
        'validation',
    )
    selection_controls = np.rint(data.controls[selection_rows]).astype(int)
    predicted_feasible = _joint_constraint_satisfaction_mask(
        selection_predictions.predicted_states,
        state_constraints,
    )
    actual_feasible = _joint_constraint_satisfaction_mask(
        selection_predictions.actual_states,
        state_constraints,
    )
    test_predictions = frozen_report_evaluator.predict_transitions(
        data,
        int(test_start),
        int(test_end),
    )
    test_rows = _prediction_row_indices(test_predictions, 'test')
    test_controls = np.rint(data.controls[test_rows]).astype(int)

    candidates = []
    for encoded_control in np.unique(selection_controls, axis=0):
        combination_mask = np.all(
            selection_controls == encoded_control,
            axis=1,
        )
        sample_count = int(combination_mask.sum())
        predicted_satisfaction_percent = float(
            100.0 * predicted_feasible[combination_mask].mean()
        )
        actual_satisfaction_percent = float(
            100.0 * actual_feasible[combination_mask].mean()
        )
        predicted_power = selection_predictions.predicted_powers[
            combination_mask
        ]
        actual_power = selection_predictions.actual_powers[combination_mask]
        test_support_count = int(
            np.all(test_controls == encoded_control, axis=1).sum()
        )
        eligible = (
            sample_count >= minimum_validation_samples
            and predicted_satisfaction_percent + 1e-12 >= minimum_satisfaction
            and test_support_count >= minimum_test_exact_matches
        )
        candidates.append(
            {
                'encoded_control': encoded_control.astype(int).tolist(),
                'action': data.encoder.decode_action(encoded_control),
                'validation_sample_count': sample_count,
                'test_exact_control_support_count': test_support_count,
                'predicted_state_constraint_satisfaction_percent': (
                    predicted_satisfaction_percent
                ),
                'actual_state_constraint_satisfaction_percent_diagnostic': (
                    actual_satisfaction_percent
                ),
                'predicted_mean_power_w_by_output': dict(
                    zip(
                        data.power_names,
                        predicted_power.mean(axis=0).tolist(),
                    )
                ),
                'predicted_mean_total_power_w': float(
                    predicted_power.sum(axis=1).mean()
                ),
                'actual_mean_total_power_w_diagnostic': float(
                    actual_power.sum(axis=1).mean()
                ),
                'eligible': bool(eligible),
            }
        )

    eligible_candidates = [
        candidate for candidate in candidates if candidate['eligible']
    ]
    selected = (
        min(
            eligible_candidates,
            key=lambda candidate: (
                candidate['predicted_mean_total_power_w'],
                tuple(candidate['encoded_control']),
            ),
        )
        if eligible_candidates
        else None
    )
    for candidate in candidates:
        candidate['selected'] = bool(candidate is selected)
    candidates.sort(
        key=lambda candidate: (
            not candidate['eligible'],
            candidate['predicted_mean_total_power_w'],
            tuple(candidate['encoded_control']),
        )
    )

    selection_summary = {
        'validation_transition_count': int(len(selection_rows)),
        'observed_combination_count': int(len(candidates)),
        'eligible_combination_count': int(len(eligible_candidates)),
        'selected_action': selected['action'] if selected else None,
        'selected_encoded_control': (
            selected['encoded_control'] if selected else None
        ),
        'selected_validation_score': (
            {
                'predicted_mean_total_power_w': selected[
                    'predicted_mean_total_power_w'
                ],
                'predicted_state_constraint_satisfaction_percent': selected[
                    'predicted_state_constraint_satisfaction_percent'
                ],
                'validation_sample_count': selected[
                    'validation_sample_count'
                ],
            }
            if selected
            else None
        ),
        'selected_test_exact_control_support_count': (
            selected['test_exact_control_support_count'] if selected else None
        ),
        'candidates': candidates,
    }
    if selected is None:
        return (
            {
                'available': False,
                'reportable': False,
                'reason': (
                    'No validation control combination met the preconfigured '
                    'validation support, predicted state-satisfaction, and '
                    'test exact-control support requirements'
                ),
                'protocol': protocol,
                'selection': selection_summary,
                'test_evaluation': None,
            },
            [],
        )

    selected_control = np.asarray(
        selected['encoded_control'],
        dtype=int,
    )
    exact_match = np.all(test_controls == selected_control, axis=1)
    match_count = int(exact_match.sum())
    test_coverage_percent = float(
        100.0 * match_count / len(test_rows)
    )
    if match_count == 0:
        return (
            {
                'available': False,
                'reportable': False,
                'reason': (
                    'The validation-selected combination has no exact joint '
                    'control match in the untouched test split'
                ),
                'protocol': protocol,
                'selection': selection_summary,
                'test_evaluation': {
                    'test_transition_count': int(len(test_rows)),
                    'exact_match_count': 0,
                    'exact_match_coverage_percent': test_coverage_percent,
                },
            },
            [],
        )

    actual_power = test_predictions.actual_powers[exact_match]
    predicted_power = test_predictions.predicted_powers[exact_match]
    component_metrics = power_prediction_metrics(
        actual_power,
        predicted_power,
        data.power_names,
        mape_min_actual_power_w,
    )
    total_actual_power = actual_power.sum(axis=1, keepdims=True)
    total_predicted_power = predicted_power.sum(axis=1, keepdims=True)
    total_metrics = power_prediction_metrics(
        total_actual_power,
        total_predicted_power,
        ('총공조유효전력(W)',),
        mape_min_actual_power_w,
    )
    valid_mape_samples = int(
        total_metrics['mape_valid_samples_per_output'][0]
    )
    primary_mape = total_metrics['mape_percent']
    available = primary_mape is not None
    reportable = (
        available and valid_mape_samples >= minimum_test_mape_samples
    )

    matched_actual_states = test_predictions.actual_states[exact_match]
    matched_predicted_states = test_predictions.predicted_states[exact_match]
    actual_state_satisfaction = _joint_constraint_satisfaction_mask(
        matched_actual_states,
        state_constraints,
    )
    predicted_state_satisfaction = _joint_constraint_satisfaction_mask(
        matched_predicted_states,
        state_constraints,
    )
    test_evaluation = {
        'test_transition_count': int(len(test_rows)),
        'exact_match_count': match_count,
        'exact_match_coverage_percent': test_coverage_percent,
        'valid_primary_mape_sample_count': valid_mape_samples,
        'meets_minimum_test_mape_samples': bool(reportable),
        'primary_metric': {
            'name': 'observed_combination_total_power_mape_percent',
            'value': primary_mape,
            'unit': 'percent',
            'reportable': bool(reportable),
        },
        'total_power_prediction': total_metrics,
        'power_component_prediction': component_metrics,
        'actual_joint_state_constraint_satisfaction_percent': float(
            100.0 * actual_state_satisfaction.mean()
        ),
        'predicted_joint_state_constraint_satisfaction_percent': float(
            100.0 * predicted_state_satisfaction.mean()
        ),
        'constraint_prediction': constraint_prediction_metrics(
            matched_actual_states,
            matched_predicted_states,
            state_constraints,
        ),
    }

    audit_rows = []
    matched_rows = test_rows[exact_match]
    for position, row_index in enumerate(matched_rows):
        actual_total = float(total_actual_power[position, 0])
        predicted_total = float(total_predicted_power[position, 0])
        valid_denominator = (
            abs(actual_total) > float(mape_min_actual_power_w)
        )
        row = {
            'row_index': int(row_index),
            'timestamp': data.timestamps.iloc[int(row_index)].isoformat(),
            **{
                f'control_{name}': value
                for name, value in selected['action'].items()
            },
            **{
                f'actual_{name}': float(actual_power[position, output])
                for output, name in enumerate(data.power_names)
            },
            **{
                f'predicted_{name}': float(predicted_power[position, output])
                for output, name in enumerate(data.power_names)
            },
            'actual_total_hvac_power_w': actual_total,
            'predicted_total_hvac_power_w': predicted_total,
            'mape_denominator_valid': bool(valid_denominator),
            'absolute_percentage_error_percent': (
                float(
                    100.0
                    * abs(predicted_total - actual_total)
                    / abs(actual_total)
                )
                if valid_denominator
                else None
            ),
            'actual_state_constraints_satisfied': bool(
                actual_state_satisfaction[position]
            ),
            'predicted_state_constraints_satisfied': bool(
                predicted_state_satisfaction[position]
            ),
        }
        audit_rows.append(row)

    return (
        {
            'available': bool(available),
            'reportable': bool(reportable),
            'reason': (
                None
                if reportable
                else (
                    'A MAPE value was computed, but its valid test sample count '
                    'is below the configured reporting minimum'
                    if available
                    else 'Every matched actual total-power denominator was near zero'
                )
            ),
            'protocol': protocol,
            'selection': selection_summary,
            'test_evaluation': test_evaluation,
        },
        audit_rows,
    )


def state_prediction_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_names: Sequence[str],
) -> Dict[str, object]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim == 1:
        actual = actual[:, None]
        predicted = predicted[:, None]
    if actual.ndim != 2 or actual.shape != predicted.shape:
        raise ValueError(
            'Actual and predicted state arrays must be equal one- or two-dimensional arrays'
        )
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError('State metric arrays must contain only finite values')
    names = tuple(output_names)
    if actual.shape[1] != len(names):
        raise ValueError('State output_names do not match array width')
    absolute_error = np.abs(predicted - actual)
    mae = absolute_error.mean(axis=0)
    rmse = np.sqrt(np.square(predicted - actual).mean(axis=0))
    return {
        'output_names': list(names),
        'sample_count': int(len(actual)),
        'mae_per_output': mae.tolist(),
        'mae_by_output': dict(zip(names, mae.tolist())),
        'rmse_per_output': rmse.tolist(),
        'rmse_by_output': dict(zip(names, rmse.tolist())),
        'r2_per_output': _safe_r2_per_output(actual, predicted),
    }


def rollout_plant(
    plant,
    initial_state: np.ndarray,
    controls: np.ndarray,
    disturbances: np.ndarray,
    *,
    history=None,
    timestamps=None,
) -> PlantRollout:
    controls = np.asarray(controls, dtype=float)
    disturbances = np.asarray(disturbances, dtype=float)
    if len(controls) != len(disturbances):
        raise ValueError('Rollout controls and disturbances must have equal length')
    states = np.empty(
        (len(controls) + 1, len(np.asarray(initial_state).reshape(-1))),
        dtype=float,
    )
    powers = np.empty((len(controls), len(plant.power_names)), dtype=float)
    states[0] = np.asarray(initial_state, dtype=float)
    session = start_rollout_session(plant, history)
    if timestamps is not None and len(timestamps) != len(controls):
        raise ValueError('Rollout timestamps and controls must have equal length')
    for step in range(len(controls)):
        timestamp = None if timestamps is None else timestamps[step]
        states[step + 1], powers[step] = session.step(
            states[step],
            controls[step],
            disturbances[step],
            timestamp=timestamp,
        )
    return PlantRollout(states=states, powers=powers)


def _distribution(values: Sequence[float]) -> Dict[str, object]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'p05': None,
            'p25': None,
            'p75': None,
            'p95': None,
            'minimum': None,
            'maximum': None,
        }
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'p05': float(np.percentile(values, 5)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'p95': float(np.percentile(values, 95)),
        'minimum': float(values.min()),
        'maximum': float(values.max()),
    }


def evaluate_recursive_rollouts(
    plant,
    data,
    starts: Sequence[int],
    horizon: int,
    state_constraints: Sequence[Mapping],
    mape_min_actual_power_w: float,
    lead_time_steps: Sequence[int],
) -> Dict[str, object]:
    """Evaluate recursive prediction under recorded controls and real disturbances."""
    actual_states = []
    predicted_states = []
    actual_powers = []
    predicted_powers = []
    actual_energy = []
    predicted_energy = []
    lead_actual = {
        int(step): [] for step in lead_time_steps if 0 < int(step) <= horizon
    }
    lead_predicted = {step: [] for step in lead_actual}

    for start in starts:
        rollout = rollout_plant(
            plant,
            data.states[start],
            data.controls[start : start + horizon],
            data.disturbances[start : start + horizon],
            history=history_for_rollout(
                data,
                start,
                getattr(plant, 'required_history_steps', 0),
            ),
            timestamps=data.timestamps.iloc[start : start + horizon].to_numpy(),
        )
        actual_state = data.states[start + 1 : start + horizon + 1]
        actual_power = data.powers[start : start + horizon]
        actual_states.append(actual_state)
        predicted_states.append(rollout.states[1:])
        actual_powers.append(actual_power)
        predicted_powers.append(rollout.powers)
        actual_energy.append(energy_wh(actual_power))
        predicted_energy.append(energy_wh(rollout.powers))
        for lead_time in lead_actual:
            lead_actual[lead_time].append(actual_power[lead_time - 1])
            lead_predicted[lead_time].append(rollout.powers[lead_time - 1])

    if not actual_states:
        raise ValueError('At least one recursive rollout window is required')
    actual_states_array = np.concatenate(actual_states)
    predicted_states_array = np.concatenate(predicted_states)
    actual_power_array = np.concatenate(actual_powers)
    predicted_power_array = np.concatenate(predicted_powers)
    actual_total_power_array = actual_power_array.sum(axis=1, keepdims=True)
    predicted_total_power_array = predicted_power_array.sum(axis=1, keepdims=True)
    actual_energy_array = np.asarray(actual_energy, dtype=float)
    predicted_energy_array = np.asarray(predicted_energy, dtype=float)
    valid_energy = actual_energy_array > 1e-12
    window_ape = (
        np.abs(predicted_energy_array[valid_energy] - actual_energy_array[valid_energy])
        / actual_energy_array[valid_energy]
        * 100.0
    )
    total_actual_energy = float(actual_energy_array.sum())
    total_predicted_energy = float(predicted_energy_array.sum())

    return {
        'protocol': (
            'recursive state rollout under recorded controls and actual future '
            'disturbances; no measured state reset inside each window'
        ),
        'window_count': int(len(starts)),
        'horizon_steps': int(horizon),
        'evaluated_steps': int(len(actual_power_array)),
        'state': state_prediction_metrics(
            actual_states_array,
            predicted_states_array,
            data.state_names,
        ),
        'power': power_prediction_metrics(
            actual_power_array,
            predicted_power_array,
            data.power_names,
            mape_min_actual_power_w,
        ),
        'total_power': power_prediction_metrics(
            actual_total_power_array,
            predicted_total_power_array,
            ('total_hvac_power_w',),
            mape_min_actual_power_w,
        ),
        'energy': {
            'actual_total_wh': total_actual_energy,
            'predicted_total_wh': total_predicted_energy,
            'error_wh': total_predicted_energy - total_actual_energy,
            'absolute_percentage_error_percent': (
                100.0
                * abs(total_predicted_energy - total_actual_energy)
                / total_actual_energy
                if total_actual_energy > 1e-12
                else None
            ),
            'window_absolute_percentage_error_percent': _distribution(window_ape),
        },
        'constraint_prediction': constraint_prediction_metrics(
            actual_states_array,
            predicted_states_array,
            state_constraints,
        ),
        'lead_time_power_metrics': {
            str(lead_time): power_prediction_metrics(
                np.asarray(lead_actual[lead_time]),
                np.asarray(lead_predicted[lead_time]),
                data.power_names,
                mape_min_actual_power_w,
            )
            for lead_time in lead_actual
        },
    }


def controller_disturbance_forecast(
    data,
    index: int,
    horizon: int,
    method: str,
) -> np.ndarray:
    if method == 'persistence':
        return np.repeat(data.disturbances[index][None, :], horizon, axis=0)
    if method == 'perfect':
        forecast = data.disturbances[index : index + horizon]
        if len(forecast) != horizon:
            raise ValueError('Perfect disturbance forecast exceeds available data')
        return forecast.copy()
    raise ValueError(
        f"Unknown disturbance forecast {method!r}; use 'perfect' or 'persistence'"
    )


def action_gate_candidate_audit_rows(
    decisions,
    *,
    data,
    timestamp,
    source_row_index: int | None,
    policy_step: int | None,
    evaluation_path: str,
    applied_control: np.ndarray | None = None,
    has_incumbent: bool | None = None,
    fallback_reason: str | None = None,
) -> list[Dict[str, object]]:
    """Flatten action-gate decisions into a durable candidate-level audit.

    ``applied_control`` is recorded on every candidate row because the final
    applied action can be a hold-previous-action fallback rather than a MILP
    incumbent.  Keeping those two facts separate prevents downstream analysis
    from falsely describing a fallback as an MPC-selected action.
    """
    applied_key = (
        None
        if applied_control is None
        else tuple(np.rint(np.asarray(applied_control, dtype=float)).astype(int))
    )
    rows = []
    for decision in decisions:
        trust = getattr(decision, 'trust', None)
        screen = getattr(decision, 'model_screen', None)
        report_action_effect = getattr(decision, 'report_action_effect', None)
        record: Dict[str, object] = {
            'evaluation_path': str(evaluation_path),
            'source_row_index': source_row_index,
            'policy_step': policy_step,
            'timestamp': str(timestamp),
            'encoded_control': list(getattr(decision, 'encoded_control', ())),
            'eligible': bool(getattr(decision, 'eligible', False)),
            'rejection_reasons': list(getattr(decision, 'rejection_reasons', ())),
            'applied_encoded_control': (
                None if applied_key is None else list(applied_key)
            ),
            'selected_by_milp': bool(
                has_incumbent
                and applied_key is not None
                and tuple(getattr(decision, 'encoded_control', ())) == applied_key
            ),
            'used_action_gate_fallback': bool(
                fallback_reason is not None and not bool(has_incumbent)
            ),
            'action_gate_fallback_reason': fallback_reason,
        }
        if trust is not None:
            action = getattr(trust, 'action', None) or {}
            record.update({f'action_{name}': value for name, value in action.items()})
            global_support = getattr(trust, 'global_support', {}) or {}
            local_support = getattr(trust, 'local_support', {}) or {}
            record.update({f'global_{name}': value for name, value in global_support.items()})
            record.update({f'local_{name}': value for name, value in local_support.items()})
            record['trust_rejection_reasons'] = list(
                getattr(trust, 'rejection_reasons', ())
            )
        if screen is not None:
            record['model_screen_rejection_reasons'] = list(
                getattr(screen, 'rejection_reasons', ())
            )
            record['maximum_normalized_component_disagreement'] = getattr(
                screen,
                'maximum_normalized_component_disagreement',
                None,
            )
            affine_power = getattr(screen, 'affine_power_components_w', None)
            evaluator_power = getattr(screen, 'evaluator_power_components_w', None)
            disagreement = getattr(screen, 'normalized_component_disagreement', None)
            if affine_power is not None:
                for position, value in enumerate(affine_power):
                    record[f'affine_{data.power_names[position]}'] = value
            if evaluator_power is not None:
                for position, value in enumerate(evaluator_power):
                    record[f'evaluator_{data.power_names[position]}'] = value
            if disagreement is not None:
                for position, value in enumerate(disagreement):
                    record[f'normalized_disagreement_{data.power_names[position]}'] = value
            record['evaluator_state_safety'] = getattr(screen, 'state_safety', None)
        if report_action_effect is not None:
            record['report_action_effect_rejection_reasons'] = list(
                getattr(report_action_effect, 'rejection_reasons', ())
            )
            record['report_reference_total_power_w'] = getattr(
                report_action_effect,
                'reference_total_power_w',
                None,
            )
            record['report_candidate_total_power_w'] = getattr(
                report_action_effect,
                'candidate_total_power_w',
                None,
            )
            record['report_total_relative_change_percent'] = getattr(
                report_action_effect,
                'total_relative_change_percent',
                None,
            )
            component_change = getattr(
                report_action_effect,
                'component_relative_change_percent',
                None,
            )
            if component_change is not None:
                for position, value in enumerate(component_change):
                    record[
                        'report_component_relative_change_percent_'
                        f'{data.power_names[position]}'
                    ] = value
        rows.append(record)
    return rows


def run_receding_horizon_policy(
    controller,
    evaluator,
    data,
    start: int,
    duration_steps: int,
    disturbance_forecast_method: str,
    training_control_support: np.ndarray,
    *,
    action_gate=None,
    progress_prefix: str = '',
    progress_interval: int = 10,
) -> Dict[str, object]:
    """Re-solve every minute, apply one action, and advance an independent plant."""
    state_count = len(data.state_names)
    power_count = len(data.power_names)
    states = np.empty((duration_steps + 1, state_count), dtype=float)
    powers = np.empty((duration_steps, power_count), dtype=float)
    affine_power = np.empty((duration_steps, power_count), dtype=float)
    controls = np.empty((duration_steps, data.encoder.n_features), dtype=float)
    states[0] = data.states[start]
    previous_control = (
        data.controls[start - 1].copy()
        if start > 0
        else data.encoder.encode_action()
    )
    begin_window = getattr(controller, 'begin_window', None)
    if begin_window is not None:
        begin_window(data, start)
    action_history_enabled = bool(
        getattr(controller, 'action_history_enabled', False)
    )
    action_history_lookback = int(
        getattr(controller, 'action_history_lookback_steps', 0)
    )
    action_history = (
        [
            vector.copy()
            for vector in recorded_action_history(
                data,
                start,
                action_history_lookback,
            )
        ]
        if action_history_enabled
        else None
    )
    support = {
        tuple(np.rint(vector).astype(int).tolist())
        for vector in np.asarray(training_control_support)
    }
    step_records = []
    candidate_diagnostic_rows = []
    evaluator_session = start_rollout_session(
        evaluator,
        history_for_rollout(
            data,
            start,
            getattr(evaluator, 'required_history_steps', 0),
        ),
    )

    for step in range(duration_steps):
        step_started = perf_counter()
        data_index = start + step
        history_before_action = (
            np.asarray(action_history, dtype=float)
            if action_history is not None
            else None
        )
        forecast = controller_disturbance_forecast(
            data,
            data_index,
            controller.prediction_horizon,
            disturbance_forecast_method,
        )
        solve_started = perf_counter()
        gated_solve = solve_first_action_with_gate(
            controller,
            initial_state=states[step],
            disturbances=forecast,
            previous_control=previous_control,
            evaluator=evaluator,
            action_gate=action_gate,
            action_history=history_before_action,
        )
        solve_seconds = perf_counter() - solve_started
        solution = gated_solve.solution
        has_incumbent = bool(gated_solve.has_incumbent)
        used_fallback = bool(gated_solve.used_fallback)
        applied_control = gated_solve.applied_control.copy()
        gate_summary = gated_solve.summary()
        history_diagnostics = (
            controller.action_history_diagnostics(
                applied_control,
                history_before_action,
                previous_control=previous_control,
            )
            if action_history_enabled
            else {'enabled': False}
        )
        candidate_diagnostic_rows.extend(
            action_gate_candidate_audit_rows(
                gated_solve.candidate_decisions,
                data=data,
                timestamp=data.timestamps.iloc[data_index].isoformat(),
                source_row_index=int(data_index),
                policy_step=int(step + 1),
                evaluation_path='closed_loop_mpc',
                applied_control=applied_control,
                has_incumbent=has_incumbent,
                fallback_reason=gated_solve.fallback_reason,
            )
        )
        controls[step] = applied_control
        actual_disturbance = data.disturbances[data_index]
        affine_power[step] = controller.power_model.predict_components(
            states[step],
            applied_control,
            actual_disturbance,
        )[0]
        states[step + 1], powers[step] = evaluator_session.step(
            states[step],
            applied_control,
            actual_disturbance,
            timestamp=data.timestamps.iloc[data_index],
        )
        observe_step = getattr(controller, 'observe_step', None)
        if observe_step is not None:
            observe_step(states[step], applied_control, states[step + 1])
        rounded_control = tuple(np.rint(applied_control).astype(int).tolist())
        is_ood = rounded_control not in support
        action = data.encoder.decode_action(applied_control)
        switched = not np.array_equal(
            np.rint(applied_control).astype(int),
            np.rint(previous_control).astype(int),
        )
        record = {
            'timestamp': data.timestamps.iloc[data_index].isoformat(),
            'step': int(step + 1),
            'solver_success': bool(solution.success) if solution is not None else False,
            'solver_status': int(solution.status) if solution is not None else None,
            'solver_message': (
                solution.message
                if solution is not None
                else str(gated_solve.fallback_reason)
            ),
            'has_incumbent': bool(has_incumbent),
            'used_fallback': bool(used_fallback),
            'solve_seconds': float(solve_seconds),
            'objective': (
                float(solution.objective)
                if has_incumbent and solution is not None
                else None
            ),
            'control_switched': bool(switched),
            'training_support_ood': bool(is_ood),
            'action_history_enabled': bool(history_diagnostics['enabled']),
            'action_history_lookback_steps_used': int(
                history_diagnostics.get('lookback_steps_used', 0)
            ),
            'action_history_weighted_distance': history_diagnostics.get(
                'weighted_distance'
            ),
            'action_history_historical_change_count': history_diagnostics.get(
                'historical_change_count'
            ),
            'action_history_current_change_count': history_diagnostics.get(
                'current_change_count'
            ),
            'action_history_rolling_change_count': history_diagnostics.get(
                'rolling_change_count'
            ),
            'action_history_rolling_change_budget': history_diagnostics.get(
                'rolling_change_budget'
            ),
            'action_history_rolling_budget_overflow': history_diagnostics.get(
                'rolling_budget_overflow'
            ),
            **gate_summary,
            **{f'action_{name}': value for name, value in action.items()},
            **{
                f'state_{name}': float(states[step, position])
                for position, name in enumerate(data.state_names)
            },
            **{
                f'disturbance_{name}': float(actual_disturbance[position])
                for position, name in enumerate(data.disturbance_names)
            },
        }
        record.update({
            f'action_history_distance_{lag}': value
            for lag, value in history_diagnostics.get('distance_by_lag', {}).items()
        })
        if getattr(controller, 'capture_next_states', False):
            record.update({
                f'next_state_{name}': float(states[step+1, position])
                for position, name in enumerate(data.state_names)
            })
        for position, name in enumerate(data.power_names):
            record[f'evaluator_{name}'] = float(powers[step, position])
            record[f'affine_{name}'] = float(affine_power[step, position])
        record['end_to_end_step_seconds'] = float(perf_counter() - step_started)
        step_records.append(record)
        if action_history is not None:
            action_history.append(applied_control.copy())
            del action_history[:-action_history_lookback]
        previous_control = applied_control

        if progress_prefix and (
            step == 0
            or step + 1 == duration_steps
            or (step + 1) % max(1, int(progress_interval)) == 0
        ):
            LOGGER.info(
                '%s: closed-loop step %d/%d | solve_seconds=%.6f | '
                'candidates=%d | eligible=%d | fallback=%s | reason=%s',
                progress_prefix,
                step + 1,
                duration_steps,
                solve_seconds,
                int(gate_summary['candidate_count']),
                int(gate_summary['eligible_candidate_count']),
                used_fallback,
                gated_solve.fallback_reason,
            )

    return {
        'states': states,
        'powers': powers,
        'affine_power_predictions': affine_power,
        'controls': controls,
        'step_records': step_records,
        'candidate_diagnostic_rows': candidate_diagnostic_rows,
        'first_action': data.encoder.decode_action(controls[0]),
    }


def solver_metrics(
    step_records: Sequence[Mapping],
    deadline_seconds: float,
) -> Dict[str, object]:
    solve_seconds = np.asarray(
        [float(record['solve_seconds']) for record in step_records],
        dtype=float,
    )
    solve_count = int(len(step_records))
    candidate_counts = np.asarray(
        [float(record.get('candidate_count', 0)) for record in step_records],
        dtype=float,
    )
    eligible_candidate_counts = np.asarray(
        [
            float(record.get('eligible_candidate_count', 0))
            for record in step_records
        ],
        dtype=float,
    )
    fallback_reasons: Dict[str, int] = {}
    for record in step_records:
        reason = record.get('action_gate_fallback_reason')
        if reason:
            fallback_reasons[str(reason)] = (
                fallback_reasons.get(str(reason), 0) + 1
            )
    history_enabled = any(
        bool(record.get('action_history_enabled', False))
        for record in step_records
    )
    history_weighted_distances = np.asarray([
        float(record['action_history_weighted_distance'])
        for record in step_records
        if record.get('action_history_weighted_distance') is not None
    ], dtype=float)
    history_rolling_counts = np.asarray([
        float(record['action_history_rolling_change_count'])
        for record in step_records
        if record.get('action_history_rolling_change_count') is not None
    ], dtype=float)
    history_budget_overflows = np.asarray([
        float(record['action_history_rolling_budget_overflow'])
        for record in step_records
        if record.get('action_history_rolling_budget_overflow') is not None
    ], dtype=float)
    return {
        'solve_count': solve_count,
        'optimal_success_count': int(
            sum(bool(record['solver_success']) for record in step_records)
        ),
        'incumbent_count': int(
            sum(bool(record['has_incumbent']) for record in step_records)
        ),
        'fallback_count': int(
            sum(bool(record['used_fallback']) for record in step_records)
        ),
        'training_support_ood_count': int(
            sum(bool(record['training_support_ood']) for record in step_records)
        ),
        'control_switch_count': int(
            sum(bool(record['control_switched']) for record in step_records)
        ),
        'action_gate': {
            'candidate_count_per_solve': {
                'mean': (
                    float(candidate_counts.mean()) if solve_count else None
                ),
                'minimum': (
                    int(candidate_counts.min()) if solve_count else None
                ),
                'maximum': (
                    int(candidate_counts.max()) if solve_count else None
                ),
            },
            'eligible_candidate_count_per_solve': {
                'mean': (
                    float(eligible_candidate_counts.mean())
                    if solve_count
                    else None
                ),
                'minimum': (
                    int(eligible_candidate_counts.min()) if solve_count else None
                ),
                'maximum': (
                    int(eligible_candidate_counts.max()) if solve_count else None
                ),
            },
            'fallback_count': int(
                sum(
                    bool(record.get('used_action_gate_fallback', False))
                    for record in step_records
                )
            ),
            'fallback_reasons': fallback_reasons,
        },
        'action_history': {
            'enabled': bool(history_enabled),
            'mean_weighted_distance': (
                float(history_weighted_distances.mean())
                if len(history_weighted_distances)
                else None
            ),
            'mean_rolling_change_count': (
                float(history_rolling_counts.mean())
                if len(history_rolling_counts)
                else None
            ),
            'rolling_budget_overflow_step_count': int(
                (history_budget_overflows > 1e-12).sum()
            ),
            'maximum_rolling_budget_overflow': (
                float(history_budget_overflows.max())
                if len(history_budget_overflows)
                else None
            ),
        },
        'deadline_seconds': float(deadline_seconds),
        'deadline_miss_count': int(np.count_nonzero(solve_seconds > deadline_seconds)),
        'solve_seconds': {
            'mean': float(solve_seconds.mean()) if solve_count else None,
            'median': float(np.median(solve_seconds)) if solve_count else None,
            'p95': float(np.percentile(solve_seconds, 95)) if solve_count else None,
            'maximum': float(solve_seconds.max()) if solve_count else None,
        },
    }


def policy_metrics(
    rollout: PlantRollout,
    state_constraints: Sequence[Mapping],
) -> Dict[str, object]:
    return {
        'energy_wh': energy_wh(rollout.powers),
        'constraint_violations': constraint_metrics(
            rollout.states[1:],
            state_constraints,
        ),
    }


def energy_comparison(
    baseline_energy_wh: float,
    candidate_energy_wh: float,
) -> Dict[str, object]:
    reduction = float(baseline_energy_wh - candidate_energy_wh)
    return {
        'baseline_energy_wh': float(baseline_energy_wh),
        'candidate_energy_wh': float(candidate_energy_wh),
        'energy_reduction_wh': reduction,
        'energy_reduction_percent': (
            100.0 * reduction / baseline_energy_wh
            if baseline_energy_wh > 1e-12
            else None
        ),
    }


def paired_bootstrap_energy_reduction(
    baseline_energy_wh: Sequence[float],
    candidate_energy_wh: Sequence[float],
    samples: int,
    confidence: float,
    seed: int,
) -> Dict[str, object]:
    baseline = np.asarray(baseline_energy_wh, dtype=float)
    candidate = np.asarray(candidate_energy_wh, dtype=float)
    if baseline.shape != candidate.shape or baseline.ndim != 1 or len(baseline) == 0:
        raise ValueError('Bootstrap energies must be equal nonempty vectors')
    if int(samples) <= 0:
        raise ValueError('bootstrap samples must be positive')
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError('bootstrap confidence must be in (0, 1)')

    rng = np.random.default_rng(int(seed))
    estimates = []
    for _ in range(int(samples)):
        indices = rng.integers(0, len(baseline), size=len(baseline))
        denominator = float(baseline[indices].sum())
        if denominator > 1e-12:
            estimates.append(
                100.0
                * float((baseline[indices] - candidate[indices]).sum())
                / denominator
            )
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        'method': 'paired window bootstrap of ratio-of-sums energy reduction',
        'samples_requested': int(samples),
        'samples_valid': int(len(estimates)),
        'confidence_level': float(confidence),
        'lower_percent': (
            float(np.quantile(estimates, alpha)) if estimates else None
        ),
        'upper_percent': (
            float(np.quantile(estimates, 1.0 - alpha)) if estimates else None
        ),
    }


def aggregate_constraint_metrics(
    window_metrics: Sequence[Mapping[str, Mapping[str, float]]],
) -> Dict[str, Dict[str, float]]:
    if not window_metrics:
        return {}
    result = {}
    for state_name in window_metrics[0]:
        entries = [window[state_name] for window in window_metrics]
        result[state_name] = {
            'violation_magnitude_minutes': float(
                sum(entry['violation_magnitude_minutes'] for entry in entries)
            ),
            'maximum_violation': float(
                max(entry['maximum_violation'] for entry in entries)
            ),
            'violating_steps': int(
                sum(entry['violating_steps'] for entry in entries)
            ),
            'violating_minutes': int(
                sum(entry['violating_minutes'] for entry in entries)
            ),
        }
    return result


def reduction_distribution(
    baseline_energy_wh: Sequence[float],
    candidate_energy_wh: Sequence[float],
) -> Dict[str, object]:
    baseline = np.asarray(baseline_energy_wh, dtype=float)
    candidate = np.asarray(candidate_energy_wh, dtype=float)
    valid = baseline > 1e-12
    reductions = (
        (baseline[valid] - candidate[valid]) / baseline[valid] * 100.0
    )
    result = _distribution(reductions)
    result['non_saving_window_count'] = int(np.count_nonzero(reductions <= 0.0))
    return result
