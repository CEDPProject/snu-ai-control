"""Frozen total-power models for chronological test and MPC-action reporting.

This module deliberately does not select HVAC actions. It trains report-model
families on chronological train rows, selects one family on chronological
validation rows, refits it on train + validation, and evaluates every valid
chronological test transition once.
"""

from dataclasses import dataclass
import logging
import math
from time import perf_counter
from typing import Dict, Sequence

import numpy as np
from omegaconf import OmegaConf

from controllers.evaluation import (
    resolve_report_model_refit_rows,
    action_gate_candidate_audit_rows,
    controller_disturbance_forecast,
    power_prediction_metrics,
    recorded_action_history,
    solve_first_action_with_gate,
)
from controllers.fixed_action_power_models import FixedActionTotalPowerModel


LOGGER = logging.getLogger('MPC.total_power')


@dataclass(frozen=True)
class ComponentAffineCalibratedReportModel:
    """Apply a pre-test component-wise affine report-model calibration.

    ``base_model`` is always trained without test rows.  The calibration
    coefficients are fitted on a chronological *inner holdout* of the rows
    available at that stage (train only during validation selection; train +
    validation after the family has been frozen).  This lets factual report
    bias be corrected without consuming a held-out test power target.
    """

    base_model: object
    slopes: np.ndarray
    intercepts: np.ndarray
    calibration_metadata: Dict[str, object]

    @property
    def input_feature_metadata(self) -> Dict[str, object]:
        """Expose the report-model interface expected by existing reporters."""
        metadata = getattr(self.base_model, 'input_feature_metadata', None)
        if not isinstance(metadata, dict):
            raise AttributeError(
                'The wrapped report model does not expose input_feature_metadata'
            )
        return metadata

    @property
    def input_selection(self):
        """Forward the immutable input-bundle descriptor when callers need it."""
        return getattr(self.base_model, 'input_selection')

    @property
    def family(self) -> str:
        """Retain the selected base-family label for generic report consumers."""
        return str(getattr(self.base_model, 'family'))

    def predict_components(self, states, controls, disturbances) -> np.ndarray:
        raw = np.asarray(
            self.base_model.predict_components(states, controls, disturbances),
            dtype=float,
        )
        calibrated = raw * self.slopes[None, :] + self.intercepts[None, :]
        return np.maximum(calibrated, 0.0)

    def predict(self, states, controls, disturbances) -> np.ndarray:
        return self.predict_components(states, controls, disturbances).sum(
            axis=1,
            keepdims=True,
        )

    def to_metadata(self) -> Dict[str, object]:
        # Preserve the FixedActionTotalPowerModel metadata surface (notably
        # ``input_features`` and ``model_family``) because result reporters
        # deliberately inspect it without knowing whether post-calibration was
        # enabled.  The nested base copy keeps the calibration provenance
        # explicit without breaking those existing report contracts.
        base_metadata = self.base_model.to_metadata()
        return {
            **base_metadata,
            'role': 'evaluation-only component-affine calibrated report model',
            'used_by_mpc_or_milp': False,
            'base_model': base_metadata,
            'post_calibration': self.calibration_metadata,
        }


@dataclass(frozen=True)
class FrozenChronologicalTotalPowerModel:
    """One validation-selected report model frozen before chronological test use.

    The model can be used in two distinct, explicitly labelled ways:

    * observed-action MAPE: score it on the recorded test action; and
    * MPC counterfactual report: score its MPC-action prediction against the
      recorded test power baseline.

    The latter is called MAPE by the requested KPI, but its difference contains
    both the estimated control effect and report-model error.  It is therefore
    never substituted for observed-action forecast MAPE.
    """

    model: object
    train_rows: np.ndarray
    validation_rows: np.ndarray
    selected_family: str
    selection_rule: str
    target_mape_percent: float
    validation_results: list[Dict[str, object]]

    def to_metadata(self) -> Dict[str, object]:
        return {
            'selection_rule': self.selection_rule,
            'selected_family': self.selected_family,
            'target_mape_percent': self.target_mape_percent,
            'frozen_before_test': True,
            'model_selection_scope': 'chronological validation only',
            'model_refit_scope': 'chronological train + validation',
            'train_transition_count': int(len(self.train_rows)),
            'validation_transition_count': int(len(self.validation_rows)),
            'validation_results': self.validation_results,
            'model': self.model.to_metadata(),
        }


def _plain(config) -> Dict[str, object]:
    if config is None:
        return {}
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


def _fit_parameters(config) -> Dict[str, object]:
    specification = _plain(config)
    minimum_active_samples = specification.get('hvac_minimum_active_samples')
    return {
        'minimum_action_samples': int(specification['minimum_action_samples']),
        'n_estimators': int(specification['n_estimators']),
        'min_samples_leaf': int(specification['min_samples_leaf']),
        'max_features': float(specification['max_features']),
        'huber_epsilon': float(specification['huber_epsilon']),
        'huber_alpha': float(specification['huber_alpha']),
        'huber_max_iter': int(specification['huber_max_iter']),
        'n_jobs': int(specification['n_jobs']),
        'input_features': specification.get('input_features'),
        # These fields are consumed only by the component_hvac_two_stage
        # report family.  Supplying defaults here keeps legacy report-model
        # configurations valid while allowing the family to be evaluated by
        # the chronological all-data protocol.
        'hvac_low_power_threshold_w': float(
            specification.get('hvac_low_power_threshold_w', 0.1)
        ),
        'hvac_component_name': specification.get('hvac_component_name'),
        'hvac_minimum_active_samples': (
            None
            if minimum_active_samples is None
            # Keep the raw configured value so the model's own strict type
            # validator can reject booleans/non-integral values instead of
            # silently coercing them (for example ``True`` -> ``1``).
            else minimum_active_samples
        ),
    }


def _fit_model(data, rows, family, config, random_state):
    return FixedActionTotalPowerModel.fit(
        data,
        rows,
        family=str(family),
        random_state=int(random_state),
        **_fit_parameters(config),
    )


def _post_calibration_specification(config) -> Dict[str, object]:
    """Validate the optional leakage-safe factual report calibration spec."""
    specification = _plain(config)
    raw = specification.get('post_calibration') or {}
    if not isinstance(raw, dict):
        raise TypeError('total_power_report_model.post_calibration must be a mapping')
    method = str(raw.get('method', 'none'))
    supported = {'none', 'inner_train_component_affine'}
    if method not in supported:
        raise ValueError(
            'Unknown total_power_report_model.post_calibration.method '
            f'{method!r}; supported={sorted(supported)}'
        )
    fraction = float(raw.get('inner_holdout_fraction', 0.15))
    minimum_samples = int(raw.get('minimum_samples_per_component', 30))
    minimum_slope = float(raw.get('minimum_slope', 0.25))
    maximum_slope = float(raw.get('maximum_slope', 4.0))
    if not 0.0 < fraction < 0.5:
        raise ValueError('post_calibration.inner_holdout_fraction must be in (0, 0.5)')
    if minimum_samples <= 0:
        raise ValueError('post_calibration.minimum_samples_per_component must be positive')
    if not 0.0 <= minimum_slope <= maximum_slope:
        raise ValueError(
            'post_calibration slopes must satisfy 0 <= minimum_slope <= maximum_slope'
        )
    return {
        'method': method,
        'inner_holdout_fraction': fraction,
        'minimum_samples_per_component': minimum_samples,
        'minimum_slope': minimum_slope,
        'maximum_slope': maximum_slope,
    }


def _inner_calibration_rows(rows, specification: Dict[str, object]):
    """Split eligible chronological rows into base-fit and calibrator slices."""
    rows = np.asarray(rows, dtype=int)
    if len(rows) < 2:
        return rows, np.empty(0, dtype=int)
    requested = max(
        int(specification['minimum_samples_per_component']),
        int(math.ceil(len(rows) * float(specification['inner_holdout_fraction']))),
    )
    # A calibration slice that consumes nearly all training rows is more
    # harmful than helpful.  The gate then stays uncalibrated and records why.
    count = min(requested, max(0, len(rows) // 3))
    if count < int(specification['minimum_samples_per_component']):
        return rows, np.empty(0, dtype=int)
    return rows[:-count], rows[-count:]


def _component_affine_parameters(actual, predicted, specification):
    """Fit robust, bounded y ~= a * yhat + b one component at a time."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 2:
        raise ValueError('Calibration actual and predicted components must align')
    slopes = np.ones(actual.shape[1], dtype=float)
    intercepts = np.zeros(actual.shape[1], dtype=float)
    details = []
    for output in range(actual.shape[1]):
        x = predicted[:, output]
        y = actual[:, output]
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if len(x) < int(specification['minimum_samples_per_component']):
            details.append({
                'component_index': int(output),
                'fitted': False,
                'reason': 'insufficient_inner_holdout_samples',
                'sample_count': int(len(x)),
                'slope': 1.0,
                'intercept_w': 0.0,
            })
            continue
        centered_x = x - np.median(x)
        centered_y = y - np.median(y)
        denominator = float(np.dot(centered_x, centered_x))
        if denominator <= 1e-12:
            slope = 1.0
            reason = 'constant_prediction_residual_only'
        else:
            slope = float(np.dot(centered_x, centered_y) / denominator)
            reason = 'bounded_robust_linear_median_intercept'
        slope = float(np.clip(
            slope,
            float(specification['minimum_slope']),
            float(specification['maximum_slope']),
        ))
        intercept = float(np.median(y - slope * x))
        slopes[output] = slope
        intercepts[output] = intercept
        details.append({
            'component_index': int(output),
            'fitted': True,
            'reason': reason,
            'sample_count': int(len(x)),
            'slope': slope,
            'intercept_w': intercept,
        })
    return slopes, intercepts, details


def _calibrate_report_model_from_inner_holdout(
    data,
    *,
    available_rows,
    family,
    config,
    random_state,
    final_base_model,
):
    """Return final base model or a wrapper whose parameters never see test Y."""
    specification = _post_calibration_specification(config)
    if specification['method'] == 'none':
        return final_base_model, {
            **specification,
            'enabled': False,
            'status': 'disabled',
        }
    fit_rows, calibration_rows = _inner_calibration_rows(
        data.valid_transition_rows(available_rows),
        specification,
    )
    if not len(calibration_rows) or not len(fit_rows):
        return final_base_model, {
            **specification,
            'enabled': True,
            'status': 'skipped_insufficient_rows',
            'base_fit_transition_count': int(len(fit_rows)),
            'inner_holdout_transition_count': int(len(calibration_rows)),
        }
    preview = _fit_model(
        data,
        fit_rows,
        family,
        config,
        int(random_state) + 37,
    )
    states, controls, disturbances, _, actual = data.model_arrays_for_rows(
        calibration_rows
    )
    predicted = preview.predict_components(states, controls, disturbances)
    slopes, intercepts, details = _component_affine_parameters(
        actual,
        predicted,
        specification,
    )
    metadata = {
        **specification,
        'enabled': True,
        'status': 'fitted_on_chronological_inner_holdout',
        'base_fit_transition_count': int(len(fit_rows)),
        'inner_holdout_transition_count': int(len(calibration_rows)),
        'components': details,
        'power_names': list(data.power_names),
        'test_targets_used': False,
    }
    return ComponentAffineCalibratedReportModel(
        base_model=final_base_model,
        slopes=slopes,
        intercepts=intercepts,
        calibration_metadata=metadata,
    ), metadata


def _evaluate_model(data, rows, model, minimum_actual_power_w):
    rows = data.valid_transition_rows(rows)
    states, controls, disturbances, _, powers = data.model_arrays_for_rows(rows)
    actual_components = np.asarray(powers, dtype=float)
    predicted_components = model.predict_components(
        states,
        controls,
        disturbances,
    )
    actual_total = powers.sum(axis=1, keepdims=True)
    predicted_total = predicted_components.sum(axis=1, keepdims=True)
    total_metrics = power_prediction_metrics(
        actual_total,
        predicted_total,
        ('total_hvac_power_w',),
        float(minimum_actual_power_w),
    )
    component_metrics = power_prediction_metrics(
        actual_components,
        predicted_components,
        data.power_names,
        float(minimum_actual_power_w),
    )
    return (
        rows,
        actual_components,
        predicted_components,
        actual_total,
        predicted_total,
        total_metrics,
        component_metrics,
    )


def _select_result(results, selection_rule: str, target_mape_percent: float):
    eligible = [result for result in results if result['eligible']]
    if not eligible:
        raise ValueError('No all-test total-power model passed validation support')
    passing = [
        result
        for result in eligible
        if float(result['validation_total_power_mape_percent'])
        <= float(target_mape_percent) + 1e-12
    ]
    if selection_rule == 'simplest_within_target' and passing:
        return min(passing, key=lambda item: int(item['priority']))
    if selection_rule not in ('simplest_within_target', 'minimum_validation_mape'):
        raise ValueError(
            'selection_rule must be simplest_within_target or '
            'minimum_validation_mape'
        )
    return min(
        eligible,
        key=lambda item: (
            float(item['validation_total_power_mape_percent']),
            float(item['validation_total_power_wape_percent']),
            float(item['validation_total_power_mae_w']),
            int(item['priority']),
        ),
    )


def fit_frozen_chronological_total_power_model(
    data,
    *,
    training_rows: Sequence[int],
    validation_rows: Sequence[int],
    report_model_config,
    minimum_actual_power_w: float,
    target_mape_percent: float,
    minimum_validation_samples: int,
    random_state: int,
    test_rows: Sequence[int] = None,
    refit_scope: str = 'train_validation',
) -> FrozenChronologicalTotalPowerModel:
    """Select a total-power family on validation, then freeze the refit fit.

    Family selection always uses train(fit)+validation(score) only; test Y never
    influences selection.  The final frozen model is refit on the rows resolved by
    ``refit_scope``:
    - ``train_validation`` (default): train+validation only; test stays held out
      (the only setting under which the test MAPE/WAPE is a valid held-out error).
    - ``train_validation_test``: also include test rows for a deployment instrument;
      the subsequent test error is then leaked, not a generalization estimate.
    """
    train_rows = data.valid_transition_rows(training_rows)
    validation_rows = data.valid_transition_rows(validation_rows)
    if not len(train_rows) or not len(validation_rows):
        raise ValueError(
            'Chronological total-power training and validation rows must be nonempty'
        )
    if np.intersect1d(train_rows, validation_rows).size:
        raise ValueError('Chronological total-power train/validation rows overlap')

    minimum_validation_samples = int(minimum_validation_samples)
    if minimum_validation_samples <= 0:
        raise ValueError('minimum_validation_samples must be positive')

    specification = _plain(report_model_config)
    families = [str(value) for value in specification['candidate_families']]
    if not families:
        raise ValueError('candidate_families must contain at least one family')
    selection_rule = str(
        specification.get('selection_rule', 'simplest_within_target')
    )
    validation_results = []
    for priority, family in enumerate(families):
        try:
            candidate = _fit_model(
                data,
                train_rows,
                family,
                specification,
                int(random_state) + priority,
            )
            candidate, calibration_metadata = (
                _calibrate_report_model_from_inner_holdout(
                    data,
                    available_rows=train_rows,
                    family=family,
                    config=specification,
                    random_state=int(random_state) + priority,
                    final_base_model=candidate,
                )
            )
            _, _, _, _, _, metrics, component_metrics = _evaluate_model(
                data,
                validation_rows,
                candidate,
                minimum_actual_power_w,
            )
            valid_samples = int(metrics['mape_valid_samples_per_output'][0])
            validation_mape = metrics['mape_percent']
            validation_results.append({
                'priority': int(priority),
                'model_family': family,
                'fit_succeeded': True,
                'eligible': bool(
                    validation_mape is not None
                    and valid_samples >= minimum_validation_samples
                ),
                'validation_transition_count': int(len(validation_rows)),
                'validation_valid_mape_sample_count': valid_samples,
                'validation_total_power_mape_percent': validation_mape,
                'validation_total_power_wape_percent': metrics['wape_percent'],
                'validation_total_power_mae_w': metrics['mae_w_per_output'][0],
                'validation_component_power_prediction': component_metrics,
                'post_calibration': calibration_metadata,
                'meets_target': bool(
                    validation_mape is not None
                    and valid_samples >= minimum_validation_samples
                    and float(validation_mape)
                    <= float(target_mape_percent) + 1e-12
                ),
                'failure': None,
            })
        except Exception as error:
            validation_results.append({
                'priority': int(priority),
                'model_family': family,
                'fit_succeeded': False,
                'eligible': False,
                'validation_transition_count': int(len(validation_rows)),
                'validation_valid_mape_sample_count': 0,
                'validation_total_power_mape_percent': None,
                'validation_total_power_wape_percent': None,
                'validation_total_power_mae_w': None,
                'validation_component_power_prediction': None,
                'post_calibration': None,
                'meets_target': False,
                'failure': f'{type(error).__name__}: {error}',
            })

    selected = _select_result(
        validation_results,
        selection_rule,
        target_mape_percent,
    )
    for result in validation_results:
        result['selected'] = bool(result is selected)

    combined_rows = resolve_report_model_refit_rows(
        data,
        train_rows,
        validation_rows,
        test_rows=test_rows,
        scope=refit_scope,
    )
    selected_family = str(selected['model_family'])
    frozen_base_model = _fit_model(
        data,
        combined_rows,
        selected_family,
        specification,
        int(random_state) + 1000,
    )
    frozen_model, frozen_calibration_metadata = (
        _calibrate_report_model_from_inner_holdout(
            data,
            available_rows=combined_rows,
            family=selected_family,
            config=specification,
            random_state=int(random_state) + 1000,
            final_base_model=frozen_base_model,
        )
    )
    selected['final_refit_post_calibration'] = frozen_calibration_metadata
    return FrozenChronologicalTotalPowerModel(
        model=frozen_model,
        train_rows=train_rows,
        validation_rows=validation_rows,
        selected_family=selected_family,
        selection_rule=selection_rule,
        target_mape_percent=float(target_mape_percent),
        validation_results=validation_results,
    )


def evaluate_all_chronological_test_total_power(
    data,
    *,
    training_rows: Sequence[int],
    validation_rows: Sequence[int],
    test_rows: Sequence[int],
    report_model_config,
    minimum_actual_power_w: float,
    target_mape_percent: float,
    minimum_validation_samples: int,
    minimum_test_samples: int,
    random_state: int,
    report_model_refit_scope: str = 'train_validation',
):
    """Evaluate one frozen report model on every chronological test row."""

    test_rows = data.valid_transition_rows(test_rows)
    if not len(test_rows):
        raise ValueError('All-data total-power train/validation/test must be nonempty')
    minimum_test_samples = int(minimum_test_samples)
    if minimum_test_samples <= 0:
        raise ValueError('All-data total-power sample minimums must be positive')
    frozen_selection = fit_frozen_chronological_total_power_model(
        data,
        training_rows=training_rows,
        validation_rows=validation_rows,
        report_model_config=report_model_config,
        minimum_actual_power_w=minimum_actual_power_w,
        target_mape_percent=target_mape_percent,
        minimum_validation_samples=minimum_validation_samples,
        random_state=random_state,
        test_rows=test_rows,
        refit_scope=report_model_refit_scope,
    )
    train_rows = frozen_selection.train_rows
    validation_rows = frozen_selection.validation_rows
    selected_family = frozen_selection.selected_family
    (
        evaluated_test_rows,
        actual_components,
        predicted_components,
        actual_total,
        predicted_total,
        test_metrics,
        component_metrics,
    ) = _evaluate_model(
        data,
        test_rows,
        frozen_selection.model,
        minimum_actual_power_w,
    )
    valid_samples = int(test_metrics['mape_valid_samples_per_output'][0])
    reportable = bool(
        test_metrics['mape_percent'] is not None
        and valid_samples >= minimum_test_samples
    )
    primary_metric = {
        'name': 'all_chronological_test_total_power_mape_percent',
        'value': test_metrics['mape_percent'] if reportable else None,
        'diagnostic_value': test_metrics['mape_percent'],
        'unit': 'percent',
        'reportable': reportable,
        'target_percent': float(target_mape_percent),
        'target_achieved': bool(
            reportable
            and float(test_metrics['mape_percent'])
            <= float(target_mape_percent) + 1e-12
        ),
    }

    audit_rows = []
    for position, row_index in enumerate(evaluated_test_rows):
        actual_value = float(actual_total[position, 0])
        predicted_value = float(predicted_total[position, 0])
        denominator_valid = abs(actual_value) > float(minimum_actual_power_w)
        audit_rows.append({
            'source_row_index': int(row_index),
            'timestamp': data.timestamps.iloc[int(row_index)].isoformat(),
            **{
                f'control_{name}': value
                for name, value in data.encoder.decode_action(
                    data.controls[int(row_index)]
                ).items()
            },
            **{
                f'actual_{name}': float(actual_components[position, output])
                for output, name in enumerate(data.power_names)
            },
            **{
                f'predicted_{name}': float(predicted_components[position, output])
                for output, name in enumerate(data.power_names)
            },
            'actual_total_hvac_power_w': actual_value,
            'predicted_total_hvac_power_w': predicted_value,
            'model_family': selected_family,
            'mape_denominator_valid': bool(denominator_valid),
            'absolute_percentage_error_percent': (
                float(
                    100.0
                    * abs(predicted_value - actual_value)
                    / abs(actual_value)
                )
                if denominator_valid
                else None
            ),
        })

    return {
        'available': test_metrics['mape_percent'] is not None,
        'reportable': reportable,
        'reason': (
            None
            if reportable
            else 'Chronological test valid-MAPE support gate failed'
        ),
        'protocol': {
            'name': 'all_chronological_test_total_power_holdout',
            'row_assignment': 'chronological, mutually exclusive 8:1:1 split',
            'action_filter': 'none; every valid chronological test transition',
            'model_selection_scope': 'chronological validation only',
            'model_refit_scope': 'chronological train + validation',
            'test_targets_used_for_selection': False,
            'target_y': (
                'total HVAC power = ventilation active power + HVAC active power'
            ),
            'claim_boundary': (
                'Future-time total-power prediction accuracy over the observed '
                'test distribution; not causal MPC policy performance'
            ),
        },
        'split': {
            'train_transition_count': int(len(train_rows)),
            'validation_transition_count': int(len(validation_rows)),
            'test_transition_count': int(len(test_rows)),
            'row_overlap_allowed': False,
        },
        'model_selection': frozen_selection.to_metadata(),
        'primary_metric': primary_metric,
        'test_evaluation': {
            'valid_primary_mape_sample_count': valid_samples,
            'minimum_test_mape_samples': minimum_test_samples,
            'valid_mape_coverage_percent': float(
                100.0 * valid_samples / len(evaluated_test_rows)
            ),
            'observed_exact_action_count': int(
                len(np.unique(
                    np.rint(data.controls[evaluated_test_rows]).astype(int),
                    axis=0,
                ))
            ),
            'total_power_prediction': test_metrics,
            'component_power_prediction': component_metrics,
        },
    }, audit_rows


def policy_window_transition_rows(
    data,
    window_starts: Sequence[int],
    duration_steps: int,
) -> np.ndarray:
    """Return unique valid transition rows covered by selected MPC windows."""
    duration = int(duration_steps)
    if duration <= 0:
        raise ValueError('duration_steps must be positive')
    pieces = [
        np.arange(int(start), int(start) + duration, dtype=int)
        for start in window_starts
    ]
    if not pieces:
        raise ValueError('window_starts must contain at least one evaluation window')
    return data.valid_transition_rows(np.unique(np.concatenate(pieces)))


def _component_energy_comparison(
    baseline_components: np.ndarray,
    candidate_components: np.ndarray,
    power_names,
    sample_interval_minutes: int,
) -> Dict[str, Dict[str, float | None]]:
    """성분별 누적 에너지와 기준 대비 차이를 물리 단위로 요약한다."""
    interval = int(sample_interval_minutes)
    result = {}
    for output, name in enumerate(power_names):
        baseline_wh = float(
            baseline_components[:, output].sum() * interval / 60.0
        )
        candidate_wh = float(
            candidate_components[:, output].sum() * interval / 60.0
        )
        difference_wh = baseline_wh - candidate_wh
        result[str(name)] = {
            'baseline_energy_wh': baseline_wh,
            'candidate_energy_wh': candidate_wh,
            'estimated_reduction_wh': difference_wh,
            'estimated_reduction_percent': (
                float(100.0 * difference_wh / baseline_wh)
                if abs(baseline_wh) > 1e-12
                else None
            ),
        }
    return result


def evaluate_mpc_counterfactual_total_power_report(
    data,
    controller,
    report_selection: FrozenChronologicalTotalPowerModel,
    *,
    evaluation_rows: Sequence[int],
    disturbance_forecast_method: str,
    training_control_support: np.ndarray,
    minimum_actual_power_w: float,
    minimum_test_mape_samples: int,
    target_mape_percent: float,
    sample_interval_minutes: int = 1,
    progress_interval: int = 100,
    action_gate=None,
    evaluator=None,
    candidate_diagnostic_rows: list[Dict[str, object]] | None = None,
    first_action_reference: str = 'recorded_previous_action',
) -> tuple[Dict[str, object], list[Dict[str, object]]]:
    """Report two MAPE cases and action-effect estimates for MPC first actions.

    Each row starts from the *recorded* test state and current disturbance.
    The controller chooses a fresh first action for that context.  A report
    model frozen before test predicts total power twice: once for the recorded
    action and once for the MPC action. The output retains two distinct cases:

    * MPC report prediction vs recorded actual power. This is the requested
      recorded-baseline MAPE, but not actual-Y accuracy under the changed action.
    * MPC report prediction vs recorded-action report prediction. This holds
      state, disturbance, and report model fixed and changes only the action.
      It is a model-based action-effect MAPE-like diagnostic, not actual-Y
      forecast accuracy.

    No recorded test target is supplied to the controller or used to select the
    report-model family.
    """
    reference_name = str(first_action_reference)
    supported_references = {
        'recorded_previous_action',
        'recorded_current_action',
    }
    if reference_name not in supported_references:
        raise ValueError(
            'first_action_reference must be one of '
            f'{sorted(supported_references)}, got {reference_name!r}'
        )
    if action_gate is not None and evaluator is None:
        raise ValueError(
            'Counterfactual action gating requires the independent evaluator'
        )
    rows = data.valid_transition_rows(evaluation_rows)
    if not len(rows):
        raise ValueError('No valid transition rows for MPC counterfactual report')
    minimum_test_mape_samples = int(minimum_test_mape_samples)
    if minimum_test_mape_samples <= 0:
        raise ValueError('minimum_test_mape_samples must be positive')
    interval = int(sample_interval_minutes)
    if interval <= 0:
        raise ValueError('sample_interval_minutes must be positive')

    support = {
        tuple(np.rint(vector).astype(int).tolist())
        for vector in np.asarray(training_control_support)
    }
    output_count = int(len(data.power_names))
    recorded_actual_components = np.empty((len(rows), output_count), dtype=float)
    recorded_action_report_components = np.empty(
        (len(rows), output_count), dtype=float
    )
    mpc_action_report_components = np.empty((len(rows), output_count), dtype=float)
    solve_seconds = np.empty(len(rows), dtype=float)
    action_changed = np.empty(len(rows), dtype=bool)
    action_in_training_support = np.empty(len(rows), dtype=bool)
    solver_success = np.empty(len(rows), dtype=bool)
    solver_has_incumbent = np.empty(len(rows), dtype=bool)
    solver_fallback = np.empty(len(rows), dtype=bool)
    changed_variable_count_from_reference = np.empty(len(rows), dtype=int)
    audit_rows = []

    for position, row_index in enumerate(rows):
        row_index = int(row_index)
        prepare_factual = getattr(controller, 'prepare_factual', None)
        if prepare_factual is not None:
            prepare_factual(data, row_index)
        previous_control = (
            data.controls[row_index].copy()
            if reference_name == 'recorded_current_action'
            else (
                data.controls[row_index - 1].copy()
                if row_index > 0
                else data.encoder.encode_action()
            )
        )
        history_enabled = bool(getattr(controller, 'action_history_enabled', False))
        action_history = (
            recorded_action_history(
                data,
                row_index,
                int(getattr(controller, 'action_history_lookback_steps', 0)),
            )
            if history_enabled
            else None
        )
        forecast = controller_disturbance_forecast(
            data,
            row_index,
            controller.prediction_horizon,
            disturbance_forecast_method,
        )
        started = perf_counter()
        gated_solve = solve_first_action_with_gate(
            controller,
            initial_state=data.states[row_index],
            disturbances=forecast,
            previous_control=previous_control,
            evaluator=evaluator,
            action_gate=action_gate,
            action_history=action_history,
        )
        solve_seconds[position] = perf_counter() - started
        solution = gated_solve.solution
        has_incumbent = bool(gated_solve.has_incumbent)
        selected_control = gated_solve.applied_control.copy()
        gate_summary = gated_solve.summary()
        history_diagnostics = (
            controller.action_history_diagnostics(
                selected_control,
                action_history,
                previous_control=previous_control,
            )
            if history_enabled
            else {'enabled': False}
        )
        if candidate_diagnostic_rows is not None:
            candidate_diagnostic_rows.extend(
                action_gate_candidate_audit_rows(
                    gated_solve.candidate_decisions,
                    data=data,
                    timestamp=data.timestamps.iloc[row_index].isoformat(),
                    source_row_index=row_index,
                    policy_step=None,
                    evaluation_path='counterfactual_mpc_action_report',
                    applied_control=selected_control,
                    has_incumbent=has_incumbent,
                    fallback_reason=gated_solve.fallback_reason,
                )
            )
        mpc_prediction = report_selection.model.predict_components(
            data.states[row_index : row_index + 1],
            selected_control[None, :],
            data.disturbances[row_index : row_index + 1],
        )
        recorded_prediction = report_selection.model.predict_components(
            data.states[row_index : row_index + 1],
            data.controls[row_index : row_index + 1],
            data.disturbances[row_index : row_index + 1],
        )
        mpc_prediction = np.asarray(mpc_prediction, dtype=float)
        recorded_prediction = np.asarray(recorded_prediction, dtype=float)
        expected_shape = (1, output_count)
        if (
            mpc_prediction.shape != expected_shape
            or recorded_prediction.shape != expected_shape
        ):
            raise ValueError(
                'Counterfactual report model must return one component vector '
                'per action and row, got shapes '
                f'mpc={mpc_prediction.shape}, recorded={recorded_prediction.shape}'
            )
        recorded_actual_components[position] = data.powers[row_index]
        recorded_action_report_components[position] = recorded_prediction[0]
        mpc_action_report_components[position] = mpc_prediction[0]
        encoded_selected = tuple(np.rint(selected_control).astype(int).tolist())
        action_in_training_support[position] = encoded_selected in support
        action_changed[position] = not np.array_equal(
            np.rint(selected_control).astype(int),
            np.rint(data.controls[row_index]).astype(int),
        )
        changed_variable_count_from_reference[position] = int(sum(
            not np.array_equal(
                np.rint(selected_control[list(group)]).astype(int),
                np.rint(previous_control[list(group)]).astype(int),
            )
            for group in data.encoder.groups.values()
        ))
        solver_success[position] = (
            bool(solution.success) if solution is not None else False
        )
        solver_has_incumbent[position] = bool(has_incumbent)
        solver_fallback[position] = bool(gated_solve.used_fallback)

        recorded_actual_total = float(recorded_actual_components[position].sum())
        recorded_action_report_total = float(
            recorded_action_report_components[position].sum()
        )
        mpc_action_report_total = float(mpc_action_report_components[position].sum())
        actual_baseline_denominator_valid = abs(recorded_actual_total) > float(
            minimum_actual_power_w
        )
        report_baseline_denominator_valid = abs(recorded_action_report_total) > float(
            minimum_actual_power_w
        )
        actual_baseline_signed_difference_w = (
            recorded_actual_total - mpc_action_report_total
        )
        report_to_report_signed_reduction_w = (
            recorded_action_report_total - mpc_action_report_total
        )
        component_audit = {}
        for output, name in enumerate(data.power_names):
            actual_component = float(recorded_actual_components[position, output])
            recorded_component = float(
                recorded_action_report_components[position, output]
            )
            mpc_component = float(mpc_action_report_components[position, output])
            actual_component_valid = abs(actual_component) > float(
                minimum_actual_power_w
            )
            report_component_valid = abs(recorded_component) > float(
                minimum_actual_power_w
            )
            actual_difference = actual_component - mpc_component
            report_difference = recorded_component - mpc_component
            component_audit.update({
                f'recorded_actual_{name}': actual_component,
                f'recorded_action_report_predicted_{name}': recorded_component,
                f'mpc_action_report_predicted_{name}': mpc_component,
                f'recorded_actual_minus_mpc_report_{name}': actual_difference,
                f'recorded_actual_mape_denominator_valid_{name}': (
                    actual_component_valid
                ),
                f'recorded_actual_absolute_percentage_error_percent_{name}': (
                    float(100.0 * abs(actual_difference) / abs(actual_component))
                    if actual_component_valid
                    else None
                ),
                f'recorded_action_report_minus_mpc_report_{name}': (
                    report_difference
                ),
                f'report_to_report_mape_denominator_valid_{name}': (
                    report_component_valid
                ),
                f'report_to_report_absolute_percentage_error_percent_{name}': (
                    float(100.0 * abs(report_difference) / abs(recorded_component))
                    if report_component_valid
                    else None
                ),
            })
        audit_rows.append({
            'source_row_index': row_index,
            'timestamp': data.timestamps.iloc[row_index].isoformat(),
            **{
                f'recorded_control_{name}': value
                for name, value in data.encoder.decode_action(
                    data.controls[row_index]
                ).items()
            },
            **{
                f'mpc_control_{name}': value
                for name, value in data.encoder.decode_action(
                    selected_control
                ).items()
            },
            **component_audit,
            'recorded_total_hvac_power_w': recorded_actual_total,
            'recorded_action_report_predicted_total_hvac_power_w': (
                recorded_action_report_total
            ),
            'mpc_action_report_predicted_total_hvac_power_w': (
                mpc_action_report_total
            ),
            # Deprecated-compatible alias for existing CSV consumers.
            'mpc_counterfactual_predicted_total_hvac_power_w': (
                mpc_action_report_total
            ),
            'recorded_actual_minus_mpc_report_w': (
                actual_baseline_signed_difference_w
            ),
            'recorded_actual_minus_mpc_report_percent': (
                float(
                    100.0
                    * actual_baseline_signed_difference_w
                    / recorded_actual_total
                )
                if actual_baseline_denominator_valid
                else None
            ),
            'recorded_actual_mape_denominator_valid': bool(
                actual_baseline_denominator_valid
            ),
            'recorded_actual_absolute_percentage_error_percent': (
                float(
                    100.0
                    * abs(actual_baseline_signed_difference_w)
                    / abs(recorded_actual_total)
                )
                if actual_baseline_denominator_valid
                else None
            ),
            'recorded_action_report_minus_mpc_report_w': (
                report_to_report_signed_reduction_w
            ),
            'recorded_action_report_minus_mpc_report_percent': (
                float(
                    100.0
                    * report_to_report_signed_reduction_w
                    / recorded_action_report_total
                )
                if report_baseline_denominator_valid
                else None
            ),
            'report_to_report_mape_denominator_valid': bool(
                report_baseline_denominator_valid
            ),
            'report_to_report_absolute_percentage_error_percent': (
                float(
                    100.0
                    * abs(report_to_report_signed_reduction_w)
                    / abs(recorded_action_report_total)
                )
                if report_baseline_denominator_valid
                else None
            ),
            # Deprecated-compatible aliases: recorded actual versus MPC report.
            'estimated_reduction_w': actual_baseline_signed_difference_w,
            'estimated_reduction_percent': (
                float(
                    100.0
                    * actual_baseline_signed_difference_w
                    / recorded_actual_total
                )
                if actual_baseline_denominator_valid
                else None
            ),
            'mape_denominator_valid': bool(actual_baseline_denominator_valid),
            'absolute_percentage_error_percent': (
                float(
                    100.0
                    * abs(actual_baseline_signed_difference_w)
                    / abs(recorded_actual_total)
                )
                if actual_baseline_denominator_valid
                else None
            ),
            'action_changed_from_recorded': bool(action_changed[position]),
            'first_action_reference': reference_name,
            'changed_control_variable_count_from_reference': int(
                changed_variable_count_from_reference[position]
            ),
            'action_history_enabled': bool(history_diagnostics['enabled']),
            'action_history_source': (
                'recorded actions before the evaluated row'
                if history_enabled
                else None
            ),
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
            'mpc_action_in_training_support': bool(
                action_in_training_support[position]
            ),
            'solver_success': (
                bool(solution.success) if solution is not None else False
            ),
            'solver_has_incumbent': bool(has_incumbent),
            'solver_used_recorded_previous_action_fallback': bool(
                gated_solve.used_fallback
            ),
            'solver_status': (
                int(solution.status) if solution is not None else None
            ),
            'solver_message': (
                str(solution.message)
                if solution is not None
                else str(gated_solve.fallback_reason)
            ),
            'solve_seconds': float(solve_seconds[position]),
            'report_model_family': report_selection.selected_family,
            **gate_summary,
            **{
                f'action_history_distance_{lag}': value
                for lag, value in history_diagnostics.get(
                    'distance_by_lag',
                    {},
                ).items()
            },
        })
        if (
            position == 0
            or position + 1 == len(rows)
            or (position + 1) % max(1, int(progress_interval)) == 0
        ):
            LOGGER.info(
                'MPC counterfactual report row %d/%d | solve_seconds=%.6f | '
                'action_changed=%s | candidates=%d | eligible=%d | '
                'fallback=%s | reason=%s',
                position + 1,
                len(rows),
                solve_seconds[position],
                action_changed[position],
                int(gate_summary['candidate_count']),
                int(gate_summary['eligible_candidate_count']),
                not has_incumbent,
                gated_solve.fallback_reason,
            )
    recorded_actual_total = recorded_actual_components.sum(axis=1, keepdims=True)
    recorded_action_report_total = recorded_action_report_components.sum(
        axis=1, keepdims=True
    )
    mpc_action_report_total = mpc_action_report_components.sum(axis=1, keepdims=True)
    recorded_actual_metrics = power_prediction_metrics(
        recorded_actual_total,
        mpc_action_report_total,
        ('total_hvac_power_w',),
        float(minimum_actual_power_w),
    )
    report_to_report_metrics = power_prediction_metrics(
        recorded_action_report_total,
        mpc_action_report_total,
        ('total_hvac_power_w',),
        float(minimum_actual_power_w),
    )
    recorded_actual_component_metrics = power_prediction_metrics(
        recorded_actual_components,
        mpc_action_report_components,
        data.power_names,
        float(minimum_actual_power_w),
    )
    report_to_report_component_metrics = power_prediction_metrics(
        recorded_action_report_components,
        mpc_action_report_components,
        data.power_names,
        float(minimum_actual_power_w),
    )
    actual_valid_mape_samples = int(
        recorded_actual_metrics['mape_valid_samples_per_output'][0]
    )
    report_to_report_valid_mape_samples = int(
        report_to_report_metrics['mape_valid_samples_per_output'][0]
    )
    actual_reportable = bool(
        recorded_actual_metrics['mape_percent'] is not None
        and actual_valid_mape_samples >= minimum_test_mape_samples
    )
    report_to_report_reportable = bool(
        report_to_report_metrics['mape_percent'] is not None
        and report_to_report_valid_mape_samples >= minimum_test_mape_samples
    )
    recorded_actual_energy_wh = float(
        recorded_actual_total.sum() * interval / 60.0
    )
    recorded_action_report_energy_wh = float(
        recorded_action_report_total.sum() * interval / 60.0
    )
    mpc_action_report_energy_wh = float(
        mpc_action_report_total.sum() * interval / 60.0
    )
    actual_baseline_difference_wh = (
        recorded_actual_energy_wh - mpc_action_report_energy_wh
    )
    actual_baseline_difference_percent = (
        float(
            100.0 * actual_baseline_difference_wh / recorded_actual_energy_wh
        )
        if abs(recorded_actual_energy_wh) > 1e-12
        else None
    )
    report_to_report_reduction_wh = (
        recorded_action_report_energy_wh - mpc_action_report_energy_wh
    )
    report_to_report_reduction_percent = (
        float(
            100.0
            * report_to_report_reduction_wh
            / recorded_action_report_energy_wh
        )
        if abs(recorded_action_report_energy_wh) > 1e-12
        else None
    )
    gate_fallback_reasons: Dict[str, int] = {}
    for row in audit_rows:
        reason = row.get('action_gate_fallback_reason')
        if reason:
            gate_fallback_reasons[str(reason)] = (
                gate_fallback_reasons.get(str(reason), 0) + 1
            )
    candidate_counts = np.asarray(
        [float(row.get('candidate_count', 0)) for row in audit_rows],
        dtype=float,
    )
    eligible_candidate_counts = np.asarray(
        [float(row.get('eligible_candidate_count', 0)) for row in audit_rows],
        dtype=float,
    )
    primary_metric = {
        'name': 'mpc_action_vs_recorded_actual_total_power_mape_percent',
        'value': (
            recorded_actual_metrics['mape_percent'] if actual_reportable else None
        ),
        'diagnostic_value': recorded_actual_metrics['mape_percent'],
        'unit': 'percent',
        'reportable': actual_reportable,
        'target_percent': float(target_mape_percent),
        'target_achieved': bool(
            actual_reportable
            and float(recorded_actual_metrics['mape_percent'])
            <= float(target_mape_percent) + 1e-12
        ),
        'definition': (
            'mean(abs(mpc_action_report_prediction - recorded_actual_power) '
            '/ abs(recorded_actual_power)) * 100'
        ),
        'interpretation': (
            'MPC action report prediction versus same-row recorded actual power; '
            'not actual-Y forecast error under the changed action.'
        ),
    }
    report_to_report_metric = {
        'name': 'mpc_action_vs_recorded_action_report_mape_percent',
        'value': (
            report_to_report_metrics['mape_percent']
            if report_to_report_reportable
            else None
        ),
        'diagnostic_value': report_to_report_metrics['mape_percent'],
        'unit': 'percent',
        'reportable': report_to_report_reportable,
        'definition': (
            'mean(abs(mpc_action_report_prediction - '
            'recorded_action_report_prediction) / '
            'abs(recorded_action_report_prediction)) * 100'
        ),
        'interpretation': (
            'Same-model, same-context action-only relative change. This is a '
            'model-based action-effect diagnostic, not actual-Y forecast error.'
        ),
    }
    return {
        'available': bool(
            recorded_actual_metrics['mape_percent'] is not None
            or report_to_report_metrics['mape_percent'] is not None
        ),
        'reportable': actual_reportable,
        'reason': (
            None
            if actual_reportable
            else 'Recorded-actual counterfactual MAPE valid-sample gate failed'
        ),
        'protocol': {
            'name': 'mpc_selected_action_dual_total_power_mape',
            'context_source': 'recorded chronological test state and disturbance',
            'action_source': 'first action from a fresh MILP solve per test row',
            'first_action_reference': reference_name,
            'first_action_reference_interpretation': (
                'the recorded action at the same test row; this is an '
                'offline recorded-action-anchored diagnostic'
                if reference_name == 'recorded_current_action'
                else 'the recorded action from the immediately preceding row'
            ),
            'first_action_eligibility': (
                'same train-support and validation-calibrated independent-model '
                'gate used by the closed-loop MPC path'
                if action_gate is not None
                else 'no additional first-action eligibility gate configured'
            ),
            'report_prediction_inputs': {
                'recorded_action': '(recorded x_t, recorded z_t, recorded d_t)',
                'mpc_action': '(recorded x_t, MPC-selected z_t, recorded d_t)',
            },
            'recorded_actual_baseline': (
                'recorded total HVAC power at the same test row'
            ),
            'test_targets_used_for_controller_or_model_selection': False,
            'claim_boundary': (
                'Recorded-actual comparison is not actual-Y error under the '
                'changed action. Report-to-report comparison is model-based '
                'action effect, not measured energy savings.'
            ),
        },
        'model_selection': report_selection.to_metadata(),
        'primary_metric': primary_metric,
        'mape_cases': {
            'mpc_action_vs_recorded_actual': primary_metric,
            'mpc_action_vs_recorded_action_report': report_to_report_metric,
        },
        'component_mape_cases': {
            'mpc_action_vs_recorded_actual': recorded_actual_component_metrics,
            'mpc_action_vs_recorded_action_report': (
                report_to_report_component_metrics
            ),
        },
        'test_evaluation': {
            'evaluated_transition_count': int(len(rows)),
            'valid_primary_mape_sample_count': actual_valid_mape_samples,
            'minimum_test_mape_samples': minimum_test_mape_samples,
            'valid_mape_coverage_percent': float(
                100.0 * actual_valid_mape_samples / len(rows)
            ),
            'report_to_report_valid_mape_sample_count': (
                report_to_report_valid_mape_samples
            ),
            'report_to_report_valid_mape_coverage_percent': float(
                100.0 * report_to_report_valid_mape_samples / len(rows)
            ),
            'action_changed_count': int(action_changed.sum()),
            'action_changed_percent': float(100.0 * action_changed.mean()),
            'changed_control_variables_from_reference': {
                'mean': float(changed_variable_count_from_reference.mean()),
                'minimum': int(changed_variable_count_from_reference.min()),
                'maximum': int(changed_variable_count_from_reference.max()),
            },
            'mpc_action_in_training_support_count': int(
                action_in_training_support.sum()
            ),
            'solver_success_count': int(solver_success.sum()),
            'solver_incumbent_count': int(solver_has_incumbent.sum()),
            'solver_fallback_count': int(solver_fallback.sum()),
            'mean_solve_seconds': float(solve_seconds.mean()),
            'first_action_gate': {
                'enabled': bool(action_gate is not None),
                'candidate_count_per_row': {
                    'mean': float(candidate_counts.mean()),
                    'minimum': int(candidate_counts.min()),
                    'maximum': int(candidate_counts.max()),
                },
                'eligible_candidate_count_per_row': {
                    'mean': float(eligible_candidate_counts.mean()),
                    'minimum': int(eligible_candidate_counts.min()),
                    'maximum': int(eligible_candidate_counts.max()),
                },
                'fallback_count': int(solver_fallback.sum()),
                'fallback_reasons': gate_fallback_reasons,
            },
            'recorded_actual_vs_mpc_report_total_power_difference': (
                recorded_actual_metrics
            ),
            'recorded_action_report_vs_mpc_report_total_power_difference': (
                report_to_report_metrics
            ),
            'recorded_actual_vs_mpc_report_component_difference': (
                recorded_actual_component_metrics
            ),
            'recorded_action_report_vs_mpc_report_component_difference': (
                report_to_report_component_metrics
            ),
        },
        'estimated_energy_reduction': {
            'recorded_actual_baseline_energy_wh': recorded_actual_energy_wh,
            'mpc_action_report_predicted_energy_wh': mpc_action_report_energy_wh,
            'estimated_difference_wh': actual_baseline_difference_wh,
            'estimated_difference_percent': actual_baseline_difference_percent,
            # Deprecated-compatible aliases.
            'recorded_baseline_energy_wh': recorded_actual_energy_wh,
            'mpc_counterfactual_predicted_energy_wh': mpc_action_report_energy_wh,
            'estimated_reduction_wh': actual_baseline_difference_wh,
            'estimated_reduction_percent': actual_baseline_difference_percent,
            'by_power_output': _component_energy_comparison(
                recorded_actual_components,
                mpc_action_report_components,
                data.power_names,
                interval,
            ),
        },
        'report_to_report_predicted_energy_reduction': {
            'recorded_action_report_predicted_energy_wh': (
                recorded_action_report_energy_wh
            ),
            'mpc_action_report_predicted_energy_wh': mpc_action_report_energy_wh,
            'estimated_reduction_wh': report_to_report_reduction_wh,
            'estimated_reduction_percent': report_to_report_reduction_percent,
            'by_power_output': _component_energy_comparison(
                recorded_action_report_components,
                mpc_action_report_components,
                data.power_names,
                interval,
            ),
            'interpretation': (
                'Same report model and same recorded context; only action changes. '
                'This is model-based predicted reduction, not measured savings.'
            ),
        },
    }, audit_rows
