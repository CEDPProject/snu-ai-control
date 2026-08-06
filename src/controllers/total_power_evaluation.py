"""Frozen total-power evaluation on every chronological test transition.

This module deliberately does not select HVAC actions. It trains report-model
families on chronological train rows, selects one family on chronological
validation rows, refits it on train + validation, and evaluates every valid
chronological test transition once.
"""

from typing import Dict, Sequence

import numpy as np
from omegaconf import OmegaConf

from controllers.evaluation import power_prediction_metrics
from controllers.fixed_action_power_models import FixedActionTotalPowerModel


def _plain(config) -> Dict[str, object]:
    if config is None:
        return {}
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


def _fit_parameters(config) -> Dict[str, object]:
    specification = _plain(config)
    return {
        'minimum_action_samples': int(specification['minimum_action_samples']),
        'n_estimators': int(specification['n_estimators']),
        'min_samples_leaf': int(specification['min_samples_leaf']),
        'max_features': float(specification['max_features']),
        'huber_epsilon': float(specification['huber_epsilon']),
        'huber_alpha': float(specification['huber_alpha']),
        'huber_max_iter': int(specification['huber_max_iter']),
        'n_jobs': int(specification['n_jobs']),
    }


def _fit_model(data, rows, family, config, random_state):
    return FixedActionTotalPowerModel.fit(
        data,
        rows,
        family=str(family),
        random_state=int(random_state),
        **_fit_parameters(config),
    )


def _evaluate_model(data, rows, model, minimum_actual_power_w):
    rows = data.valid_transition_rows(rows)
    states, controls, disturbances, _, powers = data.model_arrays_for_rows(rows)
    actual_total = powers.sum(axis=1, keepdims=True)
    predicted_total = model.predict(states, controls, disturbances)
    metrics = power_prediction_metrics(
        actual_total,
        predicted_total,
        ('total_hvac_power_w',),
        float(minimum_actual_power_w),
    )
    return rows, actual_total, predicted_total, metrics


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
):
    """Evaluate one frozen report model on every chronological test row."""

    train_rows = data.valid_transition_rows(training_rows)
    validation_rows = data.valid_transition_rows(validation_rows)
    test_rows = data.valid_transition_rows(test_rows)
    if not all(len(rows) for rows in (train_rows, validation_rows, test_rows)):
        raise ValueError('All-data total-power train/validation/test must be nonempty')
    if (
        np.intersect1d(train_rows, validation_rows).size
        or np.intersect1d(train_rows, test_rows).size
        or np.intersect1d(validation_rows, test_rows).size
    ):
        raise ValueError('All-data total-power partitions must be row-disjoint')
    minimum_validation_samples = int(minimum_validation_samples)
    minimum_test_samples = int(minimum_test_samples)
    if min(minimum_validation_samples, minimum_test_samples) <= 0:
        raise ValueError('All-data total-power sample minimums must be positive')

    specification = _plain(report_model_config)
    families = [str(value) for value in specification['candidate_families']]
    selection_rule = str(
        specification.get('selection_rule', 'simplest_within_target')
    )
    validation_results = []
    for priority, family in enumerate(families):
        try:
            model = _fit_model(
                data,
                train_rows,
                family,
                specification,
                int(random_state) + priority,
            )
            _, _, _, metrics = _evaluate_model(
                data,
                validation_rows,
                model,
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

    combined_rows = np.unique(np.concatenate([train_rows, validation_rows]))
    selected_family = str(selected['model_family'])
    frozen_model = _fit_model(
        data,
        combined_rows,
        selected_family,
        specification,
        int(random_state) + 1000,
    )
    (
        evaluated_test_rows,
        actual_total,
        predicted_total,
        test_metrics,
    ) = _evaluate_model(
        data,
        test_rows,
        frozen_model,
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
        'model_selection': {
            'selection_rule': selection_rule,
            'selected_family': selected_family,
            'target_mape_percent': float(target_mape_percent),
            'frozen_before_test': True,
            'validation_results': validation_results,
            'model': frozen_model.to_metadata(),
        },
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
        },
    }, audit_rows
