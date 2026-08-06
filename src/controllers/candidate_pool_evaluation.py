"""Validation-selected multi-action holdout for the official MAPE KPI.

The candidate split is created without power or state targets.  Candidate
eligibility is then decided from train-only model performance on validation
rows.  One report-model family is selected on the pooled validation rows,
refit on train + validation, frozen, and evaluated once on the union of exact
candidate-action test rows.
"""

from typing import Dict

import numpy as np
from omegaconf import OmegaConf

from controllers.action_crossfit import control_episode_ids
from controllers.evaluation import power_prediction_metrics
from controllers.fixed_action_evaluation import FixedActionHoldoutSplit
from controllers.fixed_action_power_models import FixedActionTotalPowerModel


def _plain(config) -> Dict[str, object]:
    if config is None:
        return {}
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


def _exact_action_rows(data, rows, control) -> np.ndarray:
    rows = data.valid_transition_rows(rows)
    encoded = np.rint(np.asarray(control, dtype=float)).astype(int)
    controls = np.rint(data.controls[rows]).astype(int)
    return rows[np.all(controls == encoded, axis=1)]


def _union_action_rows(data, rows, controls) -> np.ndarray:
    pieces = [
        _exact_action_rows(data, rows, control)
        for control in np.asarray(controls, dtype=int)
    ]
    nonempty = [piece for piece in pieces if len(piece)]
    if not nonempty:
        return np.asarray([], dtype=int)
    return np.unique(np.concatenate(nonempty))


def _model_arrays(data, rows):
    states, controls, disturbances, _, powers = data.model_arrays_for_rows(rows)
    return states, controls, disturbances, powers.sum(axis=1, keepdims=True)


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
    states, controls, disturbances, actual_total = _model_arrays(data, rows)
    predicted_total = model.predict(states, controls, disturbances)
    metrics = power_prediction_metrics(
        actual_total,
        predicted_total,
        ('total_hvac_power_w',),
        float(minimum_actual_power_w),
    )
    return actual_total, predicted_total, metrics


def _select_report_model(
    data,
    *,
    training_rows,
    validation_rows,
    selected_controls,
    config,
    minimum_actual_power_w,
    target_mape_percent,
    random_state,
):
    specification = _plain(config)
    families = [str(value) for value in specification['candidate_families']]
    if not families:
        raise ValueError('candidate_families must contain at least one family')
    selection_rule = str(
        specification.get('selection_rule', 'minimum_validation_mape')
    )
    if selection_rule not in (
        'minimum_validation_mape',
        'simplest_within_target',
    ):
        raise ValueError(
            'selection_rule must be minimum_validation_mape or '
            'simplest_within_target'
        )
    exact_validation_rows = _union_action_rows(
        data,
        validation_rows,
        selected_controls,
    )
    minimum_validation_samples = int(
        specification['minimum_validation_mape_samples']
    )
    if len(exact_validation_rows) < minimum_validation_samples:
        raise ValueError(
            'Candidate pool has too few pooled validation rows: '
            f'observed={len(exact_validation_rows)}, '
            f'required={minimum_validation_samples}'
        )

    results = []
    for priority, family in enumerate(families):
        try:
            model = _fit_model(
                data,
                training_rows,
                family,
                specification,
                int(random_state) + priority,
            )
            _, _, metrics = _evaluate_model(
                data,
                exact_validation_rows,
                model,
                minimum_actual_power_w,
            )
            valid_samples = int(metrics['mape_valid_samples_per_output'][0])
            eligible = bool(
                metrics['mape_percent'] is not None
                and valid_samples >= minimum_validation_samples
            )
            result = {
                'priority': int(priority),
                'model_family': family,
                'fit_succeeded': True,
                'eligible': eligible,
                'validation_exact_pool_sample_count': int(
                    len(exact_validation_rows)
                ),
                'validation_valid_mape_sample_count': valid_samples,
                'validation_total_power_mape_percent': metrics['mape_percent'],
                'validation_total_power_wape_percent': metrics['wape_percent'],
                'validation_total_power_mae_w': metrics['mae_w_per_output'][0],
                'meets_target': bool(
                    eligible
                    and float(metrics['mape_percent'])
                    <= float(target_mape_percent) + 1e-12
                ),
                'failure': None,
            }
        except Exception as error:
            result = {
                'priority': int(priority),
                'model_family': family,
                'fit_succeeded': False,
                'eligible': False,
                'validation_exact_pool_sample_count': int(
                    len(exact_validation_rows)
                ),
                'validation_valid_mape_sample_count': 0,
                'validation_total_power_mape_percent': None,
                'validation_total_power_wape_percent': None,
                'validation_total_power_mae_w': None,
                'meets_target': False,
                'failure': f'{type(error).__name__}: {error}',
            }
        results.append(result)

    eligible = [result for result in results if result['eligible']]
    if not eligible:
        raise ValueError('No candidate-pool report model passed validation')
    passing = [result for result in eligible if result['meets_target']]
    if selection_rule == 'simplest_within_target' and passing:
        selected = min(passing, key=lambda item: int(item['priority']))
    else:
        selected = min(
            eligible,
            key=lambda item: (
                float(item['validation_total_power_mape_percent']),
                float(item['validation_total_power_wape_percent']),
                float(item['validation_total_power_mae_w']),
                int(item['priority']),
            ),
        )
    for result in results:
        result['selected'] = bool(result is selected)

    combined_training_rows = np.unique(
        np.concatenate(
            [
                data.valid_transition_rows(training_rows),
                data.valid_transition_rows(validation_rows),
            ]
        )
    )
    selected_family = str(selected['model_family'])
    frozen_model = _fit_model(
        data,
        combined_training_rows,
        selected_family,
        specification,
        int(random_state) + 1000,
    )
    return frozen_model, selected_family, results, {
        'selection_rule': selection_rule,
        'selection_uses_test_targets': False,
        'selection_training_scope': 'episode-grouped train rows only',
        'selection_scoring_scope': (
            'pooled exact candidate-action episode-grouped validation rows only'
        ),
        'validation_exact_pool_sample_count': int(len(exact_validation_rows)),
        'target_mape_percent': float(target_mape_percent),
        'selected_family': selected_family,
        'refit_scope': 'episode-grouped train + validation rows',
        'frozen_before_test': True,
        'candidate_families': families,
        'model': frozen_model.to_metadata(),
    }


def evaluate_fixed_action_candidate_pool_holdout(
    data,
    holdout: FixedActionHoldoutSplit,
    *,
    config,
    total_power_report_config,
    mape_min_actual_power_w: float,
    random_state: int,
) -> tuple[Dict[str, object], list[Dict[str, object]]]:
    """Evaluate a validation-selected pool of exact observed joint actions."""
    specification = _plain(config)
    target_mape_percent = float(specification['target_mape_percent'])
    maximum_validation_mape = float(
        specification['maximum_validation_mape_percent_per_candidate']
    )
    minimum_validation_samples = int(
        specification['minimum_validation_mape_samples_per_candidate']
    )
    minimum_test_samples = int(
        specification['minimum_test_mape_samples_per_candidate']
    )
    minimum_pooled_test_samples = int(
        specification['minimum_pooled_test_mape_samples']
    )
    maximum_candidates = int(specification['maximum_candidates'])
    if not 0.0 <= maximum_validation_mape:
        raise ValueError(
            'maximum_validation_mape_percent_per_candidate must be non-negative'
        )
    if any(
        value <= 0
        for value in (
            minimum_validation_samples,
            minimum_test_samples,
            minimum_pooled_test_samples,
        )
    ):
        raise ValueError('Candidate-pool sample minimums must be positive')
    if maximum_candidates < 0:
        raise ValueError('maximum_candidates must be zero (all) or positive')

    selection_family = str(specification['selection_model_family'])
    selection_model = _fit_model(
        data,
        holdout.train_rows,
        selection_family,
        total_power_report_config,
        int(random_state),
    )
    candidates = []
    for candidate in holdout.metadata['candidates']:
        control = np.asarray(candidate['encoded_control'], dtype=int)
        validation_rows = _exact_action_rows(
            data,
            holdout.validation_rows,
            control,
        )
        _, _, validation_metrics = _evaluate_model(
            data,
            validation_rows,
            selection_model,
            mape_min_actual_power_w,
        )
        valid_samples = int(
            validation_metrics['mape_valid_samples_per_output'][0]
        )
        validation_mape = validation_metrics['mape_percent']
        rejection_reasons = []
        if valid_samples < minimum_validation_samples:
            rejection_reasons.append('insufficient_validation_mape_samples')
        if (
            validation_mape is None
            or float(validation_mape) > maximum_validation_mape + 1e-12
        ):
            rejection_reasons.append('validation_mape_above_pool_threshold')
        test_support = int(candidate['support']['test']['sample_count'])
        if test_support < minimum_test_samples:
            rejection_reasons.append('insufficient_test_control_support')
        candidates.append(
            {
                'rank_by_total_observed_frequency': int(
                    candidate['rank_by_total_observed_frequency']
                ),
                'candidate_id': (
                    f"action_{int(candidate['rank_by_total_observed_frequency']):03d}"
                ),
                'encoded_control': control.tolist(),
                'action': candidate['action'],
                'support': candidate['support'],
                'validation_valid_mape_sample_count': valid_samples,
                'validation_total_power_mape_percent': validation_mape,
                'validation_total_power_wape_percent': validation_metrics[
                    'wape_percent'
                ],
                'eligible_before_pool_limit': not rejection_reasons,
                'selected_for_pool': False,
                'rejection_reasons': rejection_reasons,
            }
        )

    eligible = [
        candidate
        for candidate in candidates
        if candidate['eligible_before_pool_limit']
    ]
    eligible.sort(
        key=lambda candidate: (
            float(candidate['validation_total_power_mape_percent']),
            -int(candidate['support']['test']['sample_count']),
            int(candidate['rank_by_total_observed_frequency']),
        )
    )
    selected_candidates = (
        eligible
        if maximum_candidates == 0
        else eligible[:maximum_candidates]
    )
    selected_ids = {candidate['candidate_id'] for candidate in selected_candidates}
    for candidate in candidates:
        candidate['selected_for_pool'] = candidate['candidate_id'] in selected_ids
        if (
            candidate['eligible_before_pool_limit']
            and not candidate['selected_for_pool']
        ):
            candidate['rejection_reasons'].append('candidate_pool_size_limit')
    candidates.sort(key=lambda item: int(item['rank_by_total_observed_frequency']))

    protocol = {
        'name': 'validation_selected_fixed_action_pool_holdout_mape',
        'purpose': (
            'Measure the official sample-level MAPE on multiple exact observed '
            'joint actions selected without test targets'
        ),
        'candidate_split': holdout.metadata['protocol'],
        'candidate_selection_model': selection_family,
        'candidate_selection_model_training_scope': 'episode-grouped train only',
        'candidate_selection_score_scope': (
            'per-action exact-match episode-grouped validation rows only'
        ),
        'candidate_selection_metric': 'per-action validation total-power MAPE',
        'maximum_validation_mape_percent_per_candidate': (
            maximum_validation_mape
        ),
        'maximum_candidates_zero_means_all_eligible': True,
        'maximum_candidates': maximum_candidates,
        'test_matching': 'union of exact recorded candidate joint actions',
        'test_power_targets_used_for_candidate_or_model_selection': False,
        'primary_metric_formula': (
            'MAPE(%) = 100/N * sum(abs(actual_y - predicted_y) / abs(actual_y)) '
            'over all valid exact candidate-pool test rows'
        ),
        'primary_metric_aggregation': 'sample-level pooled micro MAPE',
        'secondary_metrics': (
            'equal-action macro MAPE, worst-action MAPE, action target pass rate, '
            'and exact-action coverage'
        ),
        'minimum_actual_power_w': float(mape_min_actual_power_w),
        'target_mape_percent': target_mape_percent,
        'claim_boundary': (
            'Multi-action frozen power-prediction accuracy; not causal proof of '
            'counterfactual action optimality or real-equipment MPC savings'
        ),
    }
    selection = {
        'candidate_count': int(len(candidates)),
        'eligible_candidate_count': int(len(eligible)),
        'selected_candidate_count': int(len(selected_candidates)),
        'selected_candidate_ids': [
            candidate['candidate_id'] for candidate in selected_candidates
        ],
        'selected_actions': [
            candidate['action'] for candidate in selected_candidates
        ],
        'selection_uses_test_targets': False,
        'candidates': candidates,
    }
    if not selected_candidates:
        primary_metric = {
            'name': 'fixed_action_candidate_pool_total_power_mape_percent',
            'value': None,
            'unit': 'percent',
            'reportable': False,
            'target_percent': target_mape_percent,
            'target_achieved': False,
        }
        return (
            {
                'available': False,
                'reportable': False,
                'reason': (
                    'No candidate passed the validation MAPE and support gates'
                ),
                'protocol': protocol,
                'selection': selection,
                'total_power_model_selection': None,
                'primary_metric': primary_metric,
                'test_evaluation': None,
            },
            [],
        )

    selected_controls = np.asarray(
        [candidate['encoded_control'] for candidate in selected_candidates],
        dtype=int,
    )
    (
        report_model,
        selected_family,
        model_validation_results,
        model_metadata,
    ) = _select_report_model(
        data,
        training_rows=holdout.train_rows,
        validation_rows=holdout.validation_rows,
        selected_controls=selected_controls,
        config=total_power_report_config,
        minimum_actual_power_w=mape_min_actual_power_w,
        target_mape_percent=target_mape_percent,
        random_state=int(random_state) + 100,
    )
    test_rows = _union_action_rows(
        data,
        holdout.test_rows,
        selected_controls,
    )
    actual_total, predicted_total, pooled_metrics = _evaluate_model(
        data,
        test_rows,
        report_model,
        mape_min_actual_power_w,
    )
    valid_pooled_samples = int(
        pooled_metrics['mape_valid_samples_per_output'][0]
    )
    episode_ids = control_episode_ids(data)
    per_action = []
    audit_rows = []
    test_position = {int(row): position for position, row in enumerate(test_rows)}
    for candidate in selected_candidates:
        candidate_rows = _exact_action_rows(
            data,
            test_rows,
            candidate['encoded_control'],
        )
        positions = np.asarray(
            [test_position[int(row)] for row in candidate_rows],
            dtype=int,
        )
        candidate_actual = actual_total[positions]
        candidate_predicted = predicted_total[positions]
        candidate_metrics = power_prediction_metrics(
            candidate_actual,
            candidate_predicted,
            ('total_hvac_power_w',),
            float(mape_min_actual_power_w),
        )
        valid_samples = int(
            candidate_metrics['mape_valid_samples_per_output'][0]
        )
        candidate_mape = candidate_metrics['mape_percent']
        reportable = bool(
            candidate_mape is not None and valid_samples >= minimum_test_samples
        )
        per_action.append(
            {
                'candidate_id': candidate['candidate_id'],
                'rank_by_total_observed_frequency': candidate[
                    'rank_by_total_observed_frequency'
                ],
                'action': candidate['action'],
                'test_exact_match_count': int(len(candidate_rows)),
                'test_episode_count': int(
                    len(np.unique(episode_ids[candidate_rows]))
                ),
                'valid_mape_sample_count': valid_samples,
                'reportable': reportable,
                'total_power_mape_percent': candidate_mape,
                'target_achieved': bool(
                    reportable
                    and float(candidate_mape) <= target_mape_percent + 1e-12
                ),
                'total_power_prediction': candidate_metrics,
            }
        )
        for position, row_index in zip(positions, candidate_rows):
            actual_value = float(actual_total[position, 0])
            predicted_value = float(predicted_total[position, 0])
            denominator_valid = bool(
                abs(actual_value) > float(mape_min_actual_power_w)
            )
            audit_rows.append(
                {
                    'candidate_id': candidate['candidate_id'],
                    'rank_by_total_observed_frequency': candidate[
                        'rank_by_total_observed_frequency'
                    ],
                    'source_row_index': int(row_index),
                    'timestamp': data.timestamps.iloc[int(row_index)].isoformat(),
                    **{
                        f'control_{name}': value
                        for name, value in candidate['action'].items()
                    },
                    **{
                        f'actual_{name}': float(value)
                        for name, value in zip(
                            data.power_names,
                            data.powers[int(row_index)],
                        )
                    },
                    'actual_total_hvac_power_w': actual_value,
                    'predicted_total_hvac_power_w': predicted_value,
                    'official_total_power_model_family': selected_family,
                    'mape_denominator_valid': denominator_valid,
                    'absolute_percentage_error_percent': (
                        float(
                            100.0
                            * abs(predicted_value - actual_value)
                            / abs(actual_value)
                        )
                        if denominator_valid
                        else None
                    ),
                }
            )

    reportable_actions = [item for item in per_action if item['reportable']]
    all_actions_reportable = len(reportable_actions) == len(per_action)
    pooled_mape = pooled_metrics['mape_percent']
    reportable = bool(
        pooled_mape is not None
        and valid_pooled_samples >= minimum_pooled_test_samples
        and all_actions_reportable
    )
    action_mapes = np.asarray(
        [item['total_power_mape_percent'] for item in reportable_actions],
        dtype=float,
    )
    macro_mape = float(action_mapes.mean()) if len(action_mapes) else None
    worst_mape = float(action_mapes.max()) if len(action_mapes) else None
    action_pass_rate = (
        float(100.0 * np.mean(action_mapes <= target_mape_percent + 1e-12))
        if len(action_mapes)
        else None
    )
    target_achieved = bool(
        reportable
        and float(pooled_mape) <= target_mape_percent + 1e-12
    )
    primary_metric = {
        'name': 'fixed_action_candidate_pool_total_power_mape_percent',
        'value': pooled_mape if reportable else None,
        'unit': 'percent',
        'reportable': reportable,
        'diagnostic_value': pooled_mape,
        'target_percent': target_mape_percent,
        'target_achieved': target_achieved,
    }
    test_evaluation = {
        'test_transition_count': int(len(holdout.test_rows)),
        'pool_exact_match_count': int(len(test_rows)),
        'pool_exact_match_episode_count': int(
            len(np.unique(episode_ids[test_rows]))
        ),
        'pool_exact_match_coverage_percent': float(
            100.0 * len(test_rows) / len(holdout.test_rows)
        ),
        'valid_primary_mape_sample_count': valid_pooled_samples,
        'minimum_pooled_test_mape_samples': minimum_pooled_test_samples,
        'all_selected_actions_reportable': all_actions_reportable,
        'primary_metric': primary_metric,
        'total_power_prediction': pooled_metrics,
        'equal_action_macro_mape_percent': macro_mape,
        'worst_action_mape_percent': worst_mape,
        'action_target_pass_rate_percent': action_pass_rate,
        'per_action': per_action,
    }
    return (
        {
            'available': pooled_mape is not None,
            'reportable': reportable,
            'reason': (
                None
                if reportable
                else 'Pooled or per-action test MAPE support gate failed'
            ),
            'protocol': protocol,
            'split': holdout.metadata,
            'selection': selection,
            'total_power_model_selection': {
                **model_metadata,
                'validation_results': model_validation_results,
                'test_targets_used_to_select_family': False,
            },
            'primary_metric': primary_metric,
            'test_evaluation': test_evaluation,
        },
        audit_rows,
    )


def evaluate_fixed_action_candidate_pool_top_k_sweep(
    data,
    holdout: FixedActionHoldoutSplit,
    *,
    config,
    total_power_report_config,
    mape_min_actual_power_w: float,
    random_state: int,
) -> tuple[Dict[str, object], list[Dict[str, object]]]:
    """Evaluate every cumulative validation-ranked Top-K pool with one model.

    Candidate ranking, report-model family selection, and K selection use only
    episode-grouped training/validation rows.  The selected report model is
    refit once on training + validation, frozen, and reused for the full test
    sensitivity curve.  Test metrics never affect the selected K.
    """
    specification = _plain(config)
    target_mape_percent = float(specification['target_mape_percent'])
    minimum_k = int(specification['minimum_k'])
    maximum_k = int(specification['maximum_k'])
    step = int(specification.get('step', 1))
    k_selection_rule = str(
        specification.get(
            'k_selection_rule',
            'largest_k_within_validation_target',
        )
    )
    fallback_rule = str(
        specification.get(
            'fallback_rule',
            'minimum_validation_mape',
        )
    )
    maximum_validation_mape = float(
        specification['maximum_validation_mape_percent_per_candidate']
    )
    minimum_validation_samples = int(
        specification['minimum_validation_mape_samples_per_candidate']
    )
    minimum_test_samples = int(
        specification['minimum_test_mape_samples_per_candidate']
    )
    minimum_pooled_validation_samples = int(
        specification['minimum_pooled_validation_mape_samples']
    )
    minimum_pooled_test_samples = int(
        specification['minimum_pooled_test_mape_samples']
    )
    if minimum_k <= 0 or maximum_k < minimum_k or step <= 0:
        raise ValueError(
            'Top-K sweep requires 0 < minimum_k <= maximum_k and step > 0'
        )
    if maximum_validation_mape < 0.0:
        raise ValueError(
            'maximum_validation_mape_percent_per_candidate must be non-negative'
        )
    if any(
        value <= 0
        for value in (
            minimum_validation_samples,
            minimum_test_samples,
            minimum_pooled_validation_samples,
            minimum_pooled_test_samples,
        )
    ):
        raise ValueError('Top-K sweep sample minimums must be positive')
    if k_selection_rule != 'largest_k_within_validation_target':
        raise ValueError(
            'k_selection_rule must be largest_k_within_validation_target'
        )
    if fallback_rule != 'minimum_validation_mape':
        raise ValueError('fallback_rule must be minimum_validation_mape')

    selection_family = str(specification['selection_model_family'])
    selection_model = _fit_model(
        data,
        holdout.train_rows,
        selection_family,
        total_power_report_config,
        int(random_state),
    )
    candidates = []
    for candidate in holdout.metadata['candidates']:
        control = np.asarray(candidate['encoded_control'], dtype=int)
        validation_rows = _exact_action_rows(
            data,
            holdout.validation_rows,
            control,
        )
        _, _, validation_metrics = _evaluate_model(
            data,
            validation_rows,
            selection_model,
            mape_min_actual_power_w,
        )
        valid_samples = int(
            validation_metrics['mape_valid_samples_per_output'][0]
        )
        validation_mape = validation_metrics['mape_percent']
        rejection_reasons = []
        if valid_samples < minimum_validation_samples:
            rejection_reasons.append('insufficient_validation_mape_samples')
        if (
            validation_mape is None
            or float(validation_mape) > maximum_validation_mape + 1e-12
        ):
            rejection_reasons.append('validation_mape_above_pool_threshold')
        if int(candidate['support']['test']['sample_count']) < minimum_test_samples:
            rejection_reasons.append('insufficient_test_control_support')
        candidates.append({
            'rank_by_total_observed_frequency': int(
                candidate['rank_by_total_observed_frequency']
            ),
            'candidate_id': (
                f"action_{int(candidate['rank_by_total_observed_frequency']):03d}"
            ),
            'encoded_control': control.tolist(),
            'action': candidate['action'],
            'support': candidate['support'],
            'selection_validation_valid_mape_sample_count': valid_samples,
            'selection_validation_total_power_mape_percent': validation_mape,
            'selection_validation_total_power_wape_percent': (
                validation_metrics['wape_percent']
            ),
            'eligible': not rejection_reasons,
            'rejection_reasons': rejection_reasons,
        })

    eligible = [candidate for candidate in candidates if candidate['eligible']]
    eligible.sort(
        key=lambda candidate: (
            float(candidate['selection_validation_total_power_mape_percent']),
            -int(candidate['support']['test']['sample_count']),
            int(candidate['rank_by_total_observed_frequency']),
        )
    )
    sweep_candidates = eligible[:maximum_k]
    if len(sweep_candidates) < minimum_k:
        return (
            {
                'available': False,
                'reportable': False,
                'reason': (
                    'Too few validation-eligible actions for the configured '
                    'minimum_k'
                ),
                'protocol': {
                    'name': 'validation_selected_fixed_action_top_k_sweep',
                    'test_targets_used_for_model_or_k_selection': False,
                },
                'candidate_count': int(len(candidates)),
                'eligible_candidate_count': int(len(eligible)),
                'sweep_candidate_count': int(len(sweep_candidates)),
                'curve': [],
                'official_selection': None,
                'primary_metric': {
                    'name': 'fixed_action_top_k_sweep_total_power_mape_percent',
                    'value': None,
                    'unit': 'percent',
                    'reportable': False,
                    'target_percent': target_mape_percent,
                    'target_achieved': False,
                },
            },
            [],
        )

    sweep_controls = np.asarray(
        [candidate['encoded_control'] for candidate in sweep_candidates],
        dtype=int,
    )
    (
        frozen_report_model,
        selected_family,
        model_validation_results,
        model_metadata,
    ) = _select_report_model(
        data,
        training_rows=holdout.train_rows,
        validation_rows=holdout.validation_rows,
        selected_controls=sweep_controls,
        config=total_power_report_config,
        minimum_actual_power_w=mape_min_actual_power_w,
        target_mape_percent=target_mape_percent,
        random_state=int(random_state) + 100,
    )
    selected_validation_result = next(
        result for result in model_validation_results if result['selected']
    )
    validation_model = _fit_model(
        data,
        holdout.train_rows,
        selected_family,
        total_power_report_config,
        int(random_state) + 100
        + int(selected_validation_result['priority']),
    )

    episode_ids = control_episode_ids(data)
    candidate_cache = []
    audit_rows = []
    candidate_order = []
    for inclusion_rank, candidate in enumerate(sweep_candidates, start=1):
        control = np.asarray(candidate['encoded_control'], dtype=int)
        validation_rows = _exact_action_rows(
            data,
            holdout.validation_rows,
            control,
        )
        (
            validation_actual,
            validation_predicted,
            validation_metrics,
        ) = _evaluate_model(
            data,
            validation_rows,
            validation_model,
            mape_min_actual_power_w,
        )
        test_rows = _exact_action_rows(
            data,
            holdout.test_rows,
            control,
        )
        test_actual, test_predicted, test_metrics = _evaluate_model(
            data,
            test_rows,
            frozen_report_model,
            mape_min_actual_power_w,
        )
        test_valid_samples = int(
            test_metrics['mape_valid_samples_per_output'][0]
        )
        test_action_reportable = bool(
            test_metrics['mape_percent'] is not None
            and test_valid_samples >= minimum_test_samples
        )
        candidate_cache.append({
            'candidate': candidate,
            'validation_rows': validation_rows,
            'validation_actual': validation_actual,
            'validation_predicted': validation_predicted,
            'test_rows': test_rows,
            'test_actual': test_actual,
            'test_predicted': test_predicted,
            'test_metrics': test_metrics,
            'test_action_reportable': test_action_reportable,
        })
        candidate_order.append({
            'inclusion_rank': int(inclusion_rank),
            **candidate,
            'report_model_validation_mape_percent': (
                validation_metrics['mape_percent']
            ),
            'report_model_validation_valid_sample_count': int(
                validation_metrics['mape_valid_samples_per_output'][0]
            ),
            'test_total_power_mape_percent': test_metrics['mape_percent'],
            'test_valid_mape_sample_count': test_valid_samples,
            'test_action_reportable': test_action_reportable,
        })
        for position, row_index in enumerate(test_rows):
            actual_value = float(test_actual[position, 0])
            predicted_value = float(test_predicted[position, 0])
            denominator_valid = bool(
                abs(actual_value) > float(mape_min_actual_power_w)
            )
            audit_rows.append({
                'inclusion_rank': int(inclusion_rank),
                'candidate_id': candidate['candidate_id'],
                'rank_by_total_observed_frequency': candidate[
                    'rank_by_total_observed_frequency'
                ],
                'source_row_index': int(row_index),
                'timestamp': data.timestamps.iloc[int(row_index)].isoformat(),
                **{
                    f'control_{name}': value
                    for name, value in candidate['action'].items()
                },
                **{
                    f'actual_{name}': float(value)
                    for name, value in zip(
                        data.power_names,
                        data.powers[int(row_index)],
                    )
                },
                'actual_total_hvac_power_w': actual_value,
                'predicted_total_hvac_power_w': predicted_value,
                'official_total_power_model_family': selected_family,
                'mape_denominator_valid': denominator_valid,
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

    maximum_available_k = int(len(candidate_cache))
    k_values = list(range(minimum_k, maximum_available_k + 1, step))
    if maximum_available_k not in k_values:
        k_values.append(maximum_available_k)
    curve = []
    for k in k_values:
        prefix = candidate_cache[:k]
        validation_actual = np.concatenate(
            [item['validation_actual'] for item in prefix],
            axis=0,
        )
        validation_predicted = np.concatenate(
            [item['validation_predicted'] for item in prefix],
            axis=0,
        )
        validation_metrics = power_prediction_metrics(
            validation_actual,
            validation_predicted,
            ('total_hvac_power_w',),
            float(mape_min_actual_power_w),
        )
        validation_valid_samples = int(
            validation_metrics['mape_valid_samples_per_output'][0]
        )
        validation_mape = validation_metrics['mape_percent']
        validation_reportable = bool(
            validation_mape is not None
            and validation_valid_samples >= minimum_pooled_validation_samples
        )

        test_actual = np.concatenate(
            [item['test_actual'] for item in prefix],
            axis=0,
        )
        test_predicted = np.concatenate(
            [item['test_predicted'] for item in prefix],
            axis=0,
        )
        test_metrics = power_prediction_metrics(
            test_actual,
            test_predicted,
            ('total_hvac_power_w',),
            float(mape_min_actual_power_w),
        )
        test_valid_samples = int(
            test_metrics['mape_valid_samples_per_output'][0]
        )
        all_actions_reportable = all(
            item['test_action_reportable'] for item in prefix
        )
        test_mape = test_metrics['mape_percent']
        test_reportable = bool(
            test_mape is not None
            and test_valid_samples >= minimum_pooled_test_samples
            and all_actions_reportable
        )
        action_mapes = np.asarray(
            [
                item['test_metrics']['mape_percent']
                for item in prefix
                if item['test_action_reportable']
            ],
            dtype=float,
        )
        test_rows = np.unique(
            np.concatenate([item['test_rows'] for item in prefix])
        )
        curve.append({
            'k': int(k),
            'added_candidate_id': prefix[-1]['candidate']['candidate_id'],
            'selected_candidate_ids': [
                item['candidate']['candidate_id'] for item in prefix
            ],
            'validation_exact_match_count': int(len(validation_actual)),
            'validation_valid_mape_sample_count': validation_valid_samples,
            'validation_pooled_mape_percent': validation_mape,
            'validation_pooled_wape_percent': validation_metrics[
                'wape_percent'
            ],
            'validation_reportable': validation_reportable,
            'validation_target_achieved': bool(
                validation_reportable
                and float(validation_mape) <= target_mape_percent + 1e-12
            ),
            'test_exact_match_count': int(len(test_actual)),
            'test_exact_match_episode_count': int(
                len(np.unique(episode_ids[test_rows]))
            ),
            'test_exact_match_coverage_percent': float(
                100.0 * len(test_rows) / len(holdout.test_rows)
            ),
            'test_valid_mape_sample_count': test_valid_samples,
            'test_pooled_mape_percent': test_mape,
            'test_pooled_wape_percent': test_metrics['wape_percent'],
            'test_reportable': test_reportable,
            'test_target_achieved': bool(
                test_reportable
                and float(test_mape) <= target_mape_percent + 1e-12
            ),
            'test_equal_action_macro_mape_percent': (
                float(action_mapes.mean()) if len(action_mapes) else None
            ),
            'test_worst_action_mape_percent': (
                float(action_mapes.max()) if len(action_mapes) else None
            ),
            'test_action_target_pass_rate_percent': (
                float(
                    100.0
                    * np.mean(
                        action_mapes <= target_mape_percent + 1e-12
                    )
                )
                if len(action_mapes)
                else None
            ),
            'all_selected_actions_test_reportable': all_actions_reportable,
            'selected_by_validation': False,
            'test_used_for_selection': False,
            'test_interpretation': 'sensitivity_only',
        })

    validation_passing = [
        row for row in curve if row['validation_target_achieved']
    ]
    if validation_passing:
        selected_row = max(validation_passing, key=lambda row: int(row['k']))
        selection_basis = 'largest_k_within_validation_target'
        fallback_used = False
    else:
        validation_reportable_rows = [
            row for row in curve if row['validation_reportable']
        ]
        if not validation_reportable_rows:
            raise ValueError(
                'No Top-K prefix has enough validation samples for K selection'
            )
        selected_row = min(
            validation_reportable_rows,
            key=lambda row: (
                float(row['validation_pooled_mape_percent']),
                -int(row['k']),
            ),
        )
        selection_basis = fallback_rule
        fallback_used = True

    selected_k = int(selected_row['k'])
    for row in curve:
        row['selected_by_validation'] = bool(int(row['k']) == selected_k)
        row['test_interpretation'] = (
            'official_selected_k'
            if row['selected_by_validation']
            else 'sensitivity_only'
        )
    for candidate in candidate_order:
        candidate['included_in_official_selected_k'] = bool(
            int(candidate['inclusion_rank']) <= selected_k
        )
    for row in audit_rows:
        row['included_in_official_selected_k'] = bool(
            int(row['inclusion_rank']) <= selected_k
        )

    selected_test_mape = selected_row['test_pooled_mape_percent']
    selected_test_reportable = bool(selected_row['test_reportable'])
    primary_metric = {
        'name': 'fixed_action_top_k_sweep_total_power_mape_percent',
        'value': (
            selected_test_mape if selected_test_reportable else None
        ),
        'unit': 'percent',
        'reportable': selected_test_reportable,
        'diagnostic_value': selected_test_mape,
        'target_percent': target_mape_percent,
        'target_achieved': bool(
            selected_test_reportable
            and float(selected_test_mape) <= target_mape_percent + 1e-12
        ),
        'selected_k': selected_k,
    }
    selected_candidates = candidate_order[:selected_k]
    protocol = {
        'name': 'validation_selected_fixed_action_top_k_sweep',
        'candidate_split': holdout.metadata['protocol'],
        'candidate_ordering': (
            'train-only selection model per-action validation MAPE, descending '
            'test support, then observed-frequency rank'
        ),
        'report_model_family_selection_scope': (
            'all sweep-candidate episode-grouped validation rows'
        ),
        'report_model_refit_scope': 'episode-grouped train + validation rows',
        'report_model_frozen_before_test': True,
        'same_frozen_report_model_used_for_every_k': True,
        'k_selection_rule': k_selection_rule,
        'fallback_rule': fallback_rule,
        'test_power_targets_used_for_candidate_model_or_k_selection': False,
        'non_selected_k_test_metrics_are_sensitivity_only': True,
        'target_mape_percent': target_mape_percent,
        'minimum_k': minimum_k,
        'maximum_k': maximum_k,
        'step': step,
        'minimum_actual_power_w': float(mape_min_actual_power_w),
        'primary_metric_formula': (
            'MAPE(%) = 100/N * sum(abs(actual_y - predicted_y) / abs(actual_y))'
        ),
        'claim_boundary': (
            'The selected-K test MAPE measures frozen prediction fidelity on '
            'observed exact actions, not causal closed-loop control performance'
        ),
    }
    return (
        {
            'available': True,
            'reportable': selected_test_reportable,
            'reason': (
                None
                if selected_test_reportable
                else 'Selected-K test support gate failed'
            ),
            'protocol': protocol,
            'split': holdout.metadata,
            'candidate_count': int(len(candidates)),
            'eligible_candidate_count': int(len(eligible)),
            'sweep_candidate_count': int(len(sweep_candidates)),
            'candidate_order': candidate_order,
            'total_power_model_selection': {
                **model_metadata,
                'validation_results': model_validation_results,
                'test_targets_used_to_select_family': False,
            },
            'official_selection': {
                'selected_k': selected_k,
                'selection_basis': selection_basis,
                'fallback_used': fallback_used,
                'validation_pooled_mape_percent': selected_row[
                    'validation_pooled_mape_percent'
                ],
                'test_targets_used_for_k_selection': False,
                'selected_candidate_ids': [
                    item['candidate_id'] for item in selected_candidates
                ],
                'selected_actions': [
                    item['action'] for item in selected_candidates
                ],
            },
            'primary_metric': primary_metric,
            'curve': curve,
        },
        audit_rows,
    )
