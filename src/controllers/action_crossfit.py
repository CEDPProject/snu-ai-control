"""Leakage-safe actual-Y evaluation for the actions selected by offline MPC."""

from typing import Dict, Mapping, Sequence

import numpy as np

from controllers.evaluation import power_prediction_metrics


def _prediction_rows(predictions, label: str) -> np.ndarray:
    rows = getattr(predictions, 'row_indices', None)
    if rows is None:
        raise ValueError(f'{label} predictions do not expose source row indices')
    rows = np.asarray(rows, dtype=int)
    if rows.ndim != 1:
        raise ValueError(f'{label} row indices must be one-dimensional')
    return rows


def _nearest_context_distance(
    queries: np.ndarray,
    references: np.ndarray,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Nearest standardized RMS distance, computed in bounded-memory chunks."""
    queries = np.asarray(queries, dtype=float)
    references = np.asarray(references, dtype=float)
    if queries.ndim != 2 or references.ndim != 2:
        raise ValueError('Context arrays must be two-dimensional')
    if queries.shape[1] != references.shape[1]:
        raise ValueError('Context arrays must have the same feature width')
    if not len(queries):
        return np.empty(0, dtype=float)
    if not len(references):
        return np.full(len(queries), np.inf, dtype=float)

    nearest = np.full(len(queries), np.inf, dtype=float)
    width = max(1, queries.shape[1])
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        difference = queries[start:stop, None, :] - references[None, :, :]
        distances = np.sqrt(np.square(difference).sum(axis=2) / width)
        nearest[start:stop] = distances.min(axis=1)
    return nearest


def control_episode_ids(data) -> np.ndarray:
    """Label complete contiguous runs of one recorded joint action."""
    controls = np.rint(np.asarray(data.controls, dtype=float)).astype(int)
    episode_ids = np.zeros(len(controls), dtype=int)
    for position in range(1, len(controls)):
        time_is_adjacent = (
            data.timestamps.iloc[position]
            - data.timestamps.iloc[position - 1]
        ).total_seconds() == 60.0
        same_action = np.array_equal(controls[position], controls[position - 1])
        episode_ids[position] = episode_ids[position - 1] + int(
            not (time_is_adjacent and same_action)
        )
    return episode_ids


def align_boundaries_to_control_episodes(
    data,
    nominal_boundaries: Sequence[int],
) -> list[int]:
    """Move chronological split boundaries to the nearest complete-action episode.

    Rows are never reordered or duplicated.  The returned boundaries are strictly
    increasing and lie between complete contiguous joint-control episodes, so one
    episode cannot leak across two adjacent data partitions.
    """
    nominal = [int(value) for value in nominal_boundaries]
    if nominal != sorted(nominal) or len(set(nominal)) != len(nominal):
        raise ValueError('Nominal split boundaries must be strictly increasing')
    if not nominal:
        return []

    episode_ids = control_episode_ids(data)
    candidates = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
    candidates = candidates[(candidates > 0) & (candidates < len(episode_ids) - 1)]
    if len(candidates) < len(nominal):
        raise ValueError('Not enough complete control episodes for requested splits')

    aligned = []
    previous = 0
    for position, target in enumerate(nominal):
        remaining_after = len(nominal) - position - 1
        viable = candidates[candidates > previous]
        selectable_count = len(viable) - remaining_after
        if selectable_count <= 0:
            raise ValueError('Control-episode split boundaries cannot be ordered')
        selectable = viable[:selectable_count]
        selected = int(selectable[np.argmin(np.abs(selectable - target))])
        aligned.append(selected)
        previous = selected
    return aligned


def chronological_action_split_support(
    data,
    train_end_exclusive: int,
    test_start: int,
    *,
    minimum_samples_per_split: int,
    minimum_complete_episodes_per_split: int,
    require_complete_episode_minimum: bool = True,
) -> tuple[Dict[str, object], list[Dict[str, object]]]:
    """Audit joint-action support without changing chronological split semantics."""
    train_end = int(train_end_exclusive)
    test_start = int(test_start)
    minimum_samples = int(minimum_samples_per_split)
    minimum_episodes = int(minimum_complete_episodes_per_split)
    if not 0 < train_end < test_start < len(data.controls):
        raise ValueError('Chronological train/validation/test boundaries are invalid')
    require_episodes = bool(require_complete_episode_minimum)
    if minimum_samples <= 0 or (require_episodes and minimum_episodes <= 0):
        raise ValueError('Enabled action split support minimums must be positive')

    controls = np.rint(np.asarray(data.controls, dtype=float)).astype(int)
    episode_ids = control_episode_ids(data)
    split_ranges = {
        'train': (0, train_end),
        'validation': (train_end, test_start),
        'test': (test_start, len(controls)),
    }
    episode_split_count: Dict[int, int] = {}
    for episode_id in np.unique(episode_ids):
        positions = np.flatnonzero(episode_ids == episode_id)
        memberships = sum(
            bool(np.any((positions >= start) & (positions < end)))
            for start, end in split_ranges.values()
        )
        episode_split_count[int(episode_id)] = int(memberships)
    crossing_episode_ids = {
        episode_id
        for episode_id, memberships in episode_split_count.items()
        if memberships > 1
    }

    actions = []
    audit_rows = []
    for encoded_action in np.unique(controls, axis=0):
        action_positions = np.flatnonzero(
            np.all(controls == encoded_action, axis=1)
        )
        decoded_action = data.encoder.decode_action(encoded_action)
        split_support = {}
        sample_counts = []
        for split_name, (start, end) in split_ranges.items():
            positions = action_positions[
                (action_positions >= start) & (action_positions < end)
            ]
            observed_episodes = set(episode_ids[positions].tolist())
            complete_episodes = observed_episodes - crossing_episode_ids
            complete_positions = positions[
                ~np.isin(episode_ids[positions], list(crossing_episode_ids))
            ]
            support = {
                'sample_count': int(len(positions)),
                'complete_episode_sample_count': int(len(complete_positions)),
                'observed_episode_count': int(len(observed_episodes)),
                'complete_episode_count': int(len(complete_episodes)),
                'meets_minimum': bool(
                    len(complete_positions) >= minimum_samples
                    and (
                        not require_episodes
                        or len(complete_episodes) >= minimum_episodes
                    )
                ),
            }
            split_support[split_name] = support
            sample_counts.append(int(len(positions)))

        present_in_all = all(count > 0 for count in sample_counts)
        eligible_in_all = all(
            support['meets_minimum'] for support in split_support.values()
        )
        positive_counts = [count for count in sample_counts if count > 0]
        minimum_to_maximum_ratio = (
            float(min(positive_counts) / max(positive_counts))
            if len(positive_counts) == len(split_ranges)
            else 0.0
        )
        action_result = {
            'encoded_control': encoded_action.astype(int).tolist(),
            'action': decoded_action,
            'present_in_all_splits': bool(present_in_all),
            'eligible_in_all_splits': bool(eligible_in_all),
            'minimum_to_maximum_sample_count_ratio': minimum_to_maximum_ratio,
            'splits': split_support,
        }
        actions.append(action_result)
        audit_rows.append(
            {
                **{
                    f'control_{name}': value
                    for name, value in decoded_action.items()
                },
                **{
                    f'{split_name}_{metric}': value
                    for split_name, support in split_support.items()
                    for metric, value in support.items()
                },
                'present_in_all_splits': bool(present_in_all),
                'eligible_in_all_splits': bool(eligible_in_all),
                'minimum_to_maximum_sample_count_ratio': (
                    minimum_to_maximum_ratio
                ),
            }
        )

    actions.sort(key=lambda item: tuple(item['encoded_control']))
    audit_rows.sort(
        key=lambda row: tuple(
            str(row[f'control_{variable.name}'])
            for variable in data.encoder.variables
        )
    )
    present_count = sum(item['present_in_all_splits'] for item in actions)
    eligible_count = sum(item['eligible_in_all_splits'] for item in actions)
    split_observed_counts = {
        split_name: int(
            sum(item['splits'][split_name]['sample_count'] > 0 for item in actions)
        )
        for split_name in split_ranges
    }
    return (
        {
            'protocol': {
                'name': 'chronological_joint_action_split_support_audit',
                'row_assignment': (
                    'chronological, mutually exclusive, without resampling or '
                    'duplication'
                ),
                'episode_definition': (
                    'maximal one-minute contiguous run of one exact joint action'
                ),
                'eligibility_rule': (
                    'minimum samples from complete episodes and minimum complete '
                    'independent episodes must be met in train, validation, and test'
                ),
                'minimum_samples_per_split': minimum_samples,
                'minimum_complete_episodes_per_split': minimum_episodes,
                'complete_episode_minimum_enabled': require_episodes,
            },
            'boundaries': {
                'train_end_exclusive': train_end,
                'validation_start': train_end,
                'test_start': test_start,
                'test_end_exclusive': int(len(controls)),
            },
            'control_episode_count': int(len(np.unique(episode_ids))),
            'cross_split_episode_count': int(len(crossing_episode_ids)),
            'episode_boundaries_are_disjoint': not crossing_episode_ids,
            'observed_action_count': int(len(actions)),
            'observed_action_count_by_split': split_observed_counts,
            'present_in_all_splits_action_count': int(present_count),
            'eligible_in_all_splits_action_count': int(eligible_count),
            'eligible_in_all_splits_action_percent': float(
                100.0 * eligible_count / len(actions) if actions else 0.0
            ),
            'balanced_holdout_available': bool(eligible_count > 0),
            'interpretation': (
                'This audit measures support; it does not force naturally sparse '
                'joint actions to become balanced or leak future rows into training.'
            ),
            'actions': actions,
        },
        audit_rows,
    )


def _weighted_defined(
    values: Sequence[float | None],
    weights: Sequence[int],
) -> float | None:
    defined = [
        (float(value), int(weight))
        for value, weight in zip(values, weights)
        if value is not None and int(weight) > 0
    ]
    if not defined:
        return None
    denominator = sum(weight for _, weight in defined)
    return float(sum(value * weight for value, weight in defined) / denominator)


def _finite_distance_summary(distances: np.ndarray) -> Dict[str, float | None]:
    finite = np.asarray(distances, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {'minimum': None, 'median': None, 'maximum': None}
    return {
        'minimum': float(finite.min()),
        'median': float(np.median(finite)),
        'maximum': float(finite.max()),
    }


def _validate_mpc_arrays(
    data,
    controls: np.ndarray,
    states: np.ndarray,
    disturbances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    controls = np.rint(np.asarray(controls, dtype=float)).astype(int)
    states = np.asarray(states, dtype=float)
    disturbances = np.asarray(disturbances, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != data.encoder.n_features:
        raise ValueError('MPC controls do not match the configured encoder')
    if states.shape != (len(controls), len(data.state_names)):
        raise ValueError('MPC states do not match the selected control steps')
    if disturbances.shape != (len(controls), len(data.disturbance_names)):
        raise ValueError('MPC disturbances do not match the selected control steps')
    if not len(controls):
        raise ValueError('At least one MPC control step is required')
    if not (np.isfinite(states).all() and np.isfinite(disturbances).all()):
        raise ValueError('MPC context must contain only finite values')
    return controls, states, disturbances


def _collect_crossfit_predictions(data, folds, evaluator_family: str):
    """Validate, concatenate, and chronologically order out-of-fold predictions."""
    row_parts = []
    actual_parts = []
    predicted_parts = []
    fold_number_parts = []
    training_end_parts = []
    fold_metadata = []
    seen_rows = set()

    for fallback_number, fold in enumerate(folds, start=1):
        fold_number = int(fold.get('fold_number', fallback_number))
        training_end = int(fold['training_end_exclusive'])
        evaluation_start = int(fold['evaluation_start'])
        evaluation_end = int(fold['evaluation_end_exclusive'])
        if training_end > evaluation_start:
            raise ValueError('Cross-fit training must end before evaluation')

        predictions = fold['predictions']
        rows = _prediction_rows(predictions, f'cross-fit fold {fold_number}')
        if ((rows < evaluation_start) | (rows >= evaluation_end)).any():
            raise ValueError('Cross-fit prediction row lies outside its fold')
        if seen_rows.intersection(rows.tolist()):
            raise ValueError('Cross-fit folds contain duplicate evaluation rows')
        seen_rows.update(rows.tolist())

        actual = np.asarray(predictions.actual_powers, dtype=float)
        predicted = np.asarray(predictions.predicted_powers, dtype=float)
        if actual.shape != predicted.shape or len(actual) != len(rows):
            raise ValueError('Cross-fit power arrays and source rows must match')
        if not np.allclose(actual, data.powers[rows]):
            raise ValueError('Cross-fit actual Y does not match its source rows')

        row_parts.append(rows)
        actual_parts.append(actual)
        predicted_parts.append(predicted)
        fold_number_parts.append(np.full(len(rows), fold_number, dtype=int))
        training_end_parts.append(np.full(len(rows), training_end, dtype=int))
        fold_metadata.append(
            {
                'fold_number': fold_number,
                'training_end_exclusive': training_end,
                'evaluation_start': evaluation_start,
                'evaluation_end_exclusive': evaluation_end,
                'training_scope': f'[0, {training_end})',
                'evaluation_scope': f'[{evaluation_start}, {evaluation_end})',
                'valid_transition_count': int(len(rows)),
                'evaluator': str(fold.get('evaluator_name', evaluator_family)),
            }
        )

    if not row_parts:
        raise ValueError('At least one cross-fit fold is required')
    rows = np.concatenate(row_parts)
    order = np.argsort(rows)
    return (
        rows[order],
        np.concatenate(actual_parts)[order],
        np.concatenate(predicted_parts)[order],
        np.concatenate(fold_number_parts)[order],
        np.concatenate(training_end_parts)[order],
        fold_metadata,
    )


def evaluate_mpc_selected_action_crossfit(
    data,
    mpc_controls: np.ndarray,
    mpc_states: np.ndarray,
    mpc_disturbances: np.ndarray,
    crossfit_folds: Sequence[Mapping[str, object]],
    *,
    initial_training_end_exclusive: int,
    context_distance_threshold: float,
    minimum_matched_samples_per_action: int,
    minimum_independent_episodes_per_action: int,
    minimum_fold_training_samples_per_action: int,
    minimum_fold_training_episodes_per_action: int,
    minimum_reportable_mpc_step_coverage_percent: float,
    mape_min_actual_power_w: float,
    evaluator_family: str,
    exclude_insufficient_episode_actions: bool = True,
) -> tuple[Dict[str, object], list[Dict[str, object]]]:
    """Measure actual-Y prediction fidelity for joint actions chosen by MPC.

    Historical predictions are strictly out-of-fold. Exact joint actions are
    matched first, then historical rows outside the MPC state/disturbance
    context caliper are excluded. This is not a causal optimal-action score.
    """
    controls, states, disturbances = _validate_mpc_arrays(
        data, mpc_controls, mpc_states, mpc_disturbances
    )
    initial_end = int(initial_training_end_exclusive)
    minimum_samples = int(minimum_matched_samples_per_action)
    minimum_episodes = int(minimum_independent_episodes_per_action)
    minimum_training_samples = int(minimum_fold_training_samples_per_action)
    minimum_training_episodes = int(minimum_fold_training_episodes_per_action)
    enforce_episode_support = bool(exclude_insufficient_episode_actions)
    distance_threshold = float(context_distance_threshold)
    minimum_coverage = float(minimum_reportable_mpc_step_coverage_percent)
    if not 1 < initial_end < len(data.states) - 1:
        raise ValueError('Initial cross-fit training boundary is outside the data')
    if min(minimum_samples, minimum_training_samples) <= 0:
        raise ValueError('Cross-fit sample minimums must be positive')
    if enforce_episode_support and min(
        minimum_episodes,
        minimum_training_episodes,
    ) <= 0:
        raise ValueError('Enabled cross-fit episode minimums must be positive')
    if distance_threshold <= 0.0:
        raise ValueError('Context distance threshold must be positive')
    if not 0.0 <= minimum_coverage <= 100.0:
        raise ValueError('Minimum MPC step coverage must be in [0, 100]')

    (
        historical_rows,
        actual_power,
        predicted_power,
        historical_fold_numbers,
        historical_training_ends,
        fold_metadata,
    ) = _collect_crossfit_predictions(data, crossfit_folds, evaluator_family)
    historical_controls = np.rint(data.controls[historical_rows]).astype(int)
    recorded_controls = np.rint(np.asarray(data.controls, dtype=float)).astype(int)
    recorded_episode_ids = control_episode_ids(data)

    reference_context = np.concatenate(
        (data.states[:initial_end], data.disturbances[:initial_end]), axis=1
    )
    context_center = reference_context.mean(axis=0)
    context_scale = reference_context.std(axis=0)
    context_scale[context_scale <= 1e-12] = 1.0
    historical_context = np.concatenate(
        (data.states[historical_rows], data.disturbances[historical_rows]), axis=1
    )
    mpc_context = np.concatenate((states, disturbances), axis=1)
    historical_context = (historical_context - context_center) / context_scale
    mpc_context = (mpc_context - context_center) / context_scale

    action_results = []
    audit_rows = []
    for encoded_action in np.unique(controls, axis=0):
        mpc_positions = np.flatnonzero(np.all(controls == encoded_action, axis=1))
        raw_history_positions = np.flatnonzero(
            np.all(historical_controls == encoded_action, axis=1)
        )
        fold_training_support = []
        supported_history_positions = []
        support_by_fold = {}
        for fold_number in np.unique(
            historical_fold_numbers[raw_history_positions]
        ):
            fold_mask = (
                historical_fold_numbers[raw_history_positions] == fold_number
            )
            raw_fold_history_positions = raw_history_positions[fold_mask]
            training_ends = np.unique(
                historical_training_ends[raw_fold_history_positions]
            )
            if len(training_ends) != 1:
                raise ValueError('One cross-fit fold has inconsistent training bounds')
            training_end = int(training_ends[0])
            raw_training_positions = np.arange(training_end, dtype=int)
            raw_training_positions = raw_training_positions[
                np.all(recorded_controls[:training_end] == encoded_action, axis=1)
            ]
            training_episode_ids = set(
                recorded_episode_ids[raw_training_positions].tolist()
            )
            evaluation_episode_ids = set(
                recorded_episode_ids[
                    historical_rows[raw_fold_history_positions]
                ].tolist()
            )
            shared_episode_ids = training_episode_ids & evaluation_episode_ids
            training_positions = raw_training_positions[
                ~np.isin(
                    recorded_episode_ids[raw_training_positions],
                    list(shared_episode_ids),
                )
            ]
            fold_history_positions = raw_fold_history_positions[
                ~np.isin(
                    recorded_episode_ids[
                        historical_rows[raw_fold_history_positions]
                    ],
                    list(shared_episode_ids),
                )
            ]
            training_episode_count = int(
                len(np.unique(recorded_episode_ids[training_positions]))
                if len(training_positions)
                else 0
            )
            training_supported = bool(
                len(training_positions) >= minimum_training_samples
                and (
                    not enforce_episode_support
                    or training_episode_count >= minimum_training_episodes
                )
            )
            if training_supported:
                supported_history_positions.extend(fold_history_positions.tolist())
            support = {
                'fold_number': int(fold_number),
                'training_end_exclusive': training_end,
                'raw_training_exact_action_sample_count': int(
                    len(raw_training_positions)
                ),
                'training_exact_action_sample_count': int(len(training_positions)),
                'training_exact_action_episode_count': training_episode_count,
                'raw_evaluation_exact_action_sample_count': int(
                    len(raw_fold_history_positions)
                ),
                'evaluation_exact_action_sample_count': int(
                    len(fold_history_positions)
                ),
                'shared_boundary_episode_count': int(len(shared_episode_ids)),
                'training_support_eligible': training_supported,
            }
            fold_training_support.append(support)
            support_by_fold[int(fold_number)] = support
        history_positions = np.asarray(supported_history_positions, dtype=int)
        history_positions.sort()
        decoded_action = data.encoder.decode_action(encoded_action)
        mpc_to_history = _nearest_context_distance(
            mpc_context[mpc_positions], historical_context[history_positions]
        )
        history_to_mpc = _nearest_context_distance(
            historical_context[history_positions], mpc_context[mpc_positions]
        )
        matched_local_positions = np.flatnonzero(
            history_to_mpc <= distance_threshold
        )
        matched_positions = history_positions[matched_local_positions]
        matched_rows = historical_rows[matched_positions]
        context_supported_steps = int(
            np.count_nonzero(mpc_to_history <= distance_threshold)
        )
        episodes = int(
            len(np.unique(recorded_episode_ids[matched_rows]))
            if len(matched_rows)
            else 0
        )

        component_metrics = None
        total_metrics = None
        valid_total_samples = 0
        if len(matched_positions):
            matched_actual = actual_power[matched_positions]
            matched_predicted = predicted_power[matched_positions]
            component_metrics = power_prediction_metrics(
                matched_actual,
                matched_predicted,
                data.power_names,
                mape_min_actual_power_w,
            )
            total_metrics = power_prediction_metrics(
                matched_actual.sum(axis=1, keepdims=True),
                matched_predicted.sum(axis=1, keepdims=True),
                ('총공조유효전력(W)',),
                mape_min_actual_power_w,
            )
            valid_total_samples = int(
                total_metrics['mape_valid_samples_per_output'][0]
            )

        unsupported_reasons = []
        if not len(raw_history_positions):
            unsupported_reasons.append('no_exact_action_in_crossfit_rows')
        elif not len(history_positions):
            unsupported_reasons.append(
                'action_not_sufficiently_observed_in_fold_training'
            )
        if len(matched_positions) < minimum_samples:
            unsupported_reasons.append('insufficient_context_matched_samples')
        if enforce_episode_support and episodes < minimum_episodes:
            unsupported_reasons.append('insufficient_independent_episodes')
        if valid_total_samples < minimum_samples:
            unsupported_reasons.append(
                'insufficient_valid_total_power_mape_samples'
            )
        reportable_action = not unsupported_reasons
        selected_steps = int(len(mpc_positions))
        action_results.append(
            {
                'encoded_control': encoded_action.astype(int).tolist(),
                'action': decoded_action,
                'mpc_selected_step_count': selected_steps,
                'mpc_selection_frequency_percent': float(
                    100.0 * selected_steps / len(controls)
                ),
                'crossfit_raw_exact_action_sample_count': int(
                    len(raw_history_positions)
                ),
                'crossfit_exact_action_sample_count': int(len(history_positions)),
                'fold_training_support': fold_training_support,
                'context_matched_sample_count': int(len(matched_positions)),
                'independent_episode_count': episodes,
                'context_supported_mpc_step_count': context_supported_steps,
                'context_supported_mpc_step_percent_within_action': float(
                    100.0 * context_supported_steps / selected_steps
                ),
                'nearest_context_distance': _finite_distance_summary(
                    mpc_to_history
                ),
                'reportable': bool(reportable_action),
                'unsupported_reasons': unsupported_reasons,
                'total_power_prediction': total_metrics,
                'power_component_prediction': component_metrics,
            }
        )

        for local_position, history_position in zip(
            matched_local_positions, matched_positions
        ):
            source_row = int(historical_rows[history_position])
            actual_components = actual_power[history_position]
            predicted_components = predicted_power[history_position]
            actual_total = float(actual_components.sum())
            predicted_total = float(predicted_components.sum())
            valid_denominator = abs(actual_total) > float(
                mape_min_actual_power_w
            )
            audit_rows.append(
                {
                    'fold_number': int(
                        historical_fold_numbers[history_position]
                    ),
                    'fold_training_end_exclusive': int(
                        historical_training_ends[history_position]
                    ),
                    'fold_training_exact_action_sample_count': int(
                        support_by_fold[
                            int(historical_fold_numbers[history_position])
                        ]['training_exact_action_sample_count']
                    ),
                    'fold_training_exact_action_episode_count': int(
                        support_by_fold[
                            int(historical_fold_numbers[history_position])
                        ]['training_exact_action_episode_count']
                    ),
                    'row_index': source_row,
                    'timestamp': data.timestamps.iloc[source_row].isoformat(),
                    **{
                        f'control_{name}': value
                        for name, value in decoded_action.items()
                    },
                    'mpc_selected_step_count': selected_steps,
                    'nearest_mpc_context_standardized_rms_distance': float(
                        history_to_mpc[local_position]
                    ),
                    **{
                        f'actual_{name}': float(actual_components[position])
                        for position, name in enumerate(data.power_names)
                    },
                    **{
                        f'predicted_{name}': float(predicted_components[position])
                        for position, name in enumerate(data.power_names)
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
                    'action_reportable': bool(reportable_action),
                }
            )

    action_results.sort(
        key=lambda item: (
            -item['mpc_selected_step_count'],
            tuple(item['encoded_control']),
        )
    )
    reportable_actions = [item for item in action_results if item['reportable']]
    reportable_weights = [
        item['mpc_selected_step_count'] for item in reportable_actions
    ]
    partial_weighted_mape = _weighted_defined(
        [
            item['total_power_prediction']['mape_percent']
            for item in reportable_actions
        ],
        reportable_weights,
    )
    component_weighted_mape = {
        output_name: _weighted_defined(
            [
                item['power_component_prediction']['mape_percent_by_output'][
                    output_name
                ]
                for item in reportable_actions
            ],
            reportable_weights,
        )
        for output_name in data.power_names
    }
    partial_macro_mape = _weighted_defined(
        [
            item['total_power_prediction']['mape_percent']
            for item in reportable_actions
        ],
        [1] * len(reportable_actions),
    )
    component_macro_mape = {
        output_name: _weighted_defined(
            [
                item['power_component_prediction']['mape_percent_by_output'][
                    output_name
                ]
                for item in reportable_actions
            ],
            [1] * len(reportable_actions),
        )
        for output_name in data.power_names
    }

    exact_action_supported_steps = sum(
        item['mpc_selected_step_count']
        for item in action_results
        if item['crossfit_exact_action_sample_count'] > 0
    )
    raw_exact_action_observed_steps = sum(
        item['mpc_selected_step_count']
        for item in action_results
        if item['crossfit_raw_exact_action_sample_count'] > 0
    )
    context_supported_steps = sum(
        item['context_supported_mpc_step_count'] for item in action_results
    )
    reportable_action_steps = sum(reportable_weights)
    exact_action_coverage = 100.0 * exact_action_supported_steps / len(controls)
    raw_exact_action_coverage = (
        100.0 * raw_exact_action_observed_steps / len(controls)
    )
    context_coverage = 100.0 * context_supported_steps / len(controls)
    reportable_action_coverage = (
        100.0 * reportable_action_steps / len(controls)
    )
    reportable = bool(
        partial_weighted_mape is not None
        and reportable_action_coverage + 1e-12 >= minimum_coverage
        and context_coverage + 1e-12 >= minimum_coverage
    )
    if reportable:
        reason = None
    elif partial_weighted_mape is None:
        reason = (
            'No MPC-selected action met the configured cross-fit sample, '
            'episode, context, and valid-MAPE requirements'
        )
    else:
        reason = (
            'A partial-coverage MAPE exists, but MPC action or context coverage '
            'is below the configured reporting minimum'
        )

    protocol = {
        'name': 'mpc_selected_action_rolling_crossfit_mape',
        'purpose': (
            'Estimate actual-Y prediction error in historical regions supported '
            'by the joint actions selected by MPC'
        ),
        'target_action_source': 'every action applied in offline MPC rollouts',
        'prediction_protocol': (
            'expanding chronological cross-fit; each fold evaluator is trained '
            'only on earlier rows and frozen during its evaluation fold'
        ),
        'historical_matching': (
            'exact joint categorical action followed by standardized RMS '
            'state-and-disturbance context caliper'
        ),
        'context_features': list(data.state_names + data.disturbance_names),
        'context_standardization_scope': f'initial rows [0, {initial_end})',
        'context_distance_threshold': distance_threshold,
        'primary_target': 'total HVAC power = sum of configured power outputs',
        'primary_aggregation': (
            'per-action MAPE weighted by the MPC selection count of that action'
        ),
        'minimum_actual_power_w': float(mape_min_actual_power_w),
        'minimum_context_matched_samples_per_action': minimum_samples,
        'minimum_independent_episodes_per_action': minimum_episodes,
        'minimum_fold_training_samples_per_action': minimum_training_samples,
        'minimum_fold_training_episodes_per_action': minimum_training_episodes,
        'exclude_insufficient_episode_actions': enforce_episode_support,
        'minimum_reportable_mpc_step_coverage_percent': minimum_coverage,
        'unsupported_policy': (
            'Never impute actual Y; official primary value stays null when '
            'configured support coverage is not met'
        ),
        'claim_boundary': (
            'This is action-conditioned prediction fidelity, not action-label '
            'classification accuracy and not causal proof of optimality.'
        ),
    }
    return (
        {
            'available': bool(partial_weighted_mape is not None),
            'reportable': reportable,
            'reason': reason,
            'protocol': protocol,
            'evaluator_family': str(evaluator_family),
            'folds': fold_metadata,
            'mpc_selected_step_count': int(len(controls)),
            'mpc_unique_action_count': int(len(action_results)),
            'reportable_action_count': int(len(reportable_actions)),
            'coverage': {
                'raw_exact_action_observed_mpc_step_count': int(
                    raw_exact_action_observed_steps
                ),
                'raw_exact_action_observed_mpc_step_percent': float(
                    raw_exact_action_coverage
                ),
                'exact_action_supported_mpc_step_count': int(
                    exact_action_supported_steps
                ),
                'exact_action_supported_mpc_step_percent': float(
                    exact_action_coverage
                ),
                'context_supported_mpc_step_count': int(
                    context_supported_steps
                ),
                'context_supported_mpc_step_percent': float(context_coverage),
                'reportable_action_mpc_step_count': int(
                    reportable_action_steps
                ),
                'reportable_action_mpc_step_percent': float(
                    reportable_action_coverage
                ),
            },
            'primary_metric': {
                'name': (
                    'mpc_action_frequency_weighted_total_power_mape_percent'
                ),
                'value': partial_weighted_mape if reportable else None,
                'unit': 'percent',
                'reportable': reportable,
                'partial_coverage_diagnostic_value': partial_weighted_mape,
            },
            'combination_balanced_metric': {
                'name': 'mpc_selected_action_macro_total_power_mape_percent',
                'value': partial_macro_mape if reportable else None,
                'unit': 'percent',
                'reportable': reportable,
                'aggregation': (
                    'unweighted mean of reportable per-action MAPE; every joint '
                    'action has equal weight'
                ),
                'included_action_count': int(len(reportable_actions)),
                'partial_coverage_diagnostic_value': partial_macro_mape,
            },
            'frequency_weighted_component_mape_percent': (
                component_weighted_mape
                if reportable
                else {name: None for name in data.power_names}
            ),
            'partial_coverage_frequency_weighted_component_mape_percent': (
                component_weighted_mape
            ),
            'combination_balanced_component_mape_percent': (
                component_macro_mape
                if reportable
                else {name: None for name in data.power_names}
            ),
            'partial_coverage_combination_balanced_component_mape_percent': (
                component_macro_mape
            ),
            'actions': action_results,
        },
        audit_rows,
    )
