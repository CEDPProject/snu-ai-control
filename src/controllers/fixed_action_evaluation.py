"""Outcome-blind fixed-action holdout for the official offline MAPE.

The ordinary MPC backtest keeps its chronological split.  This module builds a
separate evaluation-only split from complete joint-action episodes because the
recorded experiment changes actions in long blocks and has no exact action with
adequate support in all three chronological partitions.  Episode assignment
uses controls and timestamps only; power and state targets are never inspected.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf

from controllers.action_crossfit import control_episode_ids
from controllers.evaluation import (
    _joint_constraint_satisfaction_mask,
    constraint_prediction_metrics,
    power_prediction_metrics,
)
from controllers.fixed_action_power_models import (
    select_and_fit_fixed_action_power_model,
)
from dynamics.evaluator_base import TransitionPredictions


@dataclass(frozen=True)
class FixedActionHoldoutSplit:
    """Explicit transition rows and candidate actions for one official holdout."""

    train_rows: np.ndarray
    validation_rows: np.ndarray
    test_rows: np.ndarray
    candidate_controls: np.ndarray
    metadata: Dict[str, object]
    audit_rows: list[Dict[str, object]]
    dataset_rows: list[Dict[str, object]]


def _plain(config) -> Dict[str, object]:
    if config is None:
        return {}
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


def _action_key(control) -> tuple[int, ...]:
    return tuple(np.rint(np.asarray(control, dtype=float)).astype(int).tolist())


def _best_episode_partition(
    episode_ids: Sequence[int],
    sample_count_by_episode: Mapping[int, int],
    *,
    minimum_training_samples: int,
    minimum_training_episodes: int,
    minimum_validation_samples: int,
    minimum_validation_episodes: int,
    minimum_test_samples: int,
    minimum_test_episodes: int,
):
    """Assign whole episodes while maximizing actual-Y test support.

    Validation and test use exactly their configured minimum number of episodes;
    every remaining episode is assigned to training.  This intentionally keeps
    as much data as possible in the frozen model while reserving a large test
    episode whenever the recorded design contains one.
    """
    episode_ids = tuple(sorted(int(value) for value in episode_ids))
    required_episodes = (
        int(minimum_training_episodes)
        + int(minimum_validation_episodes)
        + int(minimum_test_episodes)
    )
    if len(episode_ids) < required_episodes:
        return None
    total_samples = sum(sample_count_by_episode[value] for value in episode_ids)
    if total_samples < (
        int(minimum_training_samples)
        + int(minimum_validation_samples)
        + int(minimum_test_samples)
    ):
        return None

    best = None
    for test_ids in combinations(episode_ids, int(minimum_test_episodes)):
        test_samples = sum(sample_count_by_episode[value] for value in test_ids)
        if test_samples < int(minimum_test_samples):
            continue
        after_test = tuple(value for value in episode_ids if value not in test_ids)
        for validation_ids in combinations(
            after_test,
            int(minimum_validation_episodes),
        ):
            validation_samples = sum(
                sample_count_by_episode[value] for value in validation_ids
            )
            if validation_samples < int(minimum_validation_samples):
                continue
            training_ids = tuple(
                value
                for value in after_test
                if value not in validation_ids
            )
            training_samples = sum(
                sample_count_by_episode[value] for value in training_ids
            )
            if (
                len(training_ids) < int(minimum_training_episodes)
                or training_samples < int(minimum_training_samples)
            ):
                continue
            score = (
                int(test_samples),
                int(validation_samples),
                int(training_samples),
            )
            tie_break = (tuple(test_ids), tuple(validation_ids))
            candidate = (score, tie_break, training_ids, validation_ids, test_ids)
            if best is None:
                best = candidate
                continue
            if score > best[0] or (score == best[0] and tie_break < best[1]):
                best = candidate
    if best is None:
        return None
    _, _, training_ids, validation_ids, test_ids = best
    return {
        'train': tuple(int(value) for value in training_ids),
        'validation': tuple(int(value) for value in validation_ids),
        'test': tuple(int(value) for value in test_ids),
    }


def _build_candidate_dataset_rows(data, candidates, episode_rows):
    """Materialize exact-action transition rows for every candidate split."""
    rows = []
    for candidate in candidates:
        rank = int(candidate['rank_by_total_observed_frequency'])
        decoded = candidate['action']
        candidate_id = f'action_{rank:03d}'
        for split_name in ('train', 'validation', 'test'):
            episode_ids = candidate['support'][split_name]['episode_ids']
            for episode_id in episode_ids:
                for source_row_index in episode_rows[int(episode_id)]:
                    source_row_index = int(source_row_index)
                    row = {
                        'candidate_id': candidate_id,
                        'rank_by_total_observed_frequency': rank,
                        'split': split_name,
                        'episode_id': int(episode_id),
                        'source_row_index': source_row_index,
                        'timestamp': data.timestamps.iloc[
                            source_row_index
                        ].isoformat(),
                        'next_timestamp': data.timestamps.iloc[
                            source_row_index + 1
                        ].isoformat(),
                    }
                    row.update({
                        f'control_{name}': value
                        for name, value in decoded.items()
                    })
                    row.update({
                        f'state_t_{name}': float(value)
                        for name, value in zip(
                            data.state_names, data.states[source_row_index]
                        )
                    })
                    row.update({
                        f'disturbance_t_{name}': float(value)
                        for name, value in zip(
                            data.disturbance_names,
                            data.disturbances[source_row_index],
                        )
                    })
                    row.update({
                        f'target_power_{name}': float(value)
                        for name, value in zip(
                            data.power_names, data.powers[source_row_index]
                        )
                    })
                    row.update({
                        f'target_next_state_{name}': float(value)
                        for name, value in zip(
                            data.state_names,
                            data.states[source_row_index + 1],
                        )
                    })
                    rows.append(row)
    return rows


def build_fixed_action_episode_holdout(
    data,
    config,
    *,
    split_rate: Sequence[float],
) -> FixedActionHoldoutSplit:
    """Build an action-stratified, episode-disjoint, outcome-blind holdout."""
    specification = _plain(config)
    top_k = int(specification['top_k_actions'])
    minimums = {
        'train_samples': int(
            specification['minimum_training_samples_per_candidate']
        ),
        'train_episodes': int(
            specification['minimum_training_episodes_per_candidate']
        ),
        'validation_samples': int(
            specification['minimum_validation_samples_per_candidate']
        ),
        'validation_episodes': int(
            specification['minimum_validation_episodes_per_candidate']
        ),
        'test_samples': int(specification['minimum_test_exact_matches']),
        'test_episodes': int(
            specification['minimum_test_episodes_per_candidate']
        ),
    }
    if top_k < 0:
        raise ValueError('top_k_actions must be zero (all) or positive')
    if any(value <= 0 for value in minimums.values()):
        raise ValueError('Fixed-action holdout support minimums must be positive')
    if len(split_rate) != 3 or not np.isclose(sum(split_rate), 1.0):
        raise ValueError('Fixed-action holdout split_rate must sum to one')

    controls = np.rint(np.asarray(data.controls, dtype=float)).astype(int)
    episode_ids = control_episode_ids(data)
    valid_rows = data.valid_transition_rows(np.arange(len(controls) - 1))
    valid_row_set = set(valid_rows.tolist())
    episode_positions = {
        int(episode_id): np.flatnonzero(episode_ids == episode_id)
        for episode_id in np.unique(episode_ids)
    }
    episode_rows = {
        episode_id: np.asarray(
            [row for row in positions if int(row) in valid_row_set],
            dtype=int,
        )
        for episode_id, positions in episode_positions.items()
    }
    episode_rows = {
        episode_id: rows
        for episode_id, rows in episode_rows.items()
        if len(rows)
    }
    episode_action = {
        episode_id: _action_key(controls[rows[0]])
        for episode_id, rows in episode_rows.items()
    }
    action_episodes: Dict[tuple[int, ...], list[int]] = {}
    for episode_id, action in episode_action.items():
        action_episodes.setdefault(action, []).append(episode_id)
    sample_count_by_episode = {
        episode_id: int(len(rows)) for episode_id, rows in episode_rows.items()
    }

    feasible = []
    for action, action_episode_ids in action_episodes.items():
        assignment = _best_episode_partition(
            action_episode_ids,
            sample_count_by_episode,
            minimum_training_samples=minimums['train_samples'],
            minimum_training_episodes=minimums['train_episodes'],
            minimum_validation_samples=minimums['validation_samples'],
            minimum_validation_episodes=minimums['validation_episodes'],
            minimum_test_samples=minimums['test_samples'],
            minimum_test_episodes=minimums['test_episodes'],
        )
        if assignment is None:
            continue
        feasible.append(
            {
                'action_key': action,
                'total_sample_count': int(
                    sum(sample_count_by_episode[value] for value in action_episode_ids)
                ),
                'total_episode_count': int(len(action_episode_ids)),
                'assignment': assignment,
            }
        )
    feasible.sort(
        key=lambda item: (
            -item['total_sample_count'],
            -item['total_episode_count'],
            item['action_key'],
        )
    )
    selected_candidates = feasible if top_k == 0 else feasible[:top_k]

    nominal_train_end = int(len(controls) * float(split_rate[0]))
    nominal_test_start = int(
        len(controls) * float(split_rate[0] + split_rate[1])
    )
    split_by_episode = {}
    for episode_id, positions in episode_positions.items():
        midpoint = float(np.mean(positions))
        split_by_episode[int(episode_id)] = (
            'train'
            if midpoint < nominal_train_end
            else 'validation'
            if midpoint < nominal_test_start
            else 'test'
        )
    for candidate in selected_candidates:
        for split_name, assigned_episode_ids in candidate['assignment'].items():
            for episode_id in assigned_episode_ids:
                split_by_episode[int(episode_id)] = split_name

    rows_by_split = {
        split_name: np.asarray(
            sorted(
                row
                for episode_id, rows in episode_rows.items()
                if split_by_episode[episode_id] == split_name
                for row in rows.tolist()
            ),
            dtype=int,
        )
        for split_name in ('train', 'validation', 'test')
    }
    combined = np.concatenate(list(rows_by_split.values()))
    if len(np.unique(combined)) != len(combined):
        raise AssertionError('Fixed-action split rows must be mutually exclusive')
    if any(not len(rows) for rows in rows_by_split.values()):
        raise ValueError('Fixed-action split produced an empty partition')

    candidate_results = []
    audit_rows = []
    for rank, candidate in enumerate(selected_candidates, start=1):
        action = candidate['action_key']
        encoded = np.asarray(action, dtype=int)
        assignment = candidate['assignment']
        support = {}
        for split_name, assigned_episode_ids in assignment.items():
            support[split_name] = {
                'sample_count': int(
                    sum(
                        sample_count_by_episode[value]
                        for value in assigned_episode_ids
                    )
                ),
                'episode_count': int(len(assigned_episode_ids)),
                'episode_ids': [int(value) for value in assigned_episode_ids],
            }
        decoded = data.encoder.decode_action(encoded)
        result = {
            'rank_by_total_observed_frequency': int(rank),
            'encoded_control': encoded.tolist(),
            'action': decoded,
            'total_sample_count': candidate['total_sample_count'],
            'total_episode_count': candidate['total_episode_count'],
            'support': support,
        }
        candidate_results.append(result)
        audit_rows.append(
            {
                'rank_by_total_observed_frequency': int(rank),
                **{f'control_{name}': value for name, value in decoded.items()},
                'total_sample_count': candidate['total_sample_count'],
                'total_episode_count': candidate['total_episode_count'],
                **{
                    f'{split_name}_{metric}': value
                    for split_name, values in support.items()
                    for metric, value in values.items()
                    if metric != 'episode_ids'
                },
            }
        )

    metadata = {
        'protocol': {
            'name': 'outcome_blind_action_episode_grouped_holdout',
            'purpose': (
                'Create disjoint support for one fixed joint action without '
                'using power or state targets during split construction'
            ),
            'assignment_inputs': 'encoded controls, timestamps, and episode lengths only',
            'outcome_columns_used_for_assignment': [],
            'episode_definition': (
                'maximal one-minute contiguous run of one exact joint action'
            ),
            'row_overlap_allowed': False,
            'temporal_generalization_claim': False,
            'interpretation': (
                'This grouped holdout measures fixed-action prediction fidelity; '
                'the ordinary chronological split remains authoritative for future-time '
                'model and MPC policy validation.'
            ),
            'top_k_actions': top_k,
            **{f'minimum_{key}': value for key, value in minimums.items()},
        },
        'split_transition_counts': {
            name: int(len(rows)) for name, rows in rows_by_split.items()
        },
        'split_timestamp_ranges': {
            name: {
                'minimum': data.timestamps.iloc[rows].min().isoformat(),
                'maximum': data.timestamps.iloc[rows].max().isoformat(),
            }
            for name, rows in rows_by_split.items()
        },
        'candidate_count': int(len(candidate_results)),
        'availability_reason': (
            None
            if candidate_results
            else (
                'No joint action satisfies all configured train, validation, '
                'and test sample/episode support minimums'
            )
        ),
        'candidates': candidate_results,
    }
    return FixedActionHoldoutSplit(
        train_rows=rows_by_split['train'],
        validation_rows=rows_by_split['validation'],
        test_rows=rows_by_split['test'],
        candidate_controls=(
            np.asarray(
                [item['encoded_control'] for item in candidate_results],
                dtype=float,
            )
            if candidate_results
            else np.empty((0, data.encoder.n_features), dtype=float)
        ),
        metadata=metadata,
        audit_rows=audit_rows,
        dataset_rows=_build_candidate_dataset_rows(
            data, candidate_results, episode_rows
        ),
    )


def _predict_control_rows(evaluator, data, row_indices, encoded_control):
    rows = data.valid_transition_rows(row_indices)
    if not len(rows):
        raise ValueError('No valid transition rows for fixed-action prediction')
    sequence_predictor = getattr(evaluator, 'predict_control_rows', None)
    if sequence_predictor is not None:
        return sequence_predictor(data, rows, encoded_control)
    controls = np.repeat(
        np.asarray(encoded_control, dtype=float).reshape(1, -1),
        len(rows),
        axis=0,
    )
    states = data.states[rows]
    disturbances = data.disturbances[rows]
    return TransitionPredictions(
        actual_states=data.states[rows + 1],
        predicted_states=evaluator.predict_state(states, controls, disturbances),
        actual_powers=data.powers[rows],
        predicted_powers=evaluator.predict_power(states, controls, disturbances),
        row_indices=rows,
    )


def evaluate_fixed_optimal_action_holdout(
    data,
    holdout: FixedActionHoldoutSplit,
    selection_evaluator,
    frozen_report_evaluator,
    *,
    state_constraints: Sequence[Mapping],
    minimum_predicted_state_satisfaction_percent: float,
    allow_soft_constraint_fallback: bool,
    minimum_test_exact_matches: int,
    minimum_test_episodes: int,
    minimum_test_mape_samples: int,
    mape_min_actual_power_w: float,
    selection_evaluator_name: str,
    frozen_report_evaluator_name: str,
    total_power_report_config=None,
    selection_preference=None,
    total_power_random_state: int = 0,
    target_mape_percent: float = 1.95,
) -> tuple[Dict[str, object], list[Dict[str, object]]]:
    """Select one action on validation predictions, freeze it, and test MAPE."""
    minimum_satisfaction = float(minimum_predicted_state_satisfaction_percent)
    if not 0.0 <= minimum_satisfaction <= 100.0:
        raise ValueError('Predicted state satisfaction minimum must be in [0, 100]')
    preference_specification = _plain(selection_preference)
    preference_enabled = bool(preference_specification.get('enabled', False))
    preferred_values = {
        str(name): str(value).strip()
        for name, value in dict(
            preference_specification.get('preferred_values', {})
        ).items()
    }
    maximum_satisfaction_loss = float(
        preference_specification.get(
            'maximum_state_satisfaction_loss_percent', 0.0
        )
    )
    if maximum_satisfaction_loss < 0.0:
        raise ValueError(
            'maximum_state_satisfaction_loss_percent must be non-negative'
        )
    for variable_name, preferred_value in preferred_values.items():
        data.encoder.index(variable_name, preferred_value)
    if preference_enabled and not preferred_values:
        raise ValueError(
            'Enabled fixed-action selection preference needs preferred_values'
        )
    support_by_action = {
        tuple(item['encoded_control']): item
        for item in holdout.metadata['candidates']
    }
    candidates = []
    for encoded_control in holdout.candidate_controls:
        predictions = _predict_control_rows(
            selection_evaluator,
            data,
            holdout.validation_rows,
            encoded_control,
        )
        predicted_satisfied = _joint_constraint_satisfaction_mask(
            predictions.predicted_states,
            state_constraints,
        )
        satisfaction = float(100.0 * predicted_satisfied.mean())
        mean_components = predictions.predicted_powers.mean(axis=0)
        support = support_by_action[_action_key(encoded_control)]
        eligible = satisfaction + 1e-12 >= minimum_satisfaction
        decoded_action = data.encoder.decode_action(encoded_control)
        preference_match_count = sum(
            decoded_action[name] == value
            for name, value in preferred_values.items()
        )

        candidates.append(
            {
                'encoded_control': np.rint(encoded_control).astype(int).tolist(),
                'action': decoded_action,
                'validation_common_context_count': int(
                    len(predictions.row_indices)
                ),
                'predicted_state_constraint_satisfaction_percent': satisfaction,
                'predicted_mean_power_w_by_output': dict(
                    zip(data.power_names, mean_components.tolist())
                ),
                'predicted_mean_total_power_w': float(mean_components.sum()),
                'support': support['support'],
                'eligible': bool(eligible),
                'preference_match_count': int(preference_match_count),
                'matches_all_preferred_values': bool(
                    preferred_values
                    and preference_match_count == len(preferred_values)
                ),

            }
        )
    eligible_candidates = [item for item in candidates if item['eligible']]
    soft_constraint_fallback_used = bool(
        not eligible_candidates and allow_soft_constraint_fallback and candidates
    )
    preference_applied = False
    preference_pool_count = 0
    if eligible_candidates:
        selection_pool = eligible_candidates
        fully_preferred = [
            item
            for item in selection_pool
            if item['matches_all_preferred_values']
        ]
        if preference_enabled and fully_preferred:
            selection_pool = fully_preferred
            preference_applied = True
        preference_pool_count = len(selection_pool)
        selected = min(
            selection_pool,
            key=lambda item: (
                item['predicted_mean_total_power_w'],
                tuple(item['encoded_control']),
            ),
        )
        selection_mode = (
            'strict_state_constraint_then_preferred_values_then_minimum_power'
            if preference_applied
            else 'strict_state_constraint_then_minimum_power'
        )
    elif soft_constraint_fallback_used:
        maximum_satisfaction = max(
            item['predicted_state_constraint_satisfaction_percent']
            for item in candidates
        )
        selection_pool = [
            item
            for item in candidates
            if item['predicted_state_constraint_satisfaction_percent']
            + maximum_satisfaction_loss
            + 1e-12
            >= maximum_satisfaction
        ]
        if preference_enabled:
            fully_preferred = [
                item
                for item in selection_pool
                if item['matches_all_preferred_values']
            ]
            if fully_preferred:
                selection_pool = fully_preferred
                preference_applied = True
            else:
                maximum_match_count = max(
                    item['preference_match_count'] for item in selection_pool
                )
                selection_pool = [
                    item
                    for item in selection_pool
                    if item['preference_match_count'] == maximum_match_count
                ]
                preference_applied = bool(maximum_match_count)
        preference_pool_count = len(selection_pool)
        selected = min(
            selection_pool,
            key=lambda item: (
                -item['predicted_state_constraint_satisfaction_percent'],
                item['predicted_mean_total_power_w'],
                tuple(item['encoded_control']),
            ),
        )
        selection_mode = (
            'state_tolerance_then_preferred_values_then_satisfaction_then_'
            'power_fallback'
            if preference_applied
            else 'maximum_state_satisfaction_then_minimum_power_fallback'
        )
    else:
        selected = None
        selection_mode = 'no_selectable_candidate'
    for item in candidates:
        item['selected'] = bool(item is selected)
    candidates.sort(
        key=lambda item: (
            not item['eligible'],
            item['predicted_mean_total_power_w'],
            tuple(item['encoded_control']),
        )
    )

    total_power_specification = _plain(total_power_report_config)
    total_power_report_enabled = bool(
        total_power_specification.get('enabled', False)
    )
    protocol = {
        'name': 'fixed_optimal_joint_action_frozen_model_holdout_mape',
        'purpose': (
            'Implement the requested protocol: finish action exploration on '
            'validation, freeze one action and one prediction model, then compare '
            'predicted and actual target Y on an independent test partition'
        ),
        'independent_variable': 'one exact joint categorical HVAC control action',
        'target_y': 'total HVAC power = sum of configured power outputs',
        'selection_model': selection_evaluator_name,
        'selection_model_training_scope': 'episode-grouped train rows only',
        'selection_context_scope': 'episode-grouped validation rows only',
        'selection_rule': (
            'minimum predicted total power after state feasibility and the '
            'optional exact preferred-value match are applied'
        ),
        'selection_preference': {
            'enabled': preference_enabled,
            'preferred_values': preferred_values,
            'maximum_state_satisfaction_loss_percent': (
                maximum_satisfaction_loss
            ),
            'scope': (
                'Strict candidates must still satisfy the configured state '
                'rate. During soft fallback, preference is considered only '
                'among candidates within the configured loss from the maximum '
                'predicted state-satisfaction rate.'
            ),
        },
        'soft_constraint_fallback': {
            'enabled': bool(allow_soft_constraint_fallback),
            'rule': (
                'If no candidate reaches the configured state-satisfaction rate, '
                'keep candidates within the configured loss from the maximum, '
                'apply preferred values, maximize satisfaction, then minimize power. '
                'The controller acceptance state gate still fails separately.'
            ),
        },
        'state_and_component_report_model': frozen_report_evaluator_name,
        'total_power_report_model': (
            'validation-selected direct total-power model'
            if total_power_report_enabled
            else frozen_report_evaluator_name
        ),
        'report_model_training_scope': (
            'family selection on grouped train/validation only; selected family '
            'refit on episode-grouped train + validation rows'
            if total_power_report_enabled
            else 'episode-grouped train + validation rows'
        ),
        'report_model_frozen_during_test': True,
        'report_model_used_by_mpc_or_milp': False,
        'selected_action_frozen_during_test': True,
        'test_matching': 'exact recorded joint action',
        'test_power_and_state_targets_used_for_selection': False,
        'primary_metric_formula': (
            'MAPE(%) = 100/n * sum(abs(actual_y - predicted_y) / abs(actual_y))'
        ),
        'primary_metric_aggregation': 'direct mean over valid exact-action test rows',
        'minimum_actual_power_w': float(mape_min_actual_power_w),
        'minimum_predicted_state_satisfaction_percent': minimum_satisfaction,
        'minimum_test_exact_matches': int(minimum_test_exact_matches),
        'minimum_test_episodes': int(minimum_test_episodes),
        'minimum_test_mape_samples': int(minimum_test_mape_samples),
        'target_mape_percent': float(target_mape_percent),
        'split': holdout.metadata['protocol'],
        'claim_boundary': (
            'Observational fixed-action prediction accuracy; not causal proof '
            'that the selected action is optimal on real equipment'
        ),
    }
    selection = {
        'candidate_count': int(len(candidates)),
        'eligible_candidate_count': int(len(eligible_candidates)),
        'strict_constraint_candidate_available': bool(eligible_candidates),
        'soft_constraint_fallback_used': soft_constraint_fallback_used,
        'selection_mode': selection_mode,
        'selection_preference_applied': bool(preference_applied),
        'selection_preference_pool_count': int(preference_pool_count),
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
                'common_context_count': selected[
                    'validation_common_context_count'
                ],
                'preference_match_count': selected[
                    'preference_match_count'
                ],
                'matches_all_preferred_values': selected[
                    'matches_all_preferred_values'
                ],
            }
            if selected
            else None
        ),
        'frozen_before_test': bool(selected is not None),
        'candidates': candidates,
    }
    empty_primary = {
        'name': 'fixed_optimal_action_total_power_mape_percent',
        'value': None,
        'unit': 'percent',
        'reportable': False,
        'diagnostic_value': None,
        'target_percent': float(target_mape_percent),
        'target_achieved': False,
    }
    if selected is None:
        reason = (
            holdout.metadata.get('availability_reason')
            if not candidates
            else (
                'No candidate met predicted state-constraint satisfaction '
                'and soft-constraint fallback is disabled'
            )
        )
        return (
            {
                'available': False,
                'reportable': False,
                'reason': reason,
                'protocol': protocol,
                'split': holdout.metadata,
                'selection': selection,
                'primary_metric': empty_primary,
                'test_evaluation': None,
            },
            [],
        )

    selected_control = np.asarray(selected['encoded_control'], dtype=int)
    total_power_model_selection = None
    if total_power_report_enabled:
        total_power_model_selection = select_and_fit_fixed_action_power_model(
            data,
            training_rows=holdout.train_rows,
            validation_rows=holdout.validation_rows,
            selected_control=selected_control,
            config=total_power_specification,
            mape_min_actual_power_w=float(mape_min_actual_power_w),
            random_state=int(total_power_random_state),
        )
        protocol['total_power_report_model'] = (
            total_power_model_selection.selected_family
        )
    exact_rows = holdout.test_rows[
        np.all(
            np.rint(data.controls[holdout.test_rows]).astype(int)
            == selected_control,
            axis=1,
        )
    ]
    predictions = _predict_control_rows(
        frozen_report_evaluator,
        data,
        exact_rows,
        selected_control,
    )
    evaluated_rows = np.asarray(predictions.row_indices, dtype=int)
    component_metrics = power_prediction_metrics(
        predictions.actual_powers,
        predictions.predicted_powers,
        data.power_names,
        mape_min_actual_power_w,
    )
    actual_total = predictions.actual_powers.sum(axis=1, keepdims=True)
    legacy_predicted_total = predictions.predicted_powers.sum(
        axis=1,
        keepdims=True,
    )
    legacy_total_metrics = power_prediction_metrics(
        actual_total,
        legacy_predicted_total,
        ('총공조유효전력(W)',),
        mape_min_actual_power_w,
    )
    if total_power_model_selection is None:
        predicted_total = legacy_predicted_total
        official_total_power_model_family = frozen_report_evaluator_name
    else:
        predicted_total = total_power_model_selection.model.predict(
            data.states[evaluated_rows],
            data.controls[evaluated_rows],
            data.disturbances[evaluated_rows],
        )
        official_total_power_model_family = (
            total_power_model_selection.selected_family
        )
    total_metrics = power_prediction_metrics(
        actual_total,
        predicted_total,
        ('총공조유효전력(W)',),
        mape_min_actual_power_w,
    )
    valid_mape_samples = int(total_metrics['mape_valid_samples_per_output'][0])
    test_support = selected['support']['test']
    primary_mape = total_metrics['mape_percent']
    reportable = bool(
        primary_mape is not None
        and len(evaluated_rows) >= int(minimum_test_exact_matches)
        and int(test_support['episode_count']) >= int(minimum_test_episodes)
        and valid_mape_samples >= int(minimum_test_mape_samples)
    )
    predicted_satisfied = _joint_constraint_satisfaction_mask(
        predictions.predicted_states,
        state_constraints,
    )
    actual_satisfied = _joint_constraint_satisfaction_mask(
        predictions.actual_states,
        state_constraints,
    )
    primary_metric = {
        'name': 'fixed_optimal_action_total_power_mape_percent',
        'value': float(primary_mape) if reportable else None,
        'unit': 'percent',
        'reportable': reportable,
        'diagnostic_value': (
            float(primary_mape) if primary_mape is not None else None
        ),
        'target_percent': float(target_mape_percent),
        'target_achieved': bool(
            reportable
            and float(primary_mape) <= float(target_mape_percent) + 1e-12
        ),
    }
    test_evaluation = {
        'test_transition_count': int(len(holdout.test_rows)),
        'exact_match_count': int(len(evaluated_rows)),
        'exact_match_coverage_percent': float(
            100.0 * len(evaluated_rows) / len(holdout.test_rows)
        ),
        'exact_match_episode_count': int(test_support['episode_count']),
        'valid_primary_mape_sample_count': valid_mape_samples,
        'primary_metric': primary_metric,
        'official_total_power_model_family': official_total_power_model_family,
        'total_power_prediction': total_metrics,
        'legacy_component_sum_total_power_prediction': legacy_total_metrics,
        'power_component_prediction': component_metrics,
        'actual_joint_state_constraint_satisfaction_percent': float(
            100.0 * actual_satisfied.mean()
        ),
        'predicted_joint_state_constraint_satisfaction_percent': float(
            100.0 * predicted_satisfied.mean()
        ),
        'constraint_prediction': constraint_prediction_metrics(
            predictions.actual_states,
            predictions.predicted_states,
            state_constraints,
        ),
    }
    power_model_selection = (
        {
            **total_power_model_selection.metadata,
            'validation_results': (
                total_power_model_selection.validation_results
            ),
            'legacy_component_sum_test_mape_percent': (
                legacy_total_metrics['mape_percent']
            ),
            'selected_model_test_mape_percent': primary_mape,
            'test_targets_used_to_select_family': False,
        }
        if total_power_model_selection is not None
        else {
            'enabled': False,
            'selected_family': frozen_report_evaluator_name,
            'validation_results': [],
            'legacy_component_sum_test_mape_percent': (
                legacy_total_metrics['mape_percent']
            ),
            'selected_model_test_mape_percent': primary_mape,
            'test_targets_used_to_select_family': False,
        }
    )

    audit_rows = []
    for position, row_index in enumerate(evaluated_rows):
        actual_y = float(actual_total[position, 0])
        predicted_y = float(predicted_total[position, 0])
        denominator_valid = abs(actual_y) > float(mape_min_actual_power_w)
        audit_rows.append(
            {
                'source_row_index': int(row_index),
                'timestamp': data.timestamps.iloc[int(row_index)].isoformat(),
                **{
                    f'control_{name}': value
                    for name, value in selected['action'].items()
                },
                **{
                    f'actual_{name}': float(predictions.actual_powers[position, output])
                    for output, name in enumerate(data.power_names)
                },
                **{
                    f'predicted_{name}': float(
                        predictions.predicted_powers[position, output]
                    )
                    for output, name in enumerate(data.power_names)
                },
                'actual_total_hvac_power_w': actual_y,
                'predicted_total_hvac_power_w': predicted_y,
                'legacy_component_sum_predicted_total_hvac_power_w': float(
                    legacy_predicted_total[position, 0]
                ),
                'official_total_power_model_family': (
                    official_total_power_model_family
                ),
                'mape_denominator_valid': bool(denominator_valid),
                'absolute_percentage_error_percent': (
                    float(100.0 * abs(predicted_y - actual_y) / abs(actual_y))
                    if denominator_valid
                    else None
                ),
                'actual_state_constraints_satisfied': bool(
                    actual_satisfied[position]
                ),
                'predicted_state_constraints_satisfied': bool(
                    predicted_satisfied[position]
                ),
            }
        )
    return (
        {
            'available': bool(primary_mape is not None),
            'reportable': reportable,
            'reason': (
                None
                if reportable
                else 'Computed MAPE did not meet exact-match, episode, or valid-sample gates'
            ),
            'protocol': protocol,
            'split': holdout.metadata,
            'selection': selection,
            'total_power_model_selection': power_model_selection,
            'primary_metric': primary_metric,
            'test_evaluation': test_evaluation,
        },
        audit_rows,
    )
