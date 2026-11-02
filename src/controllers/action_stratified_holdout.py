"""Outcome-blind exact-action-stratified row holdout.

This supplementary protocol allocates valid transition rows inside each exact
joint categorical action to train, validation, and test.  It deliberately does
not keep a contiguous action episode in one partition.  The resulting accuracy
therefore measures action-distribution-balanced interpolation, not future-time
or episode-independent generalization.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from omegaconf import OmegaConf

from controllers.fixed_action_evaluation import (
    FixedActionHoldoutSplit,
    _action_key,
    _plain,
)


def _proportional_split_counts(
    sample_count: int,
    split_rate: Sequence[float],
) -> np.ndarray:
    """Return deterministic integer counts closest to the requested ratio."""
    quotas = np.asarray(split_rate, dtype=float) * int(sample_count)
    counts = np.floor(quotas).astype(int)
    remaining = int(sample_count) - int(counts.sum())
    fractional_order = sorted(
        range(len(counts)),
        key=lambda index: (-float(quotas[index] - counts[index]), index),
    )
    for index in fractional_order[:remaining]:
        counts[index] += 1
    return counts


def _counts_with_minimum_support(
    sample_count: int,
    split_rate: Sequence[float],
    minimum_counts: Sequence[int],
) -> np.ndarray | None:
    """Allocate rows near the requested ratio while enforcing split minima."""
    minimum_counts = np.asarray(minimum_counts, dtype=int)
    if int(sample_count) < int(minimum_counts.sum()):
        return None
    counts = _proportional_split_counts(sample_count, split_rate)
    for recipient, minimum in enumerate(minimum_counts):
        deficit = int(minimum - counts[recipient])
        while deficit > 0:
            donors = [
                index
                for index, value in enumerate(counts)
                if index != recipient and int(value) > int(minimum_counts[index])
            ]
            if not donors:
                return None
            donor = max(
                donors,
                key=lambda index: (
                    int(counts[index] - minimum_counts[index]),
                    int(counts[index]),
                    -index,
                ),
            )
            counts[donor] -= 1
            counts[recipient] += 1
            deficit -= 1
    if (counts < minimum_counts).any() or int(counts.sum()) != int(sample_count):
        return None
    return counts


def _row_stratified_rng(
    random_state: int,
    action_key: Sequence[int],
) -> np.random.Generator:
    """Make an action-specific deterministic generator."""
    entropy = [int(random_state), *(int(value) for value in action_key)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def build_action_stratified_row_holdout(
    data,
    config,
    *,
    split_rate: Sequence[float],
    random_state: int,
) -> FixedActionHoldoutSplit:
    """Create a per-exact-action row split without target-driven assignment.

    Every valid transition is assigned exactly once.  An action with enough
    rows for the configured support minima is a candidate; rare actions still
    occupy split rows and therefore remain in the coverage denominator.
    """
    specification = _plain(config)
    top_k = int(specification['top_k_actions'])
    minimums = {
        'train_samples': int(
            specification['minimum_training_samples_per_candidate']
        ),
        'validation_samples': int(
            specification['minimum_validation_samples_per_candidate']
        ),
        'test_samples': int(specification['minimum_test_exact_matches']),
    }
    if top_k < 0:
        raise ValueError('top_k_actions must be zero (all) or positive')
    if any(value <= 0 for value in minimums.values()):
        raise ValueError('Action-stratified support minimums must be positive')
    if len(split_rate) != 3 or not np.isclose(sum(split_rate), 1.0):
        raise ValueError('Action-stratified split_rate must sum to one')

    controls = np.rint(np.asarray(data.controls, dtype=float)).astype(int)
    valid_rows = data.valid_transition_rows(np.arange(len(controls) - 1))
    rows_by_action: Dict[tuple[int, ...], list[int]] = {}
    for row in valid_rows.tolist():
        rows_by_action.setdefault(_action_key(controls[int(row)]), []).append(
            int(row)
        )

    split_names = ('train', 'validation', 'test')
    required_counts = np.asarray(
        [
            minimums['train_samples'],
            minimums['validation_samples'],
            minimums['test_samples'],
        ],
        dtype=int,
    )
    assigned_rows = {name: [] for name in split_names}
    action_assignments = []
    for action_key in sorted(rows_by_action):
        rows = np.asarray(sorted(rows_by_action[action_key]), dtype=int)
        shuffled_rows = rows[
            _row_stratified_rng(random_state, action_key).permutation(len(rows))
        ]
        strict_counts = _counts_with_minimum_support(
            len(shuffled_rows),
            split_rate,
            required_counts,
        )
        if strict_counts is not None:
            counts = strict_counts
        elif len(shuffled_rows) >= len(split_names):
            # A rare action cannot meet the official 5/3/5 support gate, but
            # when it has at least three rows it should still be represented
            # in every split. This keeps the supplementary split faithful to
            # the per-action distribution objective without duplicating data.
            counts = _counts_with_minimum_support(
                len(shuffled_rows),
                split_rate,
                (1, 1, 1),
            )
            if counts is None:
                raise AssertionError(
                    'One-row-per-split allocation unexpectedly became infeasible'
                )
        else:
            # With fewer than three rows, no non-overlapping three-way split
            # can represent the action everywhere; retain the closest
            # proportional allocation and keep it as a non-candidate.
            counts = _proportional_split_counts(len(shuffled_rows), split_rate)
        cursor = 0
        split_rows = {}
        for split_name, count in zip(split_names, counts.tolist()):
            chosen = np.sort(shuffled_rows[cursor : cursor + int(count)])
            split_rows[split_name] = chosen
            assigned_rows[split_name].extend(chosen.tolist())
            cursor += int(count)
        if cursor != len(shuffled_rows):
            raise AssertionError('Action-stratified split did not assign all rows')
        action_assignments.append(
            {
                'action_key': action_key,
                'total_sample_count': int(len(rows)),
                'split_rows': split_rows,
                'candidate_eligible': bool(
                    strict_counts is not None
                    and np.all(counts >= required_counts)
                ),
            }
        )

    rows_by_split = {
        split_name: np.asarray(sorted(rows), dtype=int)
        for split_name, rows in assigned_rows.items()
    }
    combined = np.concatenate(list(rows_by_split.values()))
    if len(combined) != len(valid_rows) or len(np.unique(combined)) != len(combined):
        raise AssertionError(
            'Action-stratified split must assign every valid transition once'
        )
    if any(not len(rows) for rows in rows_by_split.values()):
        raise ValueError('Action-stratified split produced an empty partition')

    feasible = [
        assignment
        for assignment in action_assignments
        if assignment['candidate_eligible']
    ]
    feasible.sort(
        key=lambda item: (
            -item['total_sample_count'],
            item['action_key'],
        )
    )
    selected_candidates = feasible if top_k == 0 else feasible[:top_k]
    candidate_results = []
    audit_rows = []
    for rank, candidate in enumerate(selected_candidates, start=1):
        encoded = np.asarray(candidate['action_key'], dtype=int)
        support = {
            split_name: {
                'sample_count': int(len(candidate['split_rows'][split_name])),
                'episode_count': None,
                'episode_ids': [],
                'row_indices': candidate['split_rows'][split_name].tolist(),
            }
            for split_name in split_names
        }
        decoded = data.encoder.decode_action(encoded)
        candidate_results.append(
            {
                'rank_by_total_observed_frequency': int(rank),
                'encoded_control': encoded.tolist(),
                'action': decoded,
                'total_sample_count': int(candidate['total_sample_count']),
                'total_episode_count': None,
                'support': support,
            }
        )
        audit_rows.append(
            {
                'rank_by_total_observed_frequency': int(rank),
                **{f'control_{name}': value for name, value in decoded.items()},
                'total_sample_count': int(candidate['total_sample_count']),
                'train_sample_count': support['train']['sample_count'],
                'validation_sample_count': support['validation']['sample_count'],
                'test_sample_count': support['test']['sample_count'],
                'split_ratio_train': (
                    support['train']['sample_count']
                    / float(candidate['total_sample_count'])
                ),
                'split_ratio_validation': (
                    support['validation']['sample_count']
                    / float(candidate['total_sample_count'])
                ),
                'split_ratio_test': (
                    support['test']['sample_count']
                    / float(candidate['total_sample_count'])
                ),
            }
        )

    allocation_audit = []
    selected_keys = {item['action_key'] for item in selected_candidates}
    for assignment in action_assignments:
        decoded = data.encoder.decode_action(
            np.asarray(assignment['action_key'], dtype=int)
        )
        allocation_audit.append(
            {
                **{f'control_{name}': value for name, value in decoded.items()},
                'total_sample_count': int(assignment['total_sample_count']),
                'candidate_eligible': bool(
                    assignment['action_key'] in selected_keys
                ),
                **{
                    f'{split_name}_sample_count': int(
                        len(assignment['split_rows'][split_name])
                    )
                    for split_name in split_names
                },
            }
        )

    metadata = {
        'protocol': {
            'name': 'outcome_blind_action_stratified_row_holdout',
            'purpose': (
                'Allocate valid transition rows inside each exact joint action '
                'across train/validation/test without target columns'
            ),
            'assignment_inputs': (
                'encoded controls, configured split rates, and fixed random seed'
            ),
            'outcome_columns_used_for_assignment': [],
            'split_unit': 'individual valid transition row within exact joint action',
            'minimum_one_row_per_split_when_possible': True,
            'episode_overlap_between_splits_allowed': True,
            'row_overlap_allowed': False,
            'temporal_generalization_claim': False,
            'claim_boundary': (
                'Action-distribution-balanced supplementary prediction accuracy; '
                'not a future-time or episode-independent generalization claim'
            ),
            'random_state': int(random_state),
            'split_rate': [float(value) for value in split_rate],
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
        'total_observed_action_count': int(len(action_assignments)),
        'allocation_audit': allocation_audit,
        'availability_reason': (
            None
            if candidate_results
            else (
                'No joint action satisfies all configured train, validation, '
                'and test sample support minimums under the row-stratified split'
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
        dataset_rows=[],
    )
