"""Training, model validation, and independent closed-loop MPC evaluation."""

import json
import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from controllers.action_crossfit import (
    align_boundaries_to_control_episodes,
    chronological_action_split_support,
    control_episode_ids,
    evaluate_mpc_selected_action_crossfit,
)
from controllers.evaluation import (
    AffinePlantAdapter,
    PlantRollout,
    aggregate_constraint_metrics,
    constraint_prediction_metrics,
    energy_comparison,
    energy_wh,
    evaluate_recursive_rollouts,
    evaluate_observed_control_combinations,
    paired_bootstrap_energy_reduction,
    policy_metrics,
    power_prediction_metrics,
    reduction_distribution,
    rollout_plant,
    run_receding_horizon_policy,
    solver_metrics,
    state_prediction_metrics,
)
from controllers.candidate_pool_evaluation import (
    evaluate_fixed_action_candidate_pool_holdout,
)
from controllers.fixed_action_evaluation import (
    build_fixed_action_episode_holdout,
    evaluate_fixed_optimal_action_holdout,
)
from controllers.mixed_integer_mpc import MixedIntegerMPC
from data_provider.control_data import ControlData, load_control_data
from dynamics.affine import fit_affine_models
from dynamics.evaluator_base import history_for_rollout
from dynamics.evaluator_factory import (
    create_independent_evaluator,
    evaluator_label,
)


LOGGER = logging.getLogger('MPC.pipeline')


def _expanding_crossfit_ranges(
    total_rows: int,
    fold_count: int,
    initial_training_fraction: float,
) -> list[tuple[int, int]]:
    """Build non-overlapping future folds with an expanding past-only train set."""
    total_rows = int(total_rows)
    fold_count = int(fold_count)
    initial_fraction = float(initial_training_fraction)
    if fold_count <= 0:
        raise ValueError('Cross-fit fold count must be positive')
    if not 0.0 < initial_fraction < 1.0:
        raise ValueError('Cross-fit initial_training_fraction must be in (0, 1)')

    final_transition_end = total_rows - 1
    initial_end = max(2, int(np.floor(final_transition_end * initial_fraction)))
    remaining_transitions = final_transition_end - initial_end
    if remaining_transitions < fold_count:
        raise ValueError(
            'Not enough transitions after initial training for cross-fit folds'
        )
    edges = np.linspace(
        initial_end,
        final_transition_end,
        fold_count + 1,
        dtype=int,
    )
    ranges = [
        (int(edges[position]), int(edges[position + 1]))
        for position in range(fold_count)
        if int(edges[position + 1]) > int(edges[position])
    ]
    if len(ranges) != fold_count:
        raise ValueError('Cross-fit configuration produced an empty fold')
    return ranges


def _write_json(path: Path, value) -> None:
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _one_step_metrics(
    data: ControlData,
    plant,
    start: int,
    end: int,
    mape_min_actual_power_w: float,
) -> Dict[str, object]:
    predict_transitions = getattr(plant, 'predict_transitions', None)
    if predict_transitions is None:
        states, controls, disturbances, next_states, powers = data.model_arrays(
            start,
            end,
        )
        predicted_states = plant.predict_state(states, controls, disturbances)
        predicted_powers = plant.predict_power(states, controls, disturbances)
    else:
        predictions = predict_transitions(data, start, end)
        next_states = predictions.actual_states
        predicted_states = predictions.predicted_states
        powers = predictions.actual_powers
        predicted_powers = predictions.predicted_powers
    total_actual_power = powers.sum(axis=1, keepdims=True)
    total_predicted_power = predicted_powers.sum(axis=1, keepdims=True)
    return {
        'n_transitions': int(len(next_states)),
        'protocol': 'one-step prediction with measured state at every step',
        'state': state_prediction_metrics(
            next_states,
            predicted_states,
            data.state_names,
        ),
        'power': power_prediction_metrics(
            powers,
            predicted_powers,
            data.power_names,
            mape_min_actual_power_w,
        ),
        'total_power': power_prediction_metrics(
            total_actual_power,
            total_predicted_power,
            ('total_hvac_power_w',),
            mape_min_actual_power_w,
        ),
    }


def _select_evaluation_windows(
    data: ControlData,
    test_start: int,
    duration_steps: int,
    stride: int,
    max_windows: int,
    prediction_horizon: int,
    disturbance_forecast_method: str,
    end_exclusive: int | None = None,
) -> list[int]:
    if int(duration_steps) <= 0:
        raise ValueError('closed_loop_steps must be positive')
    if int(stride) <= 0:
        raise ValueError('evaluation stride must be positive')
    if int(max_windows) < 0:
        raise ValueError('max_windows must be zero (all) or a positive integer')
    last_start = len(data.states) - duration_steps - 1
    if end_exclusive is not None:
        last_start = min(
            last_start,
            int(end_exclusive) - duration_steps - 1,
        )
    starts = []
    for start in range(test_start, last_start + 1, int(stride)):
        if not data.transition_mask(start, start + duration_steps).all():
            continue
        if (
            disturbance_forecast_method == 'perfect'
            and start + duration_steps - 1 + prediction_horizon
            > len(data.disturbances)
        ):
            continue
        starts.append(start)
    if int(max_windows) > 0:
        starts = starts[: int(max_windows)]
    if not starts:
        raise ValueError('The selected split has no complete evaluation window')
    return starts


def _training_control_support(
    data: ControlData,
    train_end: int,
    logical_rules,
    restrict_to_training_support: bool,
    *,
    repeated_action_filter_enabled: bool = False,
    minimum_repeated_samples: int = 1,
    minimum_repeated_episodes: int = 1,
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    repeated_enabled = bool(repeated_action_filter_enabled)
    minimum_samples = int(minimum_repeated_samples)
    minimum_episodes = int(minimum_repeated_episodes)
    if repeated_enabled and min(minimum_samples, minimum_episodes) <= 0:
        raise ValueError('Enabled repeated-action minimums must be positive')

    recorded_controls = np.rint(data.controls).astype(int)
    observed = np.unique(recorded_controls[:train_end], axis=0)
    logically_valid = np.rint(
        data.encoder.valid_control_vectors(logical_rules)
    ).astype(int)
    episode_ids = control_episode_ids(data)
    category_metadata = data.control_preprocessing.get(
        'category_consolidation',
        {'enabled': False, 'mappings': {}},
    )
    consolidated_sources = {
        variable_name: {
            source
            for source, target in mapping.items()
            if source != target
        }
        for variable_name, mapping in category_metadata.get(
            'mappings', {}
        ).items()
    }

    allowed = []
    action_support = []
    repeated_eligible_count = 0
    canonical_combination_count = 0
    for vector in logically_valid:
        positions = np.flatnonzero(
            np.all(recorded_controls[:train_end] == vector, axis=1)
        )
        episodes = int(
            len(np.unique(episode_ids[positions])) if len(positions) else 0
        )
        decoded = data.encoder.decode_action(vector)
        canonical = not any(
            decoded.get(variable_name) in sources
            for variable_name, sources in consolidated_sources.items()
        )
        repeated_eligible = bool(
            len(positions) >= minimum_samples
            and episodes >= minimum_episodes
        )
        repeated_eligible_count += int(repeated_eligible)
        canonical_combination_count += int(canonical)

        reasons = []
        if bool(restrict_to_training_support) and not len(positions):
            reasons.append('not_observed_in_training')
        if repeated_enabled and not repeated_eligible:
            reasons.append('insufficient_repeated_training_support')
        if not canonical:
            reasons.append('non_representative_consolidated_category')
        is_allowed = not reasons
        if is_allowed:
            allowed.append(vector)
        action_support.append(
            {
                'encoded_control': vector.astype(int).tolist(),
                'action': decoded,
                'training_sample_count': int(len(positions)),
                'training_episode_count': episodes,
                'repeated_support_eligible': repeated_eligible,
                'category_representative': bool(canonical),
                'allowed_for_mpc': bool(is_allowed),
                'exclusion_reasons': reasons,
            }
        )

    if not allowed:
        raise ValueError(
            'Configured action filters leave no MPC control combination. '
            'Disable a filter, consolidate fewer categories, or lower the '
            'repeated-action sample/episode minimums.'
        )
    allowed = np.asarray(allowed, dtype=float)
    allowed_keys = {
        tuple(np.rint(vector).astype(int).tolist()) for vector in allowed
    }
    forbidden = np.asarray(
        [
            vector
            for vector in logically_valid
            if tuple(vector.tolist()) not in allowed_keys
        ],
        dtype=float,
    )
    if forbidden.size == 0:
        forbidden = np.empty((0, data.encoder.n_features))
    return allowed, forbidden, {
        'restriction_enabled': bool(restrict_to_training_support),
        'training_observed_combination_count': int(len(observed)),
        'logically_valid_combination_count': int(len(logically_valid)),
        'allowed_mpc_combination_count': int(len(allowed)),
        'logically_valid_but_unobserved_count': int(
            sum(
                item['training_sample_count'] == 0
                for item in action_support
            )
        ),
        'repeated_joint_action_filter': {
            'enabled': repeated_enabled,
            'minimum_training_samples': minimum_samples,
            'minimum_training_episodes': minimum_episodes,
            'eligible_combination_count': int(repeated_eligible_count),
        },
        'category_consolidation': {
            **category_metadata,
            'canonical_logically_valid_combination_count': int(
                canonical_combination_count
            ),
        },
        'forbidden_by_milp_no_good_cuts': int(len(forbidden)),
        'actions': action_support,
    }


def _minimum_defined(values: Sequence[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return min(defined) if defined else None


def _all_outputs_meet_minimum(
    values: Sequence[float | None],
    minimum: float,
) -> bool:
    return bool(values) and all(
        value is not None and float(value) >= minimum
        for value in values
    )


def _quality_assessment(
    cfg,
    affine_metrics: Dict[str, object],
    evaluator_metrics: Dict[str, object],
    affine_rollout: Dict[str, object],
    evaluator_rollout: Dict[str, object],
) -> tuple[Dict[str, object], list[str]]:
    affine_minimum = float(cfg.MPC.affine_models.minimum_test_power_r2)
    evaluator_minimum = float(
        cfg.MPC.independent_evaluator.minimum_test_power_r2
    )
    affine_maximum_wape = float(
        cfg.MPC.affine_models.maximum_rollout_power_wape_percent
    )
    affine_maximum_false_safe = float(
        cfg.MPC.affine_models.maximum_constraint_false_safe_rate_percent
    )
    evaluator_maximum_wape = float(
        cfg.MPC.independent_evaluator.maximum_rollout_power_wape_percent
    )
    evaluator_maximum_false_safe = float(
        cfg.MPC.independent_evaluator.maximum_constraint_false_safe_rate_percent
    )

    affine_r2_values = affine_metrics['test']['power']['r2_per_output']
    evaluator_r2_values = evaluator_metrics['test']['power']['r2_per_output']
    affine_observed = _minimum_defined(affine_r2_values)
    evaluator_observed = _minimum_defined(evaluator_r2_values)
    affine_rollout_wape = affine_rollout['power']['wape_percent']
    affine_false_safe = affine_rollout['constraint_prediction'][
        'maximum_state_false_safe_rate_percent'
    ]
    evaluator_rollout_wape = evaluator_rollout['power']['wape_percent']
    evaluator_false_safe = evaluator_rollout['constraint_prediction'][
        'maximum_state_false_safe_rate_percent'
    ]

    gates = {
        'affine_controller_power_r2': {
            'description': 'Every affine power output must meet test R2 minimum',
            'observed_minimum_r2': affine_observed,
            'required_minimum_r2': affine_minimum,
            'passed': _all_outputs_meet_minimum(
                affine_r2_values,
                affine_minimum,
            ),
        },
        'independent_evaluator_power_r2': {
            'description': (
                'Every independent evaluator power output must meet test R2 minimum'
            ),
            'observed_minimum_r2': evaluator_observed,
            'required_minimum_r2': evaluator_minimum,
            'passed': _all_outputs_meet_minimum(
                evaluator_r2_values,
                evaluator_minimum,
            ),
        },
        'affine_controller_recursive_power_wape': {
            'description': (
                'Affine recursive recorded-control rollout WAPE must not exceed maximum'
            ),
            'observed_wape_percent': affine_rollout_wape,
            'required_maximum_wape_percent': affine_maximum_wape,
            'passed': (
                affine_rollout_wape is not None
                and affine_rollout_wape <= affine_maximum_wape
            ),
        },
        'affine_controller_false_safe': {
            'description': (
                'Affine maximum false-safe rate among states with actual violations'
            ),
            'observed_maximum_false_safe_rate_percent': affine_false_safe,
            'required_maximum_false_safe_rate_percent': affine_maximum_false_safe,
            'status': (
                'not_testable_no_actual_violations'
                if affine_false_safe is None
                else 'evaluated'
            ),
            'passed': (
                affine_false_safe is None
                or affine_false_safe <= affine_maximum_false_safe
            ),
        },
        'independent_evaluator_recursive_power_wape': {
            'description': (
                'Evaluator recursive recorded-control rollout WAPE must not exceed maximum'
            ),
            'observed_wape_percent': evaluator_rollout_wape,
            'required_maximum_wape_percent': evaluator_maximum_wape,
            'passed': (
                evaluator_rollout_wape is not None
                and evaluator_rollout_wape <= evaluator_maximum_wape
            ),
        },
        'independent_evaluator_false_safe': {
            'description': (
                'Evaluator maximum false-safe rate among states with actual violations'
            ),
            'observed_maximum_false_safe_rate_percent': evaluator_false_safe,
            'required_maximum_false_safe_rate_percent': evaluator_maximum_false_safe,
            'status': (
                'not_testable_no_actual_violations'
                if evaluator_false_safe is None
                else 'evaluated'
            ),
            'passed': (
                evaluator_false_safe is None
                or evaluator_false_safe <= evaluator_maximum_false_safe
            ),
        },
    }
    warnings = []
    for name, gate in gates.items():
        if not gate['passed']:
            warnings.append(f'quality gate failed: {name}')
    if affine_false_safe is None or evaluator_false_safe is None:
        warnings.append(
            'one or more false-safe gates were not testable because no actual '
            'constraint violations occurred in the selected rollout windows'
        )
    all_passed = all(bool(gate['passed']) for gate in gates.values())
    return {
        'model_validation_passed': all_passed,
        'interpretation': (
            'Controller and evaluator models passed configured offline gates'
            if all_passed
            else (
                'Model validation failed; policy results are exploratory even '
                'before policy acceptance and real-equipment validation'
            )
        ),
        'gates': gates,
    }, warnings


def _aggregate_policy_results(
    window_results: Sequence[Dict[str, object]],
    primary_baseline: str,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    seed: int,
) -> Dict[str, object]:
    policy_names = ('recorded', 'hold_last', 'mpc')
    energies = {
        policy: np.asarray(
            [window['policies'][policy]['energy_wh'] for window in window_results],
            dtype=float,
        )
        for policy in policy_names
    }
    if primary_baseline not in ('recorded', 'hold_last'):
        raise ValueError("primary_baseline must be 'recorded' or 'hold_last'")

    baseline = energies[primary_baseline]
    candidate = energies['mpc']
    comparison = energy_comparison(float(baseline.sum()), float(candidate.sum()))
    comparison.update(
        {
            'aggregation': 'ratio of summed paired-window energies',
            'window_reduction_percent_distribution': reduction_distribution(
                baseline,
                candidate,
            ),
            'bootstrap_confidence_interval': paired_bootstrap_energy_reduction(
                baseline,
                candidate,
                bootstrap_samples,
                bootstrap_confidence,
                seed,
            ),
        }
    )

    return {
        'primary_baseline': primary_baseline,
        'policy_energy_wh': {
            policy: {
                'total': float(values.sum()),
                'mean_per_window': float(values.mean()),
                'median_per_window': float(np.median(values)),
            }
            for policy, values in energies.items()
        },
        'mpc_vs_primary_baseline': comparison,
        'constraint_violations': {
            policy: aggregate_constraint_metrics(
                [
                    window['policies'][policy]['constraint_violations']
                    for window in window_results
                ]
            )
            for policy in policy_names
        },
    }


def _policy_acceptance(
    model_quality: Dict[str, object],
    policy_aggregate: Dict[str, object],
    aggregate_solver: Dict[str, object],
    evaluated_windows: int,
    minimum_windows: int,
    official_mape_evaluation: Dict[str, object],
) -> tuple[Dict[str, object], list[str]]:
    comparison = policy_aggregate['mpc_vs_primary_baseline']
    confidence_interval = comparison['bootstrap_confidence_interval']
    confidence_lower = confidence_interval['lower_percent']
    baseline_name = policy_aggregate['primary_baseline']
    constraints = policy_aggregate['constraint_violations']
    baseline_constraints = constraints[baseline_name]
    mpc_constraints = constraints['mpc']
    constraint_nonworse = all(
        mpc_constraints[state_name]['violation_magnitude_minutes']
        <= baseline_constraints[state_name]['violation_magnitude_minutes'] + 1e-8
        and mpc_constraints[state_name]['maximum_violation']
        <= baseline_constraints[state_name]['maximum_violation'] + 1e-8
        for state_name in baseline_constraints
    )

    gates = {
        'minimum_evaluation_windows': {
            'observed': int(evaluated_windows),
            'required_minimum': int(minimum_windows),
            'passed': int(evaluated_windows) >= int(minimum_windows),
        },
        'positive_energy_reduction_confidence': {
            'observed_bootstrap_lower_percent': confidence_lower,
            'confidence_level': confidence_interval['confidence_level'],
            'required': 'paired-bootstrap lower bound > 0%',
            'passed': confidence_lower is not None and confidence_lower > 0.0,
        },
        'constraint_non_degradation': {
            'baseline': baseline_name,
            'required': (
                'MPC violation magnitude-minutes and maximum violation must '
                'not exceed the primary baseline for any state'
            ),
            'passed': bool(constraint_nonworse),
        },
        'solver_fallback': {
            'observed_count': aggregate_solver['fallback_count'],
            'required_maximum_count': 0,
            'passed': aggregate_solver['fallback_count'] == 0,
        },
        'solver_deadline': {
            'observed_miss_count': aggregate_solver['deadline_miss_count'],
            'required_maximum_count': 0,
            'passed': aggregate_solver['deadline_miss_count'] == 0,
        },
        'training_support_ood': {
            'observed_count': aggregate_solver['training_support_ood_count'],
            'required_maximum_count': 0,
            'passed': aggregate_solver['training_support_ood_count'] == 0,
        },
        'official_actual_y_mape': {
            'observed_reportable': bool(
                official_mape_evaluation.get('reportable', False)
            ),
            'observed_value': (
                official_mape_evaluation.get('primary_metric') or {}
            ).get('value'),
            'required': (
                'The validation-selected exact-action pool and its frozen '
                'prediction model must satisfy pooled and per-action test '
                'valid-MAPE-sample gates'
            ),
            'passed': bool(
                official_mape_evaluation.get('reportable', False)
            ),
        },
        'official_mape_target': {
            'observed_value_percent': (
                official_mape_evaluation.get('primary_metric') or {}
            ).get('value'),
            'required_maximum_percent': (
                official_mape_evaluation.get('primary_metric') or {}
            ).get('target_percent'),
            'required': 'Reportable official MAPE must be at most the configured target',
            'passed': bool(
                (official_mape_evaluation.get('primary_metric') or {}).get(
                    'target_achieved',
                    False,
                )
            ),
        },
    }
    policy_passed = all(bool(gate['passed']) for gate in gates.values())
    simulated_passed = bool(model_quality['model_validation_passed']) and policy_passed
    warnings = [
        f'policy acceptance gate failed: {name}'
        for name, gate in gates.items()
        if not gate['passed']
    ]
    return {
        'model_validation': model_quality,
        'simulated_policy_acceptance': {
            'passed': simulated_passed,
            'interpretation': (
                'Accepted as an offline simulated policy result'
                if simulated_passed
                else 'Exploratory offline policy result; acceptance gates failed'
            ),
            'gates': gates,
        },
        'real_equipment_performance_claim': {
            'passed': False,
            'status': 'not_tested',
            'reason': 'Optimized actions have not been applied to real equipment',
        },
    }, warnings


def run_mpc_pipeline(cfg, result_dir: Path):
    """Fit affine control models and evaluate MPC on an independent plant."""
    np.random.seed(int(cfg.DEFAULT.seed))
    if int(cfg.MPC.sample_interval_minutes) != 1:
        raise ValueError('This MPC formulation requires one-minute samples')
    result_dir = Path(result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info('Loading chronological control data')
    data = load_control_data(cfg)
    evaluation_cfg = cfg.MPC.evaluation
    action_options_cfg = evaluation_cfg.action_options
    repeated_action_cfg = action_options_cfg.repeated_joint_actions
    episode_group_cfg = action_options_cfg.episode_group_split
    official_episode_cfg = (
        action_options_cfg.exclude_insufficient_episode_actions
    )
    action_split_cfg = evaluation_cfg.action_split_support
    nominal_train_end, nominal_test_start = data.split_indices(
        cfg.DATASET.split_rate
    )
    train_end, test_start = nominal_train_end, nominal_test_start
    split_alignment = {
        'enabled': bool(episode_group_cfg.enabled),
        'nominal_train_end_exclusive': int(nominal_train_end),
        'nominal_test_start': int(nominal_test_start),
        'applied': False,
        'fallback_reason': None,
    }
    if split_alignment['enabled']:
        try:
            train_end, test_start = align_boundaries_to_control_episodes(
                data,
                [nominal_train_end, nominal_test_start],
            )
            split_alignment['applied'] = True
        except ValueError as error:
            split_alignment['fallback_reason'] = str(error)
            LOGGER.warning(
                'Could not align chronological split to complete control '
                'episodes; nominal boundaries retained: %s',
                error,
            )
    split_alignment.update(
        {
            'actual_train_end_exclusive': int(train_end),
            'actual_test_start': int(test_start),
            'train_boundary_shift_steps': int(train_end - nominal_train_end),
            'test_boundary_shift_steps': int(test_start - nominal_test_start),
        }
    )
    action_split_support, action_split_support_rows = (
        chronological_action_split_support(
            data,
            train_end,
            test_start,
            minimum_samples_per_split=(
                action_split_cfg.minimum_samples_per_split
            ),
            minimum_complete_episodes_per_split=(
                action_split_cfg.minimum_complete_episodes_per_split
            ),
            require_complete_episode_minimum=bool(
                official_episode_cfg.enabled
            ),
        )
    )
    mape_threshold = float(cfg.MPC.affine_models.mape_min_actual_power_w)
    LOGGER.info(
        'Data loaded: rows=%d | period=%s to %s | train=%d | validation=%d | test=%d',
        len(data.states),
        data.timestamps.iloc[0].isoformat(),
        data.timestamps.iloc[-1].isoformat(),
        train_end,
        test_start - train_end,
        len(data.states) - test_start,
    )
    LOGGER.info(
        'Joint-action split support: observed=%d | present_in_all=%d | '
        'eligible_in_all=%d | cross_split_episodes=%d',
        action_split_support['observed_action_count'],
        action_split_support['present_in_all_splits_action_count'],
        action_split_support['eligible_in_all_splits_action_count'],
        action_split_support['cross_split_episode_count'],
    )
    LOGGER.info(
        'Action options: repeated_joint_actions=%s | '
        'category_consolidation=%s | episode_group_split=%s | '
        'exclude_insufficient_episode_actions=%s',
        bool(repeated_action_cfg.enabled),
        bool(
            data.control_preprocessing['category_consolidation']['enabled']
        ),
        bool(episode_group_cfg.enabled),
        bool(official_episode_cfg.enabled),
    )

    LOGGER.info('Fitting affine state and power models for MPC optimization')
    affine_training = data.model_arrays(0, train_end - 1)
    state_model, power_model = fit_affine_models(
        *affine_training,
        state_names=data.state_names,
        control_feature_names=data.encoder.feature_names,
        disturbance_names=data.disturbance_names,
        power_names=data.power_names,
        state_alpha=cfg.MPC.affine_models.state_ridge_alpha,
        power_alpha=cfg.MPC.affine_models.power_ridge_alpha,
    )
    affine_plant = AffinePlantAdapter(state_model, power_model)
    affine_metrics = {
        'train': _one_step_metrics(
            data,
            affine_plant,
            0,
            train_end - 1,
            mape_threshold,
        ),
        'validation': _one_step_metrics(
            data,
            affine_plant,
            train_end,
            test_start - 1,
            mape_threshold,
        ),
        'test': _one_step_metrics(
            data,
            affine_plant,
            test_start,
            len(data.states) - 1,
            mape_threshold,
        ),
    }
    LOGGER.info(
        'Affine models ready: training_transitions=%d | test_power_r2=%s | '
        'test_power_mape_percent=%s',
        len(affine_training[0]),
        affine_metrics['test']['power']['r2_per_output'],
        affine_metrics['test']['power']['mape_percent_per_output'],
    )

    evaluator_cfg = cfg.MPC.independent_evaluator
    LOGGER.info(
        'Fitting independent evaluator: family=%s | training_scope=train+validation',
        evaluator_cfg.model_family,
    )
    independent_evaluator = create_independent_evaluator(
        evaluator_cfg,
        data,
        test_start,
        random_state=cfg.DEFAULT.seed,
    )
    active_evaluator_label = evaluator_label(independent_evaluator)
    evaluator_metadata = independent_evaluator.to_metadata()
    evaluator_device = evaluator_metadata.get('training_summary', {}).get(
        'device', 'cpu'
    )
    evaluator_metrics = {
        'test': _one_step_metrics(
            data,
            independent_evaluator,
            test_start,
            len(data.states) - 1,
            mape_threshold,
        )
    }
    LOGGER.info(
        'Independent evaluator ready: label=%s | device=%s | test_power_r2=%s | '
        'test_power_mape_percent=%s',
        active_evaluator_label,
        evaluator_device,
        evaluator_metrics['test']['power']['r2_per_output'],
        evaluator_metrics['test']['power']['mape_percent_per_output'],
    )

    fixed_action_cfg = evaluation_cfg.fixed_optimal_action_holdout
    fixed_action_rows = []
    fixed_action_split_rows = []
    fixed_action_model_comparison_rows = []
    fixed_action_candidate_dataset_rows = []
    if bool(fixed_action_cfg.enabled):
        split_strategy = str(fixed_action_cfg.split_strategy)
        if split_strategy != 'action_episode_grouped':
            raise ValueError(
                'MPC.evaluation.fixed_optimal_action_holdout.split_strategy '
                "must be 'action_episode_grouped'"
            )
        LOGGER.info(
            'Building outcome-blind fixed-action holdout: strategy=%s | top_k=%d',
            split_strategy,
            int(fixed_action_cfg.top_k_actions),
        )
        fixed_action_holdout = build_fixed_action_episode_holdout(
            data,
            fixed_action_cfg,
            split_rate=cfg.DATASET.split_rate,
        )
        fixed_action_split_rows = fixed_action_holdout.audit_rows
        fixed_action_candidate_dataset_rows = fixed_action_holdout.dataset_rows
        fixed_family = (
            fixed_action_cfg.get('model_family')
            or evaluator_cfg.model_family
        )
        LOGGER.info(
            'Fitting fixed-action selector: family=%s | grouped_train_rows=%d',
            fixed_family,
            len(fixed_action_holdout.train_rows),
        )
        fixed_selection_evaluator = create_independent_evaluator(
            evaluator_cfg,
            data,
            train_end,
            random_state=int(cfg.DEFAULT.seed) + 100,
            model_family=str(fixed_family),
            training_rows=fixed_action_holdout.train_rows,
        )
        fixed_selection_label = evaluator_label(fixed_selection_evaluator)
        fixed_report_training_rows = np.unique(
            np.concatenate(
                [
                    fixed_action_holdout.train_rows,
                    fixed_action_holdout.validation_rows,
                ]
            )
        )
        LOGGER.info(
            'Fitting frozen fixed-action report model: family=%s | '
            'grouped_train_validation_rows=%d',
            fixed_family,
            len(fixed_report_training_rows),
        )
        fixed_report_evaluator = create_independent_evaluator(
            evaluator_cfg,
            data,
            test_start,
            random_state=int(cfg.DEFAULT.seed) + 101,
            model_family=str(fixed_family),
            training_rows=fixed_report_training_rows,
        )
        fixed_report_label = evaluator_label(fixed_report_evaluator)
        (
            fixed_action_evaluation,
            fixed_action_rows,
        ) = evaluate_fixed_optimal_action_holdout(
            data,
            fixed_action_holdout,
            fixed_selection_evaluator,
            fixed_report_evaluator,
            state_constraints=cfg.MPC.state_constraints,
            minimum_predicted_state_satisfaction_percent=(
                fixed_action_cfg.minimum_predicted_state_satisfaction_percent
            ),
            allow_soft_constraint_fallback=bool(
                fixed_action_cfg.allow_soft_constraint_fallback
            ),
            minimum_test_exact_matches=(
                fixed_action_cfg.minimum_test_exact_matches
            ),
            minimum_test_episodes=(
                fixed_action_cfg.minimum_test_episodes_per_candidate
            ),
            minimum_test_mape_samples=(
                fixed_action_cfg.minimum_test_mape_samples
            ),
            mape_min_actual_power_w=mape_threshold,
            selection_evaluator_name=fixed_selection_label,
            frozen_report_evaluator_name=fixed_report_label,
            total_power_report_config=(
                fixed_action_cfg.total_power_report_model
            ),
            selection_preference=(
                fixed_action_cfg.selection_preference
            ),
            total_power_random_state=int(cfg.DEFAULT.seed) + 102,
            target_mape_percent=float(
                fixed_action_cfg.total_power_report_model.target_mape_percent
            ),
        )
        fixed_action_model_comparison_rows = (
            fixed_action_evaluation.get('total_power_model_selection') or {}
        ).get('validation_results', [])
        candidate_split_counts = {
            split_name: int(
                sum(
                    row['split'] == split_name
                    for row in fixed_action_candidate_dataset_rows
                )
            )
            for split_name in ('train', 'validation', 'test')
        }
        evaluated_candidates = (
            fixed_action_evaluation.get('selection') or {}
        ).get('candidates', [])
        fixed_action_evaluation['candidate_datasets'] = {
            'candidate_count': int(len(evaluated_candidates)),
            'all_preferred_values_candidate_count': int(sum(
                bool(item.get('matches_all_preferred_values'))
                for item in evaluated_candidates
            )),
            'row_count': int(len(fixed_action_candidate_dataset_rows)),
            'split_row_counts': candidate_split_counts,
            'row_semantics': (
                'One-step exact-action transitions: state/disturbance/control '
                'at t with power and next-state targets'
            ),
            'episode_overlap_between_splits_allowed': False,
            'artifacts': {
                'all': 'fixed_action_candidate_dataset_all.csv',
                'train': 'fixed_action_candidate_dataset_train.csv',
                'validation': 'fixed_action_candidate_dataset_validation.csv',
                'test': 'fixed_action_candidate_dataset_test.csv',
            },
        }
        LOGGER.info(
            'Fixed-action candidate datasets ready: candidates=%d | '
            'all_preferred=%d | split_rows=%s',
            len(evaluated_candidates),
            fixed_action_evaluation['candidate_datasets'][
                'all_preferred_values_candidate_count'
            ],
            candidate_split_counts,
        )
        fixed_action_evaluation['selection_model_metadata'] = {
            **fixed_selection_evaluator.to_metadata(),
            'training_row_protocol': 'explicit action-episode grouped train rows',
            'training_row_count': int(len(fixed_action_holdout.train_rows)),
        }
        fixed_action_evaluation['report_model_metadata'] = {
            **fixed_report_evaluator.to_metadata(),
            'training_row_protocol': (
                'explicit action-episode grouped train + validation rows'
            ),
            'training_row_count': int(len(fixed_report_training_rows)),
        }
        fixed_selection = fixed_action_evaluation.get('selection') or {}
        if fixed_selection.get('soft_constraint_fallback_used'):
            LOGGER.warning(
                'Fixed optimal-action MAPE used the soft state-constraint '
                'fallback: selection_mode=%s | validation_satisfaction_percent=%s '
                '| strict policy-acceptance gate will fail',
                fixed_selection.get('selection_mode'),
                (fixed_selection.get('selected_validation_score') or {}).get(
                    'predicted_state_constraint_satisfaction_percent'
                ),
            )
        elif not fixed_action_evaluation['reportable']:
            LOGGER.warning(
                'Fixed optimal-action MAPE is not reportable: %s',
                fixed_action_evaluation.get('reason'),
            )
        LOGGER.info(
            'Fixed optimal-action holdout complete: reportable=%s | '
            'total_power_mape_percent=%s | selected_action=%s | '
            'exact_test_matches=%s | total_power_model=%s | target_achieved=%s',
            fixed_action_evaluation['reportable'],
            fixed_action_evaluation['primary_metric']['value'],
            fixed_action_evaluation['selection']['selected_action'],
            (fixed_action_evaluation.get('test_evaluation') or {}).get(
                'exact_match_count'
            ),
            (
                fixed_action_evaluation.get('total_power_model_selection') or {}
            ).get('selected_family'),
            (
                fixed_action_evaluation.get('primary_metric') or {}
            ).get('target_achieved'),
        )
    else:
        fixed_action_evaluation = {
            'available': False,
            'reportable': False,
            'reason': (
                'Disabled by '
                'MPC.evaluation.fixed_optimal_action_holdout.enabled'
            ),
            'protocol': {
                'name': 'fixed_optimal_joint_action_frozen_model_holdout_mape'
            },
            'split': None,
            'selection': None,
            'primary_metric': {
                'name': 'fixed_optimal_action_total_power_mape_percent',
                'value': None,
                'unit': 'percent',
                'reportable': False,
                'diagnostic_value': None,
            },
            'test_evaluation': None,
        }

    candidate_pool_cfg = evaluation_cfg.fixed_action_candidate_pool
    candidate_pool_rows = []
    candidate_pool_model_comparison_rows = []
    if bool(candidate_pool_cfg.enabled):
        if not bool(fixed_action_cfg.enabled):
            raise ValueError(
                'MPC.evaluation.fixed_action_candidate_pool requires '
                'fixed_optimal_action_holdout.enabled=true'
            )
        LOGGER.info(
            'Running validation-selected candidate-pool MAPE: '
            'candidate_count=%d | maximum_validation_mape_percent=%.3f | '
            'maximum_candidates=%d',
            len(fixed_action_holdout.metadata['candidates']),
            float(
                candidate_pool_cfg
                .maximum_validation_mape_percent_per_candidate
            ),
            int(candidate_pool_cfg.maximum_candidates),
        )
        (
            candidate_pool_evaluation,
            candidate_pool_rows,
        ) = evaluate_fixed_action_candidate_pool_holdout(
            data,
            fixed_action_holdout,
            config=candidate_pool_cfg,
            total_power_report_config=(
                fixed_action_cfg.total_power_report_model
            ),
            mape_min_actual_power_w=mape_threshold,
            random_state=int(cfg.DEFAULT.seed) + 200,
        )
        candidate_pool_model_comparison_rows = (
            candidate_pool_evaluation.get('total_power_model_selection') or {}
        ).get('validation_results', [])
        candidate_pool_evaluation['candidate_datasets'] = (
            fixed_action_evaluation.get('candidate_datasets')
        )
        LOGGER.info(
            'Candidate-pool holdout complete: reportable=%s | '
            'selected_candidates=%d/%d | pooled_test_rows=%s | '
            'pooled_mape_percent=%s | macro_mape_percent=%s | '
            'worst_action_mape_percent=%s | target_achieved=%s',
            candidate_pool_evaluation['reportable'],
            (
                candidate_pool_evaluation.get('selection') or {}
            ).get('selected_candidate_count', 0),
            (
                candidate_pool_evaluation.get('selection') or {}
            ).get('candidate_count', 0),
            (
                candidate_pool_evaluation.get('test_evaluation') or {}
            ).get('pool_exact_match_count'),
            (
                candidate_pool_evaluation.get('primary_metric') or {}
            ).get('value'),
            (
                candidate_pool_evaluation.get('test_evaluation') or {}
            ).get('equal_action_macro_mape_percent'),
            (
                candidate_pool_evaluation.get('test_evaluation') or {}
            ).get('worst_action_mape_percent'),
            (
                candidate_pool_evaluation.get('primary_metric') or {}
            ).get('target_achieved'),
        )
    else:
        candidate_pool_evaluation = {
            'available': False,
            'reportable': False,
            'reason': (
                'Disabled by '
                'MPC.evaluation.fixed_action_candidate_pool.enabled'
            ),
            'protocol': {
                'name': 'validation_selected_fixed_action_pool_holdout_mape'
            },
            'selection': None,
            'total_power_model_selection': None,
            'primary_metric': {
                'name': 'fixed_action_candidate_pool_total_power_mape_percent',
                'value': None,
                'unit': 'percent',
                'reportable': False,
                'diagnostic_value': None,
                'target_percent': float(
                    candidate_pool_cfg.target_mape_percent
                ),
                'target_achieved': False,
            },
            'test_evaluation': None,
        }
    official_mape_evaluation = (
        candidate_pool_evaluation
        if bool(candidate_pool_cfg.enabled)
        else fixed_action_evaluation
    )
    observed_combination_cfg = evaluation_cfg.observed_combination
    observed_combination_rows = []
    if bool(observed_combination_cfg.enabled):
        LOGGER.info(
            'Fitting observed-combination selector: family=%s | '
            'training_scope=train-only',
            evaluator_cfg.model_family,
        )
        selection_evaluator = create_independent_evaluator(
            evaluator_cfg,
            data,
            train_end,
            random_state=cfg.DEFAULT.seed,
        )
        selection_evaluator_name = evaluator_label(selection_evaluator)
        minimum_validation_samples = int(
            observed_combination_cfg.minimum_validation_samples_per_combination
        )
        minimum_state_satisfaction = float(
            observed_combination_cfg.minimum_predicted_state_satisfaction_percent
        )
        minimum_test_exact_matches = int(
            observed_combination_cfg.minimum_test_exact_matches
        )
        minimum_test_mape_samples = int(
            observed_combination_cfg.minimum_test_mape_samples
        )
        LOGGER.info(
            'Running observed-combination holdout MAPE: '
            'minimum_validation_samples=%d | '
            'minimum_predicted_state_satisfaction_percent=%.3f',
            minimum_validation_samples,
            minimum_state_satisfaction,
        )
        (
            observed_combination_evaluation,
            observed_combination_rows,
        ) = evaluate_observed_control_combinations(
            data,
            selection_evaluator,
            independent_evaluator,
            validation_start=train_end,
            test_start=test_start,
            test_end=len(data.states) - 1,
            state_constraints=cfg.MPC.state_constraints,
            minimum_validation_samples=minimum_validation_samples,
            minimum_predicted_state_satisfaction_percent=(
                minimum_state_satisfaction
            ),
            minimum_test_exact_matches=minimum_test_exact_matches,
            minimum_test_mape_samples=minimum_test_mape_samples,
            mape_min_actual_power_w=mape_threshold,
            selection_evaluator_name=selection_evaluator_name,
            frozen_report_evaluator_name=active_evaluator_label,
        )
        observed_combination_evaluation['selection_model_metadata'] = (
            selection_evaluator.to_metadata()
        )
        primary_metric = (
            observed_combination_evaluation.get('test_evaluation') or {}
        ).get('primary_metric', {})
        LOGGER.info(
            'Observed-combination MAPE complete: reportable=%s | '
            'total_power_mape_percent=%s | exact_test_matches=%s',
            observed_combination_evaluation['reportable'],
            primary_metric.get('value'),
            (observed_combination_evaluation.get('test_evaluation') or {}).get(
                'exact_match_count'
            ),
        )
    else:
        observed_combination_evaluation = {
            'available': False,
            'reportable': False,
            'reason': 'Disabled by MPC.evaluation.observed_combination.enabled',
            'protocol': {
                'name': 'observed_control_combination_holdout_mape',
            },
            'selection': None,
            'test_evaluation': None,
        }
    duration_steps = int(evaluation_cfg.closed_loop_steps)
    starts = _select_evaluation_windows(
        data,
        test_start,
        duration_steps,
        evaluation_cfg.stride,
        evaluation_cfg.max_windows,
        cfg.MPC.prediction_horizon,
        cfg.MPC.disturbance_forecast,
    )
    lead_times = [int(value) for value in evaluation_cfg.lead_time_steps]
    LOGGER.info(
        'Evaluation windows selected: count=%d | duration_steps=%d | stride=%d | '
        'disturbance_forecast=%s',
        len(starts),
        duration_steps,
        int(evaluation_cfg.stride),
        cfg.MPC.disturbance_forecast,
    )
    LOGGER.info('Running recursive recorded-control model validation')
    affine_rollout_metrics = evaluate_recursive_rollouts(
        affine_plant,
        data,
        starts,
        duration_steps,
        cfg.MPC.state_constraints,
        mape_threshold,
        lead_times,
    )
    evaluator_rollout_metrics = evaluate_recursive_rollouts(
        independent_evaluator,
        data,
        starts,
        duration_steps,
        cfg.MPC.state_constraints,
        mape_threshold,
        lead_times,
    )
    LOGGER.info(
        'Recursive validation complete: affine_power_wape_percent=%s | '
        'evaluator_power_wape_percent=%s',
        affine_rollout_metrics['power']['wape_percent'],
        evaluator_rollout_metrics['power']['wape_percent'],
    )

    configured_candidates = list(
        evaluator_cfg.get('validation_model_families', [])
    )
    evaluator_model_comparison = {}
    if configured_candidates:
        validation_starts = _select_evaluation_windows(
            data,
            train_end,
            duration_steps,
            evaluation_cfg.stride,
            evaluator_cfg.get('validation_max_windows', 0),
            cfg.MPC.prediction_horizon,
            cfg.MPC.disturbance_forecast,
            end_exclusive=test_start,
        )
        comparison_families = [
            str(evaluator_cfg.model_family),
            *[str(value) for value in configured_candidates],
        ]
        seen_labels = set()
        for candidate_family in comparison_families:
            LOGGER.info(
                'Fitting evaluator validation candidate: family=%s | '
                'training_scope=train-only',
                candidate_family,
            )
            candidate = create_independent_evaluator(
                evaluator_cfg,
                data,
                train_end,
                random_state=cfg.DEFAULT.seed,
                model_family=candidate_family,
            )
            candidate_label = evaluator_label(candidate)
            if candidate_label in seen_labels:
                continue
            seen_labels.add(candidate_label)
            candidate_metrics = {
                'validation': _one_step_metrics(
                    data,
                    candidate,
                    train_end,
                    test_start - 1,
                    mape_threshold,
                )
            }
            candidate_rollout = evaluate_recursive_rollouts(
                candidate,
                data,
                validation_starts,
                duration_steps,
                cfg.MPC.state_constraints,
                mape_threshold,
                lead_times,
            )
            evaluator_model_comparison[candidate_label] = {
                'selected_for_final_policy_evaluation': (
                    candidate_label == active_evaluator_label
                ),
                'metadata': candidate.to_metadata(),
                'one_step': candidate_metrics,
                'recursive_validation_rollout': candidate_rollout,
            }

    quality_assessment, quality_warnings = _quality_assessment(
        cfg,
        affine_metrics,
        evaluator_metrics,
        affine_rollout_metrics,
        evaluator_rollout_metrics,
    )
    fail_on_low_quality = bool(
        cfg.MPC.affine_models.fail_on_low_quality
        or cfg.MPC.independent_evaluator.fail_on_low_quality
    )
    if (
        not quality_assessment['model_validation_passed']
        and fail_on_low_quality
    ):
        raise RuntimeError('; '.join(quality_warnings))
    for warning in quality_warnings:
        LOGGER.warning('%s', warning)
    if not action_split_support['balanced_holdout_available']:
        support_minimum_label = (
            'sample and complete-episode minimums'
            if bool(official_episode_cfg.enabled)
            else 'sample minimums'
        )
        support_warning = (
            'joint-action split support gate failed: no action meets the '
            f'configured {support_minimum_label} in all of '
            'train/validation/test; combination-balanced holdout claims are '
            'not reportable'
        )
        quality_warnings.append(support_warning)
        LOGGER.warning('%s', support_warning)

    observed_support, forbidden_controls, support_metadata = (
        _training_control_support(
            data,
            train_end,
            cfg.MPC.logical_rules,
            evaluation_cfg.restrict_to_training_control_support,
            repeated_action_filter_enabled=(
                repeated_action_cfg.enabled
            ),
            minimum_repeated_samples=(
                repeated_action_cfg.minimum_training_samples
            ),
            minimum_repeated_episodes=(
                repeated_action_cfg.minimum_training_episodes
            ),
        )
    )
    LOGGER.info(
        'Control support: observed=%d | logically_valid=%d | allowed=%d | '
        'forbidden=%d',
        support_metadata['training_observed_combination_count'],
        support_metadata['logically_valid_combination_count'],
        support_metadata['allowed_mpc_combination_count'],
        support_metadata['forbidden_by_milp_no_good_cuts'],
    )
    LOGGER.info(
        'Building mixed-integer controller: solver_time_limit=%.3fs | '
        'mip_gap=%.6f',
        float(cfg.MPC.solver.time_limit_seconds),
        float(cfg.MPC.solver.mip_relative_gap),
    )
    controller = MixedIntegerMPC(
        state_model=state_model,
        power_model=power_model,
        encoder=data.encoder,
        state_constraints=cfg.MPC.state_constraints,
        logical_rules=cfg.MPC.logical_rules,
        prediction_horizon=cfg.MPC.prediction_horizon,
        control_horizon=cfg.MPC.control_horizon,
        switch_penalty=cfg.MPC.objective.switch_penalty,
        time_limit_seconds=cfg.MPC.solver.time_limit_seconds,
        mip_relative_gap=cfg.MPC.solver.mip_relative_gap,
        forbidden_control_vectors=forbidden_controls,
    )

    model_validation = {
        'data_split': {
            'train_end_exclusive': int(train_end),
            'validation_start': int(train_end),
            'test_start': int(test_start),
            'test_end_exclusive': int(len(data.states)),
            'split_rate': list(cfg.DATASET.split_rate),
            'boundary_alignment': split_alignment,
            'joint_action_support': action_split_support,
        },
        'affine_controller_model': {
            'one_step': affine_metrics,
            'recursive_recorded_control_rollout': affine_rollout_metrics,
        },
        'independent_nonlinear_evaluator': {
            'metadata': independent_evaluator.to_metadata(),
            'one_step': evaluator_metrics,
            'recursive_recorded_control_rollout': evaluator_rollout_metrics,
        },
        'evaluator_model_comparison': {
            'requested': bool(configured_candidates),
            'selection_protocol': (
                'When candidates are requested, every candidate is fitted on '
                'train only and compared on validation only. The configured '
                'active family is then fitted on train + validation for final '
                'test and policy evaluation; selection is never automated on test.'
            ),
            'models': evaluator_model_comparison,
        },
        'quality_assessment': quality_assessment,
        'quality_warnings': quality_warnings,
    }

    window_results = []
    first_action_rows = []
    closed_loop_step_rows = []
    all_solver_records = []
    mpc_evaluator_power_windows = []
    mpc_affine_power_windows = []
    mpc_control_windows = []
    mpc_state_context_windows = []
    mpc_disturbance_context_windows = []
    for window_number, start in enumerate(starts, start=1):
        timestamp = data.timestamps.iloc[start].isoformat()
        LOGGER.info(
            'MPC closed-loop window %d/%d started: %s',
            window_number,
            len(starts),
            timestamp,
        )
        disturbances = data.disturbances[start : start + duration_steps]
        rollout_timestamps = data.timestamps.iloc[
            start : start + duration_steps
        ].to_numpy()
        rollout_history = history_for_rollout(
            data,
            start,
            getattr(independent_evaluator, 'required_history_steps', 0),
        )
        recorded_controls = data.controls[start : start + duration_steps]
        hold_last_controls = np.repeat(
            data.controls[start][None, :],
            duration_steps,
            axis=0,
        )
        recorded_rollout = rollout_plant(
            independent_evaluator,
            data.states[start],
            recorded_controls,
            disturbances,
            history=rollout_history,
            timestamps=rollout_timestamps,
        )
        hold_last_rollout = rollout_plant(
            independent_evaluator,
            data.states[start],
            hold_last_controls,
            disturbances,
            history=rollout_history,
            timestamps=rollout_timestamps,
        )
        mpc_trajectory = run_receding_horizon_policy(
            controller,
            independent_evaluator,
            data,
            start,
            duration_steps,
            cfg.MPC.disturbance_forecast,
            observed_support,
            progress_prefix=f'window {window_number}/{len(starts)}',
            progress_interval=evaluation_cfg.progress_interval_steps,
        )
        mpc_rollout = PlantRollout(
            states=mpc_trajectory['states'],
            powers=mpc_trajectory['powers'],
        )
        actual_rollout = PlantRollout(
            states=data.states[start : start + duration_steps + 1],
            powers=data.powers[start : start + duration_steps],
        )

        recorded_metrics = policy_metrics(
            recorded_rollout,
            cfg.MPC.state_constraints,
        )
        hold_last_metrics = policy_metrics(
            hold_last_rollout,
            cfg.MPC.state_constraints,
        )
        mpc_metrics = policy_metrics(
            mpc_rollout,
            cfg.MPC.state_constraints,
        )
        actual_metrics = policy_metrics(
            actual_rollout,
            cfg.MPC.state_constraints,
        )
        actual_energy = energy_wh(actual_rollout.powers)
        evaluator_recorded_energy = energy_wh(recorded_rollout.powers)
        recorded_calibration = {
            'power': power_prediction_metrics(
                actual_rollout.powers,
                recorded_rollout.powers,
                data.power_names,
                mape_threshold,
            ),
            'state': state_prediction_metrics(
                actual_rollout.states[1:],
                recorded_rollout.states[1:],
                data.state_names,
            ),
            'constraint_prediction': constraint_prediction_metrics(
                actual_rollout.states[1:],
                recorded_rollout.states[1:],
                cfg.MPC.state_constraints,
            ),
            'actual_energy_wh': actual_energy,
            'evaluator_energy_wh': evaluator_recorded_energy,
            'energy_absolute_percentage_error_percent': (
                100.0
                * abs(evaluator_recorded_energy - actual_energy)
                / actual_energy
                if actual_energy > 1e-12
                else None
            ),
        }
        affine_evaluator_disagreement = power_prediction_metrics(
            mpc_trajectory['powers'],
            mpc_trajectory['affine_power_predictions'],
            data.power_names,
            mape_threshold,
        )
        solver = solver_metrics(
            mpc_trajectory['step_records'],
            deadline_seconds=60.0 * int(cfg.MPC.sample_interval_minutes),
        )
        policies = {
            'actual_recorded_reference': actual_metrics,
            'recorded': recorded_metrics,
            'hold_last': hold_last_metrics,
            'mpc': mpc_metrics,
        }
        comparisons = {
            'mpc_vs_recorded': energy_comparison(
                recorded_metrics['energy_wh'],
                mpc_metrics['energy_wh'],
            ),
            'mpc_vs_hold_last': energy_comparison(
                hold_last_metrics['energy_wh'],
                mpc_metrics['energy_wh'],
            ),
        }
        window_result = {
            'timestamp': timestamp,
            'duration_steps': duration_steps,
            'evaluation_plant': active_evaluator_label,
            'disturbances_seen_by_controller': str(cfg.MPC.disturbance_forecast),
            'disturbances_used_by_evaluator': 'actual future observations',
            'policies': policies,
            'comparisons': comparisons,
            'recorded_policy_evaluator_calibration': recorded_calibration,
            'mpc_affine_vs_independent_evaluator_power': (
                affine_evaluator_disagreement
            ),
            'solver': solver,
            'first_action': mpc_trajectory['first_action'],
        }
        window_results.append(window_result)
        mpc_evaluator_power_windows.append(mpc_trajectory['powers'])
        mpc_affine_power_windows.append(
            mpc_trajectory['affine_power_predictions']
        )
        mpc_control_windows.append(mpc_trajectory['controls'])
        mpc_state_context_windows.append(mpc_trajectory['states'][:-1])
        mpc_disturbance_context_windows.append(disturbances.copy())
        all_solver_records.extend(mpc_trajectory['step_records'])
        for record in mpc_trajectory['step_records']:
            closed_loop_step_rows.append(
                {'window_timestamp': timestamp, **record}
            )
        first_action_rows.append(
            {
                'timestamp': timestamp,
                **mpc_trajectory['first_action'],
                'recorded_energy_wh': recorded_metrics['energy_wh'],
                'hold_last_energy_wh': hold_last_metrics['energy_wh'],
                'mpc_energy_wh': mpc_metrics['energy_wh'],
                'solver_fallback_count': solver['fallback_count'],
            }
        )
        LOGGER.info(
            'MPC closed-loop window %d/%d completed: mpc_energy_wh=%.6f | '
            'recorded_energy_wh=%.6f | mean_solve_seconds=%.6f | fallbacks=%d',
            window_number,
            len(starts),
            mpc_metrics['energy_wh'],
            recorded_metrics['energy_wh'],
            solver['solve_seconds']['mean'],
            solver['fallback_count'],
        )

    selected_action_cfg = evaluation_cfg.selected_action_crossfit
    selected_action_crossfit_rows = []
    if bool(selected_action_cfg.enabled):
        nominal_crossfit_ranges = _expanding_crossfit_ranges(
            len(data.states),
            selected_action_cfg.folds,
            selected_action_cfg.initial_training_fraction,
        )
        crossfit_ranges = nominal_crossfit_ranges
        if bool(episode_group_cfg.enabled):
            try:
                aligned_crossfit_boundaries = (
                    align_boundaries_to_control_episodes(
                        data,
                        [start for start, _ in nominal_crossfit_ranges],
                    )
                )
                crossfit_ranges = [
                    (start, end)
                    for start, end in zip(
                        aligned_crossfit_boundaries,
                        [
                            *aligned_crossfit_boundaries[1:],
                            nominal_crossfit_ranges[-1][1],
                        ],
                    )
                ]
            except ValueError as error:
                LOGGER.warning(
                    'Could not align cross-fit folds to complete control '
                    'episodes; nominal folds retained: %s',
                    error,
                )
        crossfit_family = (
            selected_action_cfg.get('model_family')
            or evaluator_cfg.model_family
        )
        crossfit_folds = []
        LOGGER.info(
            'Running MPC-selected-action rolling cross-fit: family=%s | '
            'folds=%d | initial_training_end=%d',
            crossfit_family,
            len(crossfit_ranges),
            crossfit_ranges[0][0],
        )
        for fold_number, (fold_start, fold_end) in enumerate(
            crossfit_ranges,
            start=1,
        ):
            LOGGER.info(
                'MPC-selected-action cross-fit fold %d/%d: '
                'train=[0,%d) | evaluate=[%d,%d)',
                fold_number,
                len(crossfit_ranges),
                fold_start,
                fold_start,
                fold_end,
            )
            fold_evaluator = create_independent_evaluator(
                evaluator_cfg,
                data,
                fold_start,
                random_state=int(cfg.DEFAULT.seed) + fold_number,
                model_family=str(crossfit_family),
            )
            crossfit_folds.append(
                {
                    'fold_number': fold_number,
                    'training_end_exclusive': fold_start,
                    'evaluation_start': fold_start,
                    'evaluation_end_exclusive': fold_end,
                    'evaluator_name': evaluator_label(fold_evaluator),
                    'predictions': fold_evaluator.predict_transitions(
                        data,
                        fold_start,
                        fold_end,
                    ),
                }
            )
        (
            selected_action_crossfit_evaluation,
            selected_action_crossfit_rows,
        ) = evaluate_mpc_selected_action_crossfit(
            data,
            np.concatenate(mpc_control_windows),
            np.concatenate(mpc_state_context_windows),
            np.concatenate(mpc_disturbance_context_windows),
            crossfit_folds,
            initial_training_end_exclusive=crossfit_ranges[0][0],
            context_distance_threshold=(
                selected_action_cfg.context_standardized_rms_distance_threshold
            ),
            minimum_matched_samples_per_action=(
                selected_action_cfg.minimum_context_matched_samples_per_action
            ),
            minimum_independent_episodes_per_action=(
                selected_action_cfg.minimum_independent_episodes_per_action
            ),
            minimum_fold_training_samples_per_action=(
                selected_action_cfg.minimum_fold_training_samples_per_action
            ),
            minimum_fold_training_episodes_per_action=(
                selected_action_cfg.minimum_fold_training_episodes_per_action
            ),
            minimum_reportable_mpc_step_coverage_percent=(
                selected_action_cfg.minimum_reportable_mpc_step_coverage_percent
            ),
            mape_min_actual_power_w=mape_threshold,
            evaluator_family=str(crossfit_family),
            exclude_insufficient_episode_actions=bool(
                official_episode_cfg.enabled
            ),
        )
        LOGGER.info(
            'MPC-selected-action cross-fit complete: reportable=%s | '
            'weighted_total_power_mape_percent=%s | '
            'reportable_action_step_coverage_percent=%.3f | '
            'context_step_coverage_percent=%.3f',
            selected_action_crossfit_evaluation['reportable'],
            selected_action_crossfit_evaluation['primary_metric']['value'],
            selected_action_crossfit_evaluation['coverage'][
                'reportable_action_mpc_step_percent'
            ],
            selected_action_crossfit_evaluation['coverage'][
                'context_supported_mpc_step_percent'
            ],
        )
    else:
        selected_action_crossfit_evaluation = {
            'available': False,
            'reportable': False,
            'reason': (
                'Disabled by MPC.evaluation.selected_action_crossfit.enabled'
            ),
            'protocol': {
                'name': 'mpc_selected_action_rolling_crossfit_mape',
            },
            'primary_metric': {
                'name': (
                    'mpc_action_frequency_weighted_total_power_mape_percent'
                ),
                'value': None,
                'unit': 'percent',
                'reportable': False,
            },
            'combination_balanced_metric': {
                'name': 'mpc_selected_action_macro_total_power_mape_percent',
                'value': None,
                'unit': 'percent',
                'reportable': False,
            },
            'actions': [],
        }

    policy_aggregate = _aggregate_policy_results(
        window_results,
        str(evaluation_cfg.primary_baseline),
        int(evaluation_cfg.bootstrap_samples),
        float(evaluation_cfg.bootstrap_confidence),
        int(cfg.DEFAULT.seed),
    )
    aggregate_solver = solver_metrics(
        all_solver_records,
        deadline_seconds=60.0 * int(cfg.MPC.sample_interval_minutes),
    )
    aggregate_mpc_power_model_disagreement = power_prediction_metrics(
        np.concatenate(mpc_evaluator_power_windows),
        np.concatenate(mpc_affine_power_windows),
        data.power_names,
        mape_threshold,
    )
    final_quality_assessment, policy_warnings = _policy_acceptance(
        quality_assessment,
        policy_aggregate,
        aggregate_solver,
        len(window_results),
        int(evaluation_cfg.minimum_policy_windows),
        official_mape_evaluation,
    )
    quality_warnings.extend(policy_warnings)
    primary_comparison = policy_aggregate['mpc_vs_primary_baseline']
    official_metric = official_mape_evaluation['primary_metric']
    official_metric_source = (
        'fixed_action_candidate_pool_evaluation.primary_metric.value'
        if bool(candidate_pool_cfg.enabled)
        else 'fixed_optimal_action_holdout_evaluation.primary_metric.value'
    )
    kpi_assessment = {
        'name': 'official_total_hvac_power_mape_percent',
        'value': official_metric.get('value'),
        'target_maximum_percent': official_metric.get('target_percent'),
        'reportable': bool(official_metric.get('reportable', False)),
        'target_achieved': bool(official_metric.get('target_achieved', False)),
        'passed': bool(
            official_metric.get('reportable', False)
            and official_metric.get('target_achieved', False)
        ),
        'source': official_metric_source,
    }

    summary = {
        'framework': (
            'affine mixed-integer MPC + independent nonlinear receding-horizon '
            'offline evaluation'
        ),
        'evaluation_protocol': {
            'sample_interval_minutes': int(cfg.MPC.sample_interval_minutes),
            'prediction_horizon_steps': int(cfg.MPC.prediction_horizon),
            'control_horizon_steps': int(cfg.MPC.control_horizon),
            'binary_control_variables_per_solve': (
                int(cfg.MPC.control_horizon) * data.encoder.n_features
            ),
            'closed_loop_steps_per_window': duration_steps,
            'reoptimization_interval_steps': 1,
            'evaluated_windows': int(len(starts)),
            'window_stride_steps': int(evaluation_cfg.stride),
            'max_windows': int(evaluation_cfg.max_windows),
            'max_windows_zero_means_all': True,
            'controller_disturbance_forecast': str(
                cfg.MPC.disturbance_forecast
            ),
            'evaluator_disturbance_source': 'actual future observations',
            'optimization_model': 'affine state and power models',
            'evaluation_plant': active_evaluator_label,
            'action_options': OmegaConf.to_container(
                action_options_cfg,
                resolve=True,
            ),
            'control_preprocessing': data.control_preprocessing,
            'primary_offline_control_metric': (
                official_metric_source
            ),
            'fixed_action_candidate_pool_protocol': (
                candidate_pool_evaluation['protocol']
            ),
            'fixed_optimal_action_holdout_protocol': (
                fixed_action_evaluation['protocol']
            ),
            'mpc_selected_action_crossfit_protocol': (
                selected_action_crossfit_evaluation['protocol']
            ),
            'observed_combination_protocol': (
                observed_combination_evaluation['protocol']
            ),
            'counterfactual_limit': (
                'Results remain simulated and are not proof of real equipment savings'
            ),
        },
        'quality_assessment': final_quality_assessment,
        'quality_warnings': quality_warnings,
        'kpi_assessment': kpi_assessment,
        'control_support': support_metadata,
        'chronological_action_split_support': action_split_support,
        'primary_metric': official_metric,
        'primary_metric_source': official_metric_source,
        'fixed_action_candidate_pool_evaluation': (
            candidate_pool_evaluation
        ),
        'fixed_optimal_action_holdout_evaluation': fixed_action_evaluation,
        'observed_control_combination_evaluation': (
            observed_combination_evaluation
        ),
        'mpc_selected_action_crossfit_evaluation': (
            selected_action_crossfit_evaluation
        ),
        'affine_power_model_test_mape': {
            key: value
            for key, value in affine_metrics['test']['power'].items()
            if key.startswith('mape_')
            or key in ('minimum_actual_power_w', 'output_names')
        },
        'model_validation': model_validation,
        'policy_evaluation': {
            **policy_aggregate,
            'mpc_affine_vs_independent_evaluator_power': (
                aggregate_mpc_power_model_disagreement
            ),
            'mpc_affine_vs_independent_evaluator_interpretation': (
                'Diagnostic disagreement between two model predictions; '
                'not the official actual-Y MAPE'
            ),
            'solver': aggregate_solver,
        },
        'actual_closed_loop_mape': {
            'available': False,
            'value': None,
            'reason': (
                'Optimized MPC actions have not been applied to the real equipment; '
                'the reported closed-loop comparison uses an independent evaluator'
            ),
        },
        'evaluated_windows': int(len(window_results)),
        'baseline_energy_wh': primary_comparison['baseline_energy_wh'],
        'optimized_energy_wh': primary_comparison['candidate_energy_wh'],
        'energy_reduction_wh': primary_comparison['energy_reduction_wh'],
        'energy_reduction_percent': primary_comparison[
            'energy_reduction_percent'
        ],
        'windows': window_results,
    }

    _write_json(result_dir / 'affine_state_model.json', state_model.to_dict())
    _write_json(result_dir / 'affine_power_model.json', power_model.to_dict())
    _write_json(result_dir / 'affine_model_metrics.json', affine_metrics)
    _write_json(
        result_dir / 'independent_evaluator_metadata.json',
        independent_evaluator.to_metadata(),
    )
    _write_json(result_dir / 'model_validation.json', model_validation)
    _write_json(
        result_dir / 'chronological_action_split_support.json',
        action_split_support,
    )
    _write_json(
        result_dir / 'fixed_action_power_model_comparison.json',
        fixed_action_evaluation.get('total_power_model_selection') or {
            'enabled': False,
        },
    )
    _write_json(
        result_dir / 'fixed_action_candidate_pool_evaluation.json',
        candidate_pool_evaluation,
    )
    _write_json(
        result_dir / 'fixed_action_candidate_pool_model_comparison.json',
        candidate_pool_evaluation.get('total_power_model_selection') or {
            'enabled': False,
        },
    )
    _write_json(result_dir / 'mpc_evaluation.json', summary)
    pd.DataFrame(first_action_rows).to_csv(
        result_dir / 'mpc_first_actions.csv',
        index=False,
        encoding='utf-8-sig',
    )
    pd.DataFrame(closed_loop_step_rows).to_csv(
        result_dir / 'mpc_closed_loop_steps.csv',
        index=False,
        encoding='utf-8-sig',
    )
    pd.DataFrame(observed_combination_rows).to_csv(
        result_dir / 'observed_combination_test_predictions.csv',
        index=False,
        encoding='utf-8-sig',
    )
    fixed_action_candidate_dataset_frame = pd.DataFrame(
        fixed_action_candidate_dataset_rows
    )
    if fixed_action_candidate_dataset_frame.empty:
        fixed_action_candidate_dataset_frame = pd.DataFrame(columns=[
            'candidate_id',
            'rank_by_total_observed_frequency',
            'split',
            'episode_id',
            'source_row_index',
            'timestamp',
            'next_timestamp',
            *[
                f'control_{variable.name}'
                for variable in data.encoder.variables
            ],
            *[f'state_t_{name}' for name in data.state_names],
            *[
                f'disturbance_t_{name}'
                for name in data.disturbance_names
            ],
            *[f'target_power_{name}' for name in data.power_names],
            *[
                f'target_next_state_{name}'
                for name in data.state_names
            ],
        ])
    fixed_action_candidate_dataset_frame.to_csv(
        result_dir / 'fixed_action_candidate_dataset_all.csv',
        index=False,
        encoding='utf-8-sig',
    )
    for split_name in ('train', 'validation', 'test'):
        split_frame = fixed_action_candidate_dataset_frame.loc[
            fixed_action_candidate_dataset_frame['split'] == split_name
        ]
        split_frame.to_csv(
            result_dir / f'fixed_action_candidate_dataset_{split_name}.csv',
            index=False,
            encoding='utf-8-sig',
        )
    fixed_action_prediction_frame = pd.DataFrame(fixed_action_rows)
    if fixed_action_prediction_frame.empty:
        fixed_action_prediction_frame = pd.DataFrame(
            columns=[
                'source_row_index',
                'timestamp',
                *[
                    f'control_{variable.name}'
                    for variable in data.encoder.variables
                ],
                *[f'actual_{name}' for name in data.power_names],
                *[f'predicted_{name}' for name in data.power_names],
                'actual_total_hvac_power_w',
                'predicted_total_hvac_power_w',
                'legacy_component_sum_predicted_total_hvac_power_w',
                'official_total_power_model_family',
                'mape_denominator_valid',
                'absolute_percentage_error_percent',
                'actual_state_constraints_satisfied',
                'predicted_state_constraints_satisfied',
            ]
        )
    fixed_action_prediction_frame.to_csv(
        result_dir / 'fixed_optimal_action_test_predictions.csv',
        index=False,
        encoding='utf-8-sig',
    )
    pd.DataFrame(fixed_action_split_rows).to_csv(
        result_dir / 'fixed_optimal_action_split_support.csv',
        index=False,
        encoding='utf-8-sig',
    )
    pd.DataFrame(fixed_action_model_comparison_rows).to_csv(
        result_dir / 'fixed_action_power_model_comparison.csv',
        index=False,
        encoding='utf-8-sig',
    )
    candidate_pool_selection_rows = []
    for candidate in (
        (candidate_pool_evaluation.get('selection') or {}).get(
            'candidates',
            [],
        )
    ):
        support = candidate.get('support') or {}
        candidate_pool_selection_rows.append(
            {
                'candidate_id': candidate['candidate_id'],
                'rank_by_total_observed_frequency': candidate[
                    'rank_by_total_observed_frequency'
                ],
                **{
                    f'control_{name}': value
                    for name, value in candidate['action'].items()
                },
                'train_sample_count': (support.get('train') or {}).get(
                    'sample_count'
                ),
                'train_episode_count': (support.get('train') or {}).get(
                    'episode_count'
                ),
                'validation_sample_count': (
                    support.get('validation') or {}
                ).get('sample_count'),
                'validation_episode_count': (
                    support.get('validation') or {}
                ).get('episode_count'),
                'test_sample_count': (support.get('test') or {}).get(
                    'sample_count'
                ),
                'test_episode_count': (support.get('test') or {}).get(
                    'episode_count'
                ),
                'validation_valid_mape_sample_count': candidate[
                    'validation_valid_mape_sample_count'
                ],
                'validation_total_power_mape_percent': candidate[
                    'validation_total_power_mape_percent'
                ],
                'eligible_before_pool_limit': candidate[
                    'eligible_before_pool_limit'
                ],
                'selected_for_pool': candidate['selected_for_pool'],
                'rejection_reasons': ';'.join(candidate['rejection_reasons']),
            }
        )
    pd.DataFrame(candidate_pool_selection_rows).to_csv(
        result_dir / 'fixed_action_candidate_pool_selection.csv',
        index=False,
        encoding='utf-8-sig',
    )
    pd.DataFrame(candidate_pool_rows).to_csv(
        result_dir / 'fixed_action_candidate_pool_test_predictions.csv',
        index=False,
        encoding='utf-8-sig',
    )
    pd.DataFrame(candidate_pool_model_comparison_rows).to_csv(
        result_dir / 'fixed_action_candidate_pool_model_comparison.csv',
        index=False,
        encoding='utf-8-sig',
    )
    pd.DataFrame(action_split_support_rows).to_csv(
        result_dir / 'chronological_action_split_support.csv',
        index=False,
        encoding='utf-8-sig',
    )
    selected_action_crossfit_frame = pd.DataFrame(
        selected_action_crossfit_rows
    )
    if selected_action_crossfit_frame.empty:
        selected_action_crossfit_frame = pd.DataFrame(
            columns=[
                'fold_number',
                'fold_training_end_exclusive',
                'fold_training_exact_action_sample_count',
                'fold_training_exact_action_episode_count',
                'row_index',
                'timestamp',
                *[
                    f'control_{variable.name}'
                    for variable in data.encoder.variables
                ],
                'mpc_selected_step_count',
                'nearest_mpc_context_standardized_rms_distance',
                *[f'actual_{name}' for name in data.power_names],
                *[f'predicted_{name}' for name in data.power_names],
                'actual_total_hvac_power_w',
                'predicted_total_hvac_power_w',
                'mape_denominator_valid',
                'absolute_percentage_error_percent',
                'action_reportable',
            ]
        )
    selected_action_crossfit_frame.to_csv(
        result_dir / 'mpc_selected_action_crossfit_predictions.csv',
        index=False,
        encoding='utf-8-sig',
    )
    OmegaConf.save(cfg, result_dir / 'configs.yaml')

    console_summary = {
        'quality_assessment': final_quality_assessment,
        'official_kpi': kpi_assessment,
        'primary_candidate_pool_mape': {
            'reportable': candidate_pool_evaluation['reportable'],
            'candidate_count': (
                candidate_pool_evaluation.get('selection') or {}
            ).get('candidate_count'),
            'eligible_candidate_count': (
                candidate_pool_evaluation.get('selection') or {}
            ).get('eligible_candidate_count'),
            'selected_candidate_count': (
                candidate_pool_evaluation.get('selection') or {}
            ).get('selected_candidate_count'),
            'selected_candidate_ids': (
                candidate_pool_evaluation.get('selection') or {}
            ).get('selected_candidate_ids'),
            'pooled_total_power_mape_percent': (
                candidate_pool_evaluation.get('primary_metric') or {}
            ).get('value'),
            'valid_mape_sample_count': (
                candidate_pool_evaluation.get('test_evaluation') or {}
            ).get('valid_primary_mape_sample_count'),
            'exact_test_coverage_percent': (
                candidate_pool_evaluation.get('test_evaluation') or {}
            ).get('pool_exact_match_coverage_percent'),
            'equal_action_macro_mape_percent': (
                candidate_pool_evaluation.get('test_evaluation') or {}
            ).get('equal_action_macro_mape_percent'),
            'worst_action_mape_percent': (
                candidate_pool_evaluation.get('test_evaluation') or {}
            ).get('worst_action_mape_percent'),
            'action_target_pass_rate_percent': (
                candidate_pool_evaluation.get('test_evaluation') or {}
            ).get('action_target_pass_rate_percent'),
            'target_achieved': (
                candidate_pool_evaluation.get('primary_metric') or {}
            ).get('target_achieved'),
            'reason': candidate_pool_evaluation.get('reason'),
        },
        'primary_fixed_optimal_action_mape': {
            'reportable': fixed_action_evaluation['reportable'],
            'selected_action': (
                fixed_action_evaluation.get('selection') or {}
            ).get('selected_action'),
            'total_power_mape_percent': (
                fixed_action_evaluation.get('primary_metric') or {}
            ).get('value'),
            'exact_test_matches': (
                fixed_action_evaluation.get('test_evaluation') or {}
            ).get('exact_match_count'),
            'test_episode_count': (
                fixed_action_evaluation.get('test_evaluation') or {}
            ).get('exact_match_episode_count'),
            'valid_mape_sample_count': (
                fixed_action_evaluation.get('test_evaluation') or {}
            ).get('valid_primary_mape_sample_count'),
            'reason': fixed_action_evaluation.get('reason'),
            'selection_mode': (
                fixed_action_evaluation.get('selection') or {}
            ).get('selection_mode'),
            'strict_constraint_candidate_available': (
                fixed_action_evaluation.get('selection') or {}
            ).get('strict_constraint_candidate_available'),
        },
        'observed_combination_mape': {
            'reportable': observed_combination_evaluation['reportable'],
            'selected_action': (
                observed_combination_evaluation.get('selection') or {}
            ).get('selected_action'),
            'total_power_mape_percent': (
                observed_combination_evaluation.get('test_evaluation') or {}
            ).get('primary_metric', {}).get('value'),
            'exact_test_matches': (
                observed_combination_evaluation.get('test_evaluation') or {}
            ).get('exact_match_count'),
        },
        'mpc_selected_action_crossfit_mape': {
            'reportable': selected_action_crossfit_evaluation['reportable'],
            'total_power_mape_percent': (
                selected_action_crossfit_evaluation['primary_metric']['value']
            ),
            'partial_coverage_diagnostic_value': (
                selected_action_crossfit_evaluation['primary_metric'].get(
                    'partial_coverage_diagnostic_value'
                )
            ),
            'coverage': selected_action_crossfit_evaluation.get('coverage'),
            'reason': selected_action_crossfit_evaluation.get('reason'),
            'combination_balanced_total_power_mape_percent': (
                selected_action_crossfit_evaluation.get(
                    'combination_balanced_metric', {}
                ).get('value')
            ),
        },
        'chronological_action_split_support': {
            'observed_action_count': action_split_support[
                'observed_action_count'
            ],
            'present_in_all_splits_action_count': action_split_support[
                'present_in_all_splits_action_count'
            ],
            'eligible_in_all_splits_action_count': action_split_support[
                'eligible_in_all_splits_action_count'
            ],
            'cross_split_episode_count': action_split_support[
                'cross_split_episode_count'
            ],
        },
        'evaluated_windows': summary['evaluated_windows'],
        'primary_baseline': policy_aggregate['primary_baseline'],
        'baseline_energy_wh': summary['baseline_energy_wh'],
        'optimized_energy_wh': summary['optimized_energy_wh'],
        'energy_reduction_percent': summary['energy_reduction_percent'],
        'bootstrap_confidence_interval': primary_comparison[
            'bootstrap_confidence_interval'
        ],
        'solver': aggregate_solver,
    }
    LOGGER.info(
        'MPC summary:\n%s',
        json.dumps(console_summary, ensure_ascii=False, indent=2),
    )
    LOGGER.info('MPC artifacts: %s', result_dir)
    return summary
