"""Direct total-power report models for the official fixed-action MAPE.

These models are deliberately evaluation-only.  They are never imported by the
mixed-integer controller and therefore cannot change the MPC objective,
constraints, or selected actions.  A family is selected on grouped validation
rows, refit on grouped train + validation rows, frozen, and then used once on
the grouped test rows.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from omegaconf import OmegaConf
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from controllers.evaluation import power_prediction_metrics


SUPPORTED_TOTAL_POWER_FAMILIES = (
    'action_median',
    'action_median_huber_residual',
    'action_median_extra_trees_residual',
    'direct_total_extra_trees',
)


def _plain(config) -> Dict[str, object]:
    if config is None:
        return {}
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


def _action_keys(controls: np.ndarray) -> list[tuple[int, ...]]:
    rounded = np.rint(np.asarray(controls, dtype=float)).astype(int)
    return [tuple(row.tolist()) for row in rounded]


def _design(states, controls, disturbances) -> np.ndarray:
    arrays = [
        np.atleast_2d(np.asarray(values, dtype=float))
        for values in (states, controls, disturbances)
    ]
    if len({len(values) for values in arrays}) != 1:
        raise ValueError('Total-power model inputs must have the same row count')
    return np.hstack(arrays)


@dataclass
class FixedActionTotalPowerModel:
    """One frozen direct-total-power report model."""

    family: str
    action_medians: Dict[tuple[int, ...], float]
    action_counts: Dict[tuple[int, ...], int]
    global_median: float
    minimum_action_samples: int
    estimator: object | None
    residual_target: bool
    training_transition_count: int
    parameters: Dict[str, object]

    @classmethod
    def fit(
        cls,
        data,
        training_rows,
        *,
        family: str,
        minimum_action_samples: int,
        n_estimators: int,
        min_samples_leaf: int,
        max_features: float,
        huber_epsilon: float,
        huber_alpha: float,
        huber_max_iter: int,
        n_jobs: int,
        random_state: int,
    ) -> 'FixedActionTotalPowerModel':
        family = str(family)
        if family not in SUPPORTED_TOTAL_POWER_FAMILIES:
            raise ValueError(
                f'Unknown fixed-action total-power family {family!r}; '
                f'available={list(SUPPORTED_TOTAL_POWER_FAMILIES)}'
            )
        rows = data.valid_transition_rows(training_rows)
        if not len(rows):
            raise ValueError('Total-power report model has no training transitions')
        if int(minimum_action_samples) <= 0:
            raise ValueError('minimum_action_samples must be positive')
        states, controls, disturbances, _, powers = data.model_arrays_for_rows(rows)
        actual_total = np.asarray(powers, dtype=float).sum(axis=1)
        if not np.isfinite(actual_total).all():
            raise ValueError('Total-power report targets must be finite')

        keys = _action_keys(controls)
        action_counts: Dict[tuple[int, ...], int] = {}
        grouped_values: Dict[tuple[int, ...], list[float]] = {}
        for key, value in zip(keys, actual_total):
            grouped_values.setdefault(key, []).append(float(value))
        action_medians = {}
        for key, values in grouped_values.items():
            action_counts[key] = int(len(values))
            if len(values) >= int(minimum_action_samples):
                action_medians[key] = float(np.median(values))
        global_median = float(np.median(actual_total))
        baseline = np.asarray(
            [action_medians.get(key, global_median) for key in keys],
            dtype=float,
        )

        estimator = None
        residual_target = family.startswith('action_median_')
        target = actual_total - baseline if residual_target else actual_total
        design = _design(states, controls, disturbances)
        parameters = {
            'minimum_action_samples': int(minimum_action_samples),
            'random_state': int(random_state),
        }
        if family == 'action_median_huber_residual':
            if float(huber_epsilon) <= 1.0:
                raise ValueError('huber_epsilon must be greater than 1.0')
            estimator = make_pipeline(
                StandardScaler(),
                HuberRegressor(
                    epsilon=float(huber_epsilon),
                    alpha=float(huber_alpha),
                    max_iter=int(huber_max_iter),
                ),
            )
            estimator.fit(design, target)
            parameters.update(
                {
                    'huber_epsilon': float(huber_epsilon),
                    'huber_alpha': float(huber_alpha),
                    'huber_max_iter': int(huber_max_iter),
                }
            )
        elif family in (
            'action_median_extra_trees_residual',
            'direct_total_extra_trees',
        ):
            estimator = ExtraTreesRegressor(
                n_estimators=int(n_estimators),
                min_samples_leaf=int(min_samples_leaf),
                max_features=float(max_features),
                n_jobs=int(n_jobs),
                random_state=int(random_state),
            )
            estimator.fit(design, target)
            parameters.update(
                {
                    'n_estimators': int(n_estimators),
                    'min_samples_leaf': int(min_samples_leaf),
                    'max_features': float(max_features),
                    'n_jobs': int(n_jobs),
                }
            )

        return cls(
            family=family,
            action_medians=action_medians,
            action_counts=action_counts,
            global_median=global_median,
            minimum_action_samples=int(minimum_action_samples),
            estimator=estimator,
            residual_target=residual_target,
            training_transition_count=int(len(rows)),
            parameters=parameters,
        )

    def predict(self, states, controls, disturbances) -> np.ndarray:
        controls_array = np.atleast_2d(np.asarray(controls, dtype=float))
        baseline = np.asarray(
            [
                self.action_medians.get(key, self.global_median)
                for key in _action_keys(controls_array)
            ],
            dtype=float,
        )
        if self.family == 'action_median':
            prediction = baseline
        else:
            model_prediction = np.asarray(
                self.estimator.predict(_design(states, controls, disturbances)),
                dtype=float,
            ).reshape(-1)
            prediction = (
                baseline + model_prediction
                if self.residual_target
                else model_prediction
            )
        return np.maximum(prediction, 0.0).reshape(-1, 1)

    def to_metadata(self, selected_control=None) -> Dict[str, object]:
        selected_key = (
            None
            if selected_control is None
            else tuple(
                np.rint(np.asarray(selected_control, dtype=float))
                .astype(int)
                .tolist()
            )
        )
        return {
            'role': 'evaluation-only frozen direct total-HVAC-power report model',
            'model_family': self.family,
            'target': 'sum of configured HVAC power outputs in watts',
            'used_by_mpc_or_milp': False,
            'training_transition_count': self.training_transition_count,
            'supported_exact_action_count': int(len(self.action_medians)),
            'minimum_action_samples': self.minimum_action_samples,
            'global_training_median_w': self.global_median,
            'selected_action_training_sample_count': (
                self.action_counts.get(selected_key) if selected_key is not None else None
            ),
            'selected_action_training_median_w': (
                self.action_medians.get(selected_key) if selected_key is not None else None
            ),
            'unseen_or_sparse_action_fallback': 'global training median',
            'parameters': self.parameters,
        }


@dataclass(frozen=True)
class FixedActionPowerModelSelection:
    """Validation selection output plus the final train+validation model."""

    model: FixedActionTotalPowerModel
    selected_family: str
    validation_results: list[Dict[str, object]]
    validation_rows: np.ndarray
    metadata: Dict[str, object]


def select_and_fit_fixed_action_power_model(
    data,
    *,
    training_rows,
    validation_rows,
    selected_control,
    config,
    mape_min_actual_power_w: float,
    random_state: int,
) -> FixedActionPowerModelSelection:
    """Choose a family without test Y, then fit one frozen report model."""
    specification = _plain(config)
    families = [str(value) for value in specification['candidate_families']]
    if not families:
        raise ValueError('candidate_families must contain at least one family')
    if len(families) != len(set(families)):
        raise ValueError('candidate_families must not contain duplicates')
    unknown = sorted(set(families) - set(SUPPORTED_TOTAL_POWER_FAMILIES))
    if unknown:
        raise ValueError(
            f'Unknown fixed-action total-power families: {unknown}; '
            f'available={list(SUPPORTED_TOTAL_POWER_FAMILIES)}'
        )
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
    target_mape_percent = float(
        specification.get('target_mape_percent', np.inf)
    )

    selected_control = np.rint(np.asarray(selected_control, dtype=float)).astype(int)
    validation_rows = data.valid_transition_rows(validation_rows)
    exact_validation_rows = validation_rows[
        np.all(
            np.rint(data.controls[validation_rows]).astype(int)
            == selected_control,
            axis=1,
        )
    ]
    minimum_validation_samples = int(
        specification['minimum_validation_mape_samples']
    )
    if len(exact_validation_rows) < minimum_validation_samples:
        raise ValueError(
            'Selected action has too few exact validation rows for total-power '
            f'model selection: observed={len(exact_validation_rows)}, '
            f'required={minimum_validation_samples}'
        )

    fit_parameters = {
        'minimum_action_samples': int(specification['minimum_action_samples']),
        'n_estimators': int(specification['n_estimators']),
        'min_samples_leaf': int(specification['min_samples_leaf']),
        'max_features': float(specification['max_features']),
        'huber_epsilon': float(specification['huber_epsilon']),
        'huber_alpha': float(specification['huber_alpha']),
        'huber_max_iter': int(specification['huber_max_iter']),
        'n_jobs': int(specification['n_jobs']),
    }
    states, controls, disturbances, _, powers = data.model_arrays_for_rows(
        exact_validation_rows
    )
    actual_total = powers.sum(axis=1, keepdims=True)
    validation_results = []
    for priority, family in enumerate(families):
        try:
            candidate = FixedActionTotalPowerModel.fit(
                data,
                training_rows,
                family=family,
                random_state=int(random_state) + priority,
                **fit_parameters,
            )
            predicted_total = candidate.predict(states, controls, disturbances)
            metrics = power_prediction_metrics(
                actual_total,
                predicted_total,
                ('total_hvac_power_w',),
                float(mape_min_actual_power_w),
            )
            valid_samples = int(metrics['mape_valid_samples_per_output'][0])
            eligible = bool(
                metrics['mape_percent'] is not None
                and valid_samples >= minimum_validation_samples
            )
            validation_results.append(
                {
                    'priority': int(priority),
                    'model_family': family,
                    'fit_succeeded': True,
                    'eligible': eligible,
                    'validation_exact_action_sample_count': int(
                        len(exact_validation_rows)
                    ),
                    'validation_valid_mape_sample_count': valid_samples,
                    'validation_total_power_mape_percent': metrics['mape_percent'],
                    'validation_total_power_wape_percent': metrics['wape_percent'],
                    'validation_total_power_mae_w': metrics['mae_w_per_output'][0],
                    'meets_target': bool(
                        eligible
                        and float(metrics['mape_percent'])
                        <= target_mape_percent + 1e-12
                    ),
                    'failure': None,
                }
            )
        except Exception as error:  # keep other configured families testable
            validation_results.append(
                {
                    'priority': int(priority),
                    'model_family': family,
                    'fit_succeeded': False,
                    'eligible': False,
                    'validation_exact_action_sample_count': int(
                        len(exact_validation_rows)
                    ),
                    'validation_valid_mape_sample_count': 0,
                    'validation_total_power_mape_percent': None,
                    'validation_total_power_wape_percent': None,
                    'validation_total_power_mae_w': None,
                    'meets_target': False,
                    'failure': f'{type(error).__name__}: {error}',
                }
            )

    eligible = [item for item in validation_results if item['eligible']]
    if not eligible:
        failures = [
            item['failure']
            for item in validation_results
            if item['failure'] is not None
        ]
        raise ValueError(
            'No total-power report model passed the validation MAPE sample gate; '
            f'failures={failures}'
        )
    passing = [item for item in eligible if item['meets_target']]
    if selection_rule == 'simplest_within_target' and passing:
        selected_result = min(passing, key=lambda item: int(item['priority']))
    else:
        selected_result = min(
            eligible,
            key=lambda item: (
                float(item['validation_total_power_mape_percent']),
                float(item['validation_total_power_wape_percent']),
                float(item['validation_total_power_mae_w']),
                int(item['priority']),
            ),
        )
    for item in validation_results:
        item['selected'] = bool(item is selected_result)

    combined_training_rows = np.unique(
        np.concatenate(
            [
                data.valid_transition_rows(training_rows),
                data.valid_transition_rows(validation_rows),
            ]
        )
    )
    selected_family = str(selected_result['model_family'])
    frozen_model = FixedActionTotalPowerModel.fit(
        data,
        combined_training_rows,
        family=selected_family,
        random_state=int(random_state) + 1000,
        **fit_parameters,
    )
    metadata = {
        'selection_rule': selection_rule,
        'selection_uses_test_targets': False,
        'selection_training_scope': 'episode-grouped train rows only',
        'selection_scoring_scope': (
            'selected exact action on episode-grouped validation rows only'
        ),
        'selection_metric': (
            'configured family order among models meeting the validation MAPE '
            'target; otherwise minimum validation total-power MAPE percent'
            if selection_rule == 'simplest_within_target'
            else 'minimum validation total-power MAPE percent'
        ),
        'target_mape_percent': (
            target_mape_percent if np.isfinite(target_mape_percent) else None
        ),
        'tie_breakers': 'WAPE, MAE, then configured family order',
        'validation_exact_action_sample_count': int(len(exact_validation_rows)),
        'minimum_validation_mape_samples': minimum_validation_samples,
        'selected_family': selected_family,
        'refit_scope': 'episode-grouped train + validation rows',
        'frozen_before_test': True,
        'candidate_families': families,
        'model': frozen_model.to_metadata(selected_control),
    }
    return FixedActionPowerModelSelection(
        model=frozen_model,
        selected_family=selected_family,
        validation_results=validation_results,
        validation_rows=exact_validation_rows,
        metadata=metadata,
    )
