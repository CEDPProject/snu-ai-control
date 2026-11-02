"""Component-aware total-power report models for the official fixed-action MAPE.

These models are deliberately evaluation-only.  They are never imported by the
mixed-integer controller and therefore cannot change the MPC objective,
constraints, or selected actions.  A family is selected on grouped validation
rows, refit on grouped train + validation rows, frozen, and then used once on
the grouped test rows.
"""

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
from omegaconf import OmegaConf
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from controllers.evaluation import (
    power_prediction_metrics,
    resolve_report_model_refit_rows,
)


SUPPORTED_TOTAL_POWER_FAMILIES = (
    'action_median',
    'action_median_huber_residual',
    'action_median_extra_trees_residual',
    'direct_total_extra_trees',
    'component_hvac_two_stage_extra_trees',
)


TWO_STAGE_HVAC_FAMILY = 'component_hvac_two_stage_extra_trees'
_HVAC_POWER_NAME_TOKENS = (
    'hvac',
    '냉난방',
    '냉방',
    '난방',
    'aircondition',
    'air_condition',
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


@dataclass(frozen=True)
class ReportModelInputSelection:
    """명시적으로 선택한 report model 입력 묶음.

    각 묶음은 현재 시점의 관측치만 사용한다. 미래 state, 미래 disturbance,
    test target power는 어떠한 경우에도 입력으로 포함하지 않는다.
    """

    current_state: bool = True
    action: bool = True
    current_disturbance: bool = True

    @classmethod
    def from_config(cls, raw: Mapping[str, object] | None):
        specification = _plain(raw)
        allowed = {
            'current_state',
            'action',
            'current_disturbance',
        }
        unknown = sorted(set(specification) - allowed)
        if unknown:
            raise ValueError(
                'Unknown total-power report input feature groups: '
                f'{unknown}; allowed={sorted(allowed)}'
            )

        values = {}
        for name, default in (
            ('current_state', True),
            ('action', True),
            ('current_disturbance', True),
        ):
            value = specification.get(name, default)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    'total_power_report_model.input_features.'
                    f'{name} must be a boolean, got {type(value).__name__}'
                )
            values[name] = bool(value)
        if not any(values.values()):
            raise ValueError(
                'At least one total-power report model input feature group '
                'must be enabled'
            )
        return cls(**values)

    def to_metadata(
        self,
        *,
        state_names=(),
        control_feature_names=(),
        disturbance_names=(),
    ) -> Dict[str, object]:
        groups = []
        feature_names = []
        if self.current_state:
            groups.append('current_state')
            feature_names.extend(str(name) for name in state_names)
        if self.action:
            groups.append('action')
            feature_names.extend(str(name) for name in control_feature_names)
        if self.current_disturbance:
            groups.append('current_disturbance')
            feature_names.extend(str(name) for name in disturbance_names)
        return {
            'current_state': self.current_state,
            'action': self.action,
            'current_disturbance': self.current_disturbance,
            'enabled_groups': groups,
            'feature_names': feature_names,
            'feature_dimension': int(len(feature_names)),
        }


def _design(
    states,
    controls,
    disturbances,
    input_selection: ReportModelInputSelection,
) -> np.ndarray:
    arrays = [
        np.atleast_2d(np.asarray(values, dtype=float))
        for values in (states, controls, disturbances)
    ]
    if len({len(values) for values in arrays}) != 1:
        raise ValueError('Total-power model inputs must have the same row count')
    selected = [
        values
        for enabled, values in zip(
            (
                input_selection.current_state,
                input_selection.action,
                input_selection.current_disturbance,
            ),
            arrays,
        )
        if enabled
    ]
    if not selected:
        raise ValueError('At least one report-model input feature group is required')
    return np.hstack(selected)


def _resolve_hvac_component_index(
    power_names,
    requested_name: str | None,
) -> int:
    """Resolve the component to receive the HVAC-specific two-stage model.

    The report model must never silently apply a HVAC gate to the wrong power
    output.  A caller may therefore pin the output name explicitly.  Without
    that option we only infer it from stable, domain-specific name tokens.
    """
    names = tuple(str(name) for name in power_names)
    if requested_name is not None:
        matches = [
            index
            for index, name in enumerate(names)
            if name == str(requested_name)
        ]
        if len(matches) != 1:
            raise ValueError(
                'hvac_component_name must exactly match one configured power '
                f'output; requested={requested_name!r}, outputs={list(names)}'
            )
        return int(matches[0])

    matches = [
        index
        for index, name in enumerate(names)
        if any(token in name.casefold() for token in _HVAC_POWER_NAME_TOKENS)
    ]
    if len(matches) != 1:
        raise ValueError(
            'Could not unambiguously infer the HVAC power output for '
            f'{TWO_STAGE_HVAC_FAMILY!r}; outputs={list(names)}, '
            'set hvac_component_name explicitly.'
        )
    return int(matches[0])


@dataclass
class HVACTwoStageComponentModel:
    """A non-negative HVAC component model with an OFF/low-power gate.

    ``predict`` returns the conditional expectation rather than a hard class
    decision: ``P(low) * median(low) + P(active) * E(power | active)``.  This
    avoids a discontinuous controller-report boundary while retaining the
    physically meaningful zero/low-power regime.  The report family is used
    only after an MPC action has been selected, never inside the MILP.
    """

    component_index: int
    component_name: str
    low_power_threshold_w: float
    low_power_median_w: float
    active_power_median_w: float
    classifier: object | None
    active_regressor: object | None
    low_power_training_count: int
    active_training_count: int
    conditional_regressor_minimum_samples: int
    fallback_mode: str

    @classmethod
    def fit(
        cls,
        design: np.ndarray,
        target: np.ndarray,
        *,
        component_index: int,
        component_name: str,
        low_power_threshold_w: float,
        n_estimators: int,
        min_samples_leaf: int,
        max_features: float,
        n_jobs: int,
        random_state: int,
        minimum_active_samples: int | None,
    ) -> 'HVACTwoStageComponentModel':
        target = np.asarray(target, dtype=float).reshape(-1)
        if len(design) != len(target):
            raise ValueError(
                'Two-stage HVAC design/target row count mismatch: '
                f'design={len(design)}, target={len(target)}'
            )
        if not len(target):
            raise ValueError('Two-stage HVAC model has no training rows')
        if not np.isfinite(target).all():
            raise ValueError('Two-stage HVAC targets must be finite')
        if not np.isfinite(float(low_power_threshold_w)):
            raise ValueError('hvac_low_power_threshold_w must be finite')
        if float(low_power_threshold_w) < 0.0:
            raise ValueError('hvac_low_power_threshold_w must be non-negative')

        low_power = target <= float(low_power_threshold_w)
        low_count = int(low_power.sum())
        active_count = int((~low_power).sum())
        low_median = (
            float(np.median(target[low_power])) if low_count else 0.0
        )
        active_median = (
            float(np.median(target[~low_power]))
            if active_count
            else float(np.median(target))
        )
        requested_minimum = 0
        if minimum_active_samples is not None:
            if isinstance(minimum_active_samples, (bool, np.bool_)):
                raise TypeError(
                    'hvac_minimum_active_samples must be a positive integer, '
                    'not a boolean'
                )
            if int(minimum_active_samples) != minimum_active_samples:
                raise TypeError(
                    'hvac_minimum_active_samples must be a positive integer'
                )
            requested_minimum = int(minimum_active_samples)
            if requested_minimum <= 0:
                raise ValueError('hvac_minimum_active_samples must be positive')
        conditional_minimum = int(
            max(2, int(min_samples_leaf), requested_minimum)
        )

        classifier = None
        if low_count and active_count:
            classifier = ExtraTreesClassifier(
                n_estimators=int(n_estimators),
                min_samples_leaf=int(min_samples_leaf),
                max_features=float(max_features),
                n_jobs=int(n_jobs),
                random_state=int(random_state),
            )
            classifier.fit(design, low_power.astype(int))

        active_regressor = None
        if active_count >= conditional_minimum:
            active_regressor = ExtraTreesRegressor(
                n_estimators=int(n_estimators),
                min_samples_leaf=int(min_samples_leaf),
                max_features=float(max_features),
                n_jobs=int(n_jobs),
                random_state=int(random_state) + 1,
            )
            active_regressor.fit(design[~low_power], target[~low_power])

        if low_count == len(target):
            fallback_mode = 'one_class_low_power_median'
        elif active_count == len(target):
            fallback_mode = (
                'one_class_active_conditional_regressor'
                if active_regressor is not None
                else 'one_class_active_median'
            )
        elif active_regressor is None:
            fallback_mode = 'mixed_class_active_median'
        else:
            fallback_mode = 'two_stage_classifier_and_conditional_regressor'

        return cls(
            component_index=int(component_index),
            component_name=str(component_name),
            low_power_threshold_w=float(low_power_threshold_w),
            low_power_median_w=low_median,
            active_power_median_w=active_median,
            classifier=classifier,
            active_regressor=active_regressor,
            low_power_training_count=low_count,
            active_training_count=active_count,
            conditional_regressor_minimum_samples=conditional_minimum,
            fallback_mode=fallback_mode,
        )

    def predict(self, design: np.ndarray) -> np.ndarray:
        """Predict non-negative HVAC power on arbitrary current-time rows."""
        design = np.asarray(design, dtype=float)
        if design.ndim != 2:
            raise ValueError('Two-stage HVAC design must be a 2D matrix')
        sample_count = int(len(design))
        if self.classifier is None:
            low_probability = np.full(
                sample_count,
                1.0 if self.low_power_training_count else 0.0,
                dtype=float,
            )
        else:
            probabilities = np.asarray(
                self.classifier.predict_proba(design), dtype=float
            )
            classes = np.asarray(self.classifier.classes_)
            low_index = np.flatnonzero(classes == 1)
            if len(low_index) != 1:
                raise ValueError(
                    'Two-stage HVAC classifier is missing its low-power class'
                )
            low_probability = probabilities[:, int(low_index[0])]

        if self.active_regressor is None:
            active_prediction = np.full(
                sample_count,
                self.active_power_median_w,
                dtype=float,
            )
        else:
            active_prediction = np.asarray(
                self.active_regressor.predict(design), dtype=float
            ).reshape(-1)
        prediction = (
            low_probability * self.low_power_median_w
            + (1.0 - low_probability) * active_prediction
        )
        return np.maximum(prediction, 0.0)

    def to_metadata(self) -> Dict[str, object]:
        return {
            'type': 'off_low_power_classifier_plus_conditional_regressor',
            'component_index': int(self.component_index),
            'component_name': self.component_name,
            'low_power_threshold_w': float(self.low_power_threshold_w),
            'low_power_training_count': int(self.low_power_training_count),
            'active_training_count': int(self.active_training_count),
            'low_power_median_w': float(self.low_power_median_w),
            'active_power_median_w': float(self.active_power_median_w),
            'classifier_fitted': self.classifier is not None,
            'conditional_regressor_fitted': self.active_regressor is not None,
            'conditional_regressor_minimum_samples': int(
                self.conditional_regressor_minimum_samples
            ),
            'fallback_mode': self.fallback_mode,
        }


@dataclass
class FixedActionTotalPowerModel:
    """하나의 동결된 component/total HVAC power report model.

    내부 예측은 ``data.power_names`` 순서의 component vector다. ``predict``는
    하위 호환성을 위해 그 합인 총전력을 반환하고, component별 값은
    ``predict_components``로 명시적으로 꺼낸다.
    """

    family: str
    action_medians: Dict[tuple[int, ...], np.ndarray]
    action_counts: Dict[tuple[int, ...], int]
    global_median: np.ndarray
    minimum_action_samples: int
    estimator: object | None
    residual_target: bool
    training_transition_count: int
    parameters: Dict[str, object]
    power_names: tuple[str, ...]
    input_selection: ReportModelInputSelection
    input_feature_metadata: Dict[str, object]
    hvac_two_stage_model: HVACTwoStageComponentModel | None = None

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
        input_features: Mapping[str, object] | None = None,
        hvac_low_power_threshold_w: float = 0.1,
        hvac_component_name: str | None = None,
        hvac_minimum_active_samples: int | None = None,
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
        actual_components = np.asarray(powers, dtype=float)
        if actual_components.ndim != 2 or not actual_components.shape[1]:
            raise ValueError('Total-power report targets must be a nonempty matrix')
        if not np.isfinite(actual_components).all():
            raise ValueError('Total-power report targets must be finite')
        input_selection = ReportModelInputSelection.from_config(input_features)
        input_feature_metadata = input_selection.to_metadata(
            state_names=data.state_names,
            control_feature_names=data.encoder.feature_names,
            disturbance_names=data.disturbance_names,
        )

        keys = _action_keys(controls)
        action_counts: Dict[tuple[int, ...], int] = {}
        grouped_values: Dict[tuple[int, ...], list[np.ndarray]] = {}
        for key, value in zip(keys, actual_components):
            grouped_values.setdefault(key, []).append(np.asarray(value, dtype=float))
        action_medians = {}
        for key, values in grouped_values.items():
            action_counts[key] = int(len(values))
            if len(values) >= int(minimum_action_samples):
                action_medians[key] = np.median(
                    np.vstack(values), axis=0
                ).astype(float)
        global_median = np.median(actual_components, axis=0).astype(float)
        baseline = (
            np.vstack([
                action_medians.get(key, global_median) for key in keys
            ]).astype(float)
            if input_selection.action
            else np.tile(global_median[None, :], (len(keys), 1))
        )

        estimator = None
        residual_target = family.startswith('action_median_')
        target = actual_components - baseline if residual_target else actual_components
        design = _design(states, controls, disturbances, input_selection)
        parameters = {
            'minimum_action_samples': int(minimum_action_samples),
            'random_state': int(random_state),
        }
        hvac_two_stage_model = None
        if family == 'action_median_huber_residual':
            if float(huber_epsilon) <= 1.0:
                raise ValueError('huber_epsilon must be greater than 1.0')
            huber = HuberRegressor(
                epsilon=float(huber_epsilon),
                alpha=float(huber_alpha),
                max_iter=int(huber_max_iter),
            )
            regressor = (
                MultiOutputRegressor(huber)
                if target.shape[1] > 1
                else huber
            )
            estimator = make_pipeline(StandardScaler(), regressor)
            estimator.fit(
                design,
                target if target.shape[1] > 1 else target.ravel(),
            )
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
            TWO_STAGE_HVAC_FAMILY,
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
            if family == TWO_STAGE_HVAC_FAMILY:
                hvac_component_index = _resolve_hvac_component_index(
                    data.power_names,
                    hvac_component_name,
                )
                hvac_two_stage_model = HVACTwoStageComponentModel.fit(
                    design,
                    actual_components[:, hvac_component_index],
                    component_index=hvac_component_index,
                    component_name=str(data.power_names[hvac_component_index]),
                    low_power_threshold_w=float(hvac_low_power_threshold_w),
                    n_estimators=int(n_estimators),
                    min_samples_leaf=int(min_samples_leaf),
                    max_features=float(max_features),
                    n_jobs=int(n_jobs),
                    random_state=int(random_state),
                    minimum_active_samples=(
                        int(minimum_action_samples)
                        if hvac_minimum_active_samples is None
                        else hvac_minimum_active_samples
                    ),
                )
                parameters['hvac_two_stage'] = hvac_two_stage_model.to_metadata()

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
            power_names=tuple(str(name) for name in data.power_names),
            input_selection=input_selection,
            input_feature_metadata=input_feature_metadata,
            hvac_two_stage_model=hvac_two_stage_model,
        )

    def predict_components(self, states, controls, disturbances) -> np.ndarray:
        """각 전력 성분을 ``power_names`` 순서로 예측한다."""
        controls_array = np.atleast_2d(np.asarray(controls, dtype=float))
        keys = _action_keys(controls_array)
        baseline = (
            np.vstack([
                self.action_medians.get(key, self.global_median)
                for key in keys
            ]).astype(float)
            if self.input_selection.action
            else np.tile(self.global_median[None, :], (len(keys), 1))
        )
        if self.family == 'action_median':
            prediction = baseline
        else:
            design = _design(
                states,
                controls,
                disturbances,
                self.input_selection,
            )
            model_prediction = np.asarray(
                self.estimator.predict(design),
                dtype=float,
            )
            if model_prediction.ndim == 1:
                model_prediction = model_prediction[:, None]
            prediction = (
                baseline + model_prediction
                if self.residual_target
                else model_prediction
            )
            if self.hvac_two_stage_model is not None:
                prediction[:, self.hvac_two_stage_model.component_index] = (
                    self.hvac_two_stage_model.predict(design)
                )
        if prediction.shape != baseline.shape:
            raise ValueError(
                'Total-power report component prediction shape mismatch: '
                f'predicted={prediction.shape}, expected={baseline.shape}'
            )
        return np.maximum(prediction, 0.0)

    def predict(self, states, controls, disturbances) -> np.ndarray:
        """총 HVAC 전력을 예측한다(기존 호출부 호환용)."""
        return self.predict_components(
            states,
            controls,
            disturbances,
        ).sum(axis=1, keepdims=True)

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
            'role': (
                'evaluation-only frozen component-aware total-HVAC-power '
                'report model'
            ),
            'model_family': self.family,
            'target': (
                'configured HVAC power outputs in watts; total HVAC power is '
                'their per-row sum'
            ),
            'used_by_mpc_or_milp': False,
            'training_transition_count': self.training_transition_count,
            'supported_exact_action_count': int(len(self.action_medians)),
            'minimum_action_samples': self.minimum_action_samples,
            'power_output_names': list(self.power_names),
            'input_features': self.input_feature_metadata,
            'action_median_baseline_uses_action': self.input_selection.action,
            'global_training_component_median_w': dict(
                zip(self.power_names, self.global_median.astype(float).tolist())
            ),
            'global_training_median_w': float(self.global_median.sum()),
            'selected_action_training_sample_count': (
                self.action_counts.get(selected_key) if selected_key is not None else None
            ),
            'selected_action_training_median_w': (
                (
                    float(self.action_medians[selected_key].sum())
                    if selected_key in self.action_medians
                    else None
                )
                if selected_key is not None
                else None
            ),
            'selected_action_training_component_median_w': (
                dict(
                    zip(
                        self.power_names,
                        self.action_medians[selected_key].astype(float).tolist(),
                    )
                )
                if selected_key in self.action_medians
                else None
            ) if selected_key is not None else None,
            'unseen_or_sparse_action_fallback': 'global training median',
            'component_models': (
                {
                    self.hvac_two_stage_model.component_name:
                    self.hvac_two_stage_model.to_metadata(),
                }
                if self.hvac_two_stage_model is not None
                else {}
            ),
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
    test_rows=None,
    refit_scope: str = 'train_validation',
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
        'input_features': specification.get('input_features'),
        'hvac_low_power_threshold_w': float(
            specification.get('hvac_low_power_threshold_w', 0.1)
        ),
        'hvac_component_name': specification.get('hvac_component_name'),
        'hvac_minimum_active_samples': specification.get(
            'hvac_minimum_active_samples'
        ),
    }
    states, controls, disturbances, _, powers = data.model_arrays_for_rows(
        exact_validation_rows
    )
    actual_components = np.asarray(powers, dtype=float)
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
            predicted_components = candidate.predict_components(
                states,
                controls,
                disturbances,
            )
            metrics = power_prediction_metrics(
                actual_total,
                predicted_total,
                ('total_hvac_power_w',),
                float(mape_min_actual_power_w),
            )
            component_metrics = power_prediction_metrics(
                actual_components,
                predicted_components,
                data.power_names,
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
                    'validation_component_power_prediction': component_metrics,
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
                    'validation_component_power_prediction': None,
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

    combined_training_rows = resolve_report_model_refit_rows(
        data,
        training_rows,
        validation_rows,
        test_rows=test_rows,
        scope=refit_scope,
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
