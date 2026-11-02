"""Independent-model action screen used before a receding-horizon MILP solve.

The affine controller is intentionally simple enough for an exact MILP.  This
module never replaces it inside the optimization.  Instead it calibrates two
train/validation-only guards and applies them to each *candidate first action*:

* component-power disagreement between the affine controller and an
  independent evaluator; and
* one-step evaluator state safety with validation-calibrated conservative
  margins.

It is deliberately a filter, not a learned policy.  That keeps the MILP's
objective auditable and makes every rejected action explainable in an action
diagnostics artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf


def _plain(value) -> dict:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError('Action model screen configuration must be a mapping')


def _positive_quantile(values: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values):
        return 0.0
    return float(np.quantile(np.maximum(values, 0.0), float(quantile)))


@dataclass(frozen=True)
class ActionModelScreenConfig:
    """Configurable conservative pre-MILP checks.

    Thresholds are either explicitly configured or calibrated on chronological
    validation rows using a train-only evaluator.  Test state/power targets are
    never used to tune these values.
    """

    enabled: bool = True
    unavailable_model_policy: str = 'monitor'
    power_disagreement_enabled: bool = True
    power_floor_w: float = 1.0
    power_disagreement_quantile: float = 0.95
    maximum_power_disagreement: float | None = None
    state_safety_enabled: bool = True
    state_margin_quantile: float = 0.95
    # 'reject': every candidate must keep the margin-adjusted next state in band.
    # 'non_worsening': only when *no* candidate passes that rule (state already
    # out of band), accept candidates whose per-state predicted violation is no
    # larger than that of holding the reference (previously applied) action.
    state_out_of_band_policy: str = 'reject'

    @classmethod
    def from_mapping(cls, config=None) -> 'ActionModelScreenConfig':
        spec = _plain(config)
        disagreement = _plain(spec.get('power_disagreement', {}))
        safety = _plain(spec.get('state_safety', {}))
        result = cls(
            enabled=bool(spec.get('enabled', True)),
            unavailable_model_policy=str(
                spec.get('unavailable_model_policy', 'monitor')
            ),
            power_disagreement_enabled=bool(
                disagreement.get('enabled', spec.get('power_disagreement_enabled', True))
            ),
            power_floor_w=float(
                disagreement.get('power_floor_w', spec.get('power_floor_w', 1.0))
            ),
            power_disagreement_quantile=float(
                disagreement.get(
                    'calibration_quantile',
                    spec.get('power_disagreement_quantile', 0.95),
                )
            ),
            maximum_power_disagreement=(
                None
                if disagreement.get(
                    'maximum_normalized_component_disagreement',
                    spec.get('maximum_power_disagreement'),
                )
                is None
                else float(
                    disagreement.get(
                        'maximum_normalized_component_disagreement',
                        spec.get('maximum_power_disagreement'),
                    )
                )
            ),
            state_safety_enabled=bool(
                safety.get('enabled', spec.get('state_safety_enabled', True))
            ),
            state_margin_quantile=float(
                safety.get(
                    'calibration_quantile',
                    spec.get('state_margin_quantile', 0.95),
                )
            ),
            state_out_of_band_policy=str(
                safety.get('out_of_band_policy', 'reject')
            ),
        )
        result._validate()
        return result

    def _validate(self) -> None:
        if self.unavailable_model_policy not in ('monitor', 'reject'):
            raise ValueError(
                "unavailable_model_policy must be 'monitor' or 'reject'"
            )
        if self.state_out_of_band_policy not in ('reject', 'non_worsening'):
            raise ValueError(
                "state_safety.out_of_band_policy must be 'reject' or 'non_worsening'"
            )
        if not np.isfinite(self.power_floor_w) or self.power_floor_w <= 0.0:
            raise ValueError('power_floor_w must be finite and positive')
        for name, value in (
            ('power_disagreement_quantile', self.power_disagreement_quantile),
            ('state_margin_quantile', self.state_margin_quantile),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f'{name} must be in (0, 1]')
        if (
            self.maximum_power_disagreement is not None
            and (
                not np.isfinite(self.maximum_power_disagreement)
                or self.maximum_power_disagreement < 0.0
            )
        ):
            raise ValueError(
                'maximum_power_disagreement must be null or finite/non-negative'
            )

    def to_metadata(self) -> Dict[str, object]:
        return {
            'enabled': bool(self.enabled),
            'unavailable_model_policy': self.unavailable_model_policy,
            'power_disagreement': {
                'enabled': bool(self.power_disagreement_enabled),
                'power_floor_w': float(self.power_floor_w),
                'calibration_quantile': float(self.power_disagreement_quantile),
                'maximum_normalized_component_disagreement': (
                    self.maximum_power_disagreement
                ),
            },
            'state_safety': {
                'enabled': bool(self.state_safety_enabled),
                'calibration_quantile': float(self.state_margin_quantile),
                'out_of_band_policy': self.state_out_of_band_policy,
            },
        }


@dataclass(frozen=True)
class ActionModelScreenDecision:
    """Independent-model assessment of one exact action in one context."""

    encoded_control: tuple[int, ...]
    eligible: bool
    rejection_reasons: tuple[str, ...]
    notes: tuple[str, ...]
    affine_power_components_w: tuple[float, ...] | None
    evaluator_power_components_w: tuple[float, ...] | None
    normalized_component_disagreement: tuple[float, ...] | None
    maximum_normalized_component_disagreement: float | None
    evaluator_next_state: tuple[float, ...] | None
    state_safety: Dict[str, object]

    def to_metadata(self) -> Dict[str, object]:
        return {
            'encoded_control': list(self.encoded_control),
            'eligible': bool(self.eligible),
            'rejection_reasons': list(self.rejection_reasons),
            'notes': list(self.notes),
            'affine_power_components_w': self.affine_power_components_w,
            'evaluator_power_components_w': self.evaluator_power_components_w,
            'normalized_component_disagreement': (
                self.normalized_component_disagreement
            ),
            'maximum_normalized_component_disagreement': (
                self.maximum_normalized_component_disagreement
            ),
            'evaluator_next_state': self.evaluator_next_state,
            'state_safety': self.state_safety,
        }


class ActionModelScreen:
    """Validation-calibrated independent-model eligibility screen."""

    def __init__(
        self,
        *,
        config: ActionModelScreenConfig,
        state_constraints: Sequence[Mapping],
        power_names: Sequence[str],
        state_names: Sequence[str],
        disagreement_threshold: float | None,
        upper_state_margins: np.ndarray,
        lower_state_margins: np.ndarray,
        calibration_metadata: Dict[str, object],
    ):
        self.config = config
        self.state_constraints = tuple(
            OmegaConf.to_container(value, resolve=True)
            if OmegaConf.is_config(value)
            else dict(value)
            for value in state_constraints
        )
        self.power_names = tuple(str(value) for value in power_names)
        self.state_names = tuple(str(value) for value in state_names)
        self.disagreement_threshold = disagreement_threshold
        self.upper_state_margins = np.asarray(upper_state_margins, dtype=float)
        self.lower_state_margins = np.asarray(lower_state_margins, dtype=float)
        self.calibration_metadata = dict(calibration_metadata)
        if len(self.state_constraints) != len(self.state_names):
            raise ValueError('State constraints must align with state_names')
        if self.upper_state_margins.shape != (len(self.state_names),):
            raise ValueError('Upper state margins must align with state_names')
        if self.lower_state_margins.shape != (len(self.state_names),):
            raise ValueError('Lower state margins must align with state_names')

    @staticmethod
    def _available(
        evaluator,
        *,
        require_power: bool = True,
        require_state: bool = True,
    ) -> bool:
        """Whether the evaluator supplies every preview required by the config.

        A state-only safety screen must not be disabled merely because the
        evaluator has no direct power preview (and vice versa).  In
        particular, a configuration with both checks disabled is a legitimate
        no-op even for a sequence evaluator whose predictions require a
        rollout session.
        """
        return (
            (not require_power or callable(getattr(evaluator, 'predict_power', None)))
            and (
                not require_state
                or callable(getattr(evaluator, 'predict_state', None))
            )
        )

    @staticmethod
    def _requires_preview(config: ActionModelScreenConfig) -> bool:
        return bool(
            config.power_disagreement_enabled or config.state_safety_enabled
        )

    @classmethod
    def calibrate(
        cls,
        data,
        *,
        validation_rows,
        affine_power_model,
        validation_evaluator,
        state_constraints: Sequence[Mapping],
        config=None,
    ) -> 'ActionModelScreen':
        """Calibrate thresholds from a train-only model on validation rows."""
        parsed = (
            config
            if isinstance(config, ActionModelScreenConfig)
            else ActionModelScreenConfig.from_mapping(config)
        )
        rows = data.valid_transition_rows(validation_rows)
        state_count = len(data.state_names)
        zeros = np.zeros(state_count, dtype=float)
        metadata: Dict[str, object] = {
            'calibration_scope': 'chronological validation rows only',
            'validation_transition_count': int(len(rows)),
            'uses_test_targets': False,
            'evaluator_supports_direct_preview': cls._available(
                validation_evaluator,
                require_power=parsed.power_disagreement_enabled,
                require_state=parsed.state_safety_enabled,
            ),
        }
        if not parsed.enabled:
            return cls(
                config=parsed,
                state_constraints=state_constraints,
                power_names=data.power_names,
                state_names=data.state_names,
                disagreement_threshold=None,
                upper_state_margins=zeros,
                lower_state_margins=zeros,
                calibration_metadata={**metadata, 'status': 'disabled'},
            )
        if not len(rows):
            metadata.update({'status': 'unavailable', 'reason': 'no_validation_rows'})
            return cls(
                config=parsed,
                state_constraints=state_constraints,
                power_names=data.power_names,
                state_names=data.state_names,
                disagreement_threshold=None,
                upper_state_margins=zeros,
                lower_state_margins=zeros,
                calibration_metadata=metadata,
            )
        if not cls._requires_preview(parsed):
            metadata.update({'status': 'calibrated', 'reason': 'no_enabled_checks'})
            return cls(
                config=parsed,
                state_constraints=state_constraints,
                power_names=data.power_names,
                state_names=data.state_names,
                disagreement_threshold=None,
                upper_state_margins=zeros,
                lower_state_margins=zeros,
                calibration_metadata=metadata,
            )
        if not cls._available(
            validation_evaluator,
            require_power=parsed.power_disagreement_enabled,
            require_state=parsed.state_safety_enabled,
        ):
            metadata.update({
                'status': 'unavailable',
                'reason': 'evaluator_has_no_stateless_predict_state_and_predict_power',
            })
            return cls(
                config=parsed,
                state_constraints=state_constraints,
                power_names=data.power_names,
                state_names=data.state_names,
                disagreement_threshold=None,
                upper_state_margins=zeros,
                lower_state_margins=zeros,
                calibration_metadata=metadata,
            )

        states, controls, disturbances, next_states, _ = data.model_arrays_for_rows(rows)
        affine_power = None
        evaluator_power = None
        evaluator_state = None
        per_row_score = None
        threshold = None
        if parsed.power_disagreement_enabled:
            row_predictor = getattr(
                affine_power_model, 'predict_components_for_rows', None,
            )
            affine_power = np.asarray(
                row_predictor(data, rows) if row_predictor is not None
                else affine_power_model.predict_components(states, controls, disturbances),
                dtype=float,
            )
            evaluator_power = np.asarray(
                validation_evaluator.predict_power(states, controls, disturbances),
                dtype=float,
            )
            if (
                affine_power.shape != evaluator_power.shape
                or affine_power.shape != (len(rows), len(data.power_names))
                or not np.isfinite(affine_power).all()
                or not np.isfinite(evaluator_power).all()
            ):
                raise ValueError(
                    'Action model screen calibration power predictions must be '
                    'finite and align with configured power outputs'
                )
            normalized = np.abs(affine_power - evaluator_power) / np.maximum(
                np.abs(evaluator_power), float(parsed.power_floor_w)
            )
            per_row_score = normalized.max(axis=1)
            threshold = (
                float(parsed.maximum_power_disagreement)
                if parsed.maximum_power_disagreement is not None
                else float(
                    np.quantile(per_row_score, parsed.power_disagreement_quantile)
                )
            )
        upper_margins = np.zeros(state_count, dtype=float)
        lower_margins = np.zeros(state_count, dtype=float)
        if parsed.state_safety_enabled:
            evaluator_state = np.asarray(
                validation_evaluator.predict_state(states, controls, disturbances),
                dtype=float,
            )
            if (
                evaluator_state.shape != next_states.shape
                or not np.isfinite(evaluator_state).all()
            ):
                raise ValueError(
                    'Action model screen calibration state predictions must be '
                    'finite and align with next-state targets'
                )
            for position, specification in enumerate(state_constraints):
                spec = (
                    OmegaConf.to_container(specification, resolve=True)
                    if OmegaConf.is_config(specification)
                    else dict(specification)
                )
                if spec.get('upper') is not None:
                    upper_margins[position] = _positive_quantile(
                        next_states[:, position] - evaluator_state[:, position],
                        parsed.state_margin_quantile,
                    )
                if spec.get('lower') is not None:
                    lower_margins[position] = _positive_quantile(
                        evaluator_state[:, position] - next_states[:, position],
                        parsed.state_margin_quantile,
                    )
        metadata.update({
            'status': 'calibrated',
            'power_disagreement_threshold': threshold,
            'validation_power_disagreement_quantiles': (
                None
                if per_row_score is None
                else {
                    'p50': float(np.quantile(per_row_score, 0.50)),
                    'p90': float(np.quantile(per_row_score, 0.90)),
                    'p95': float(np.quantile(per_row_score, 0.95)),
                    'p99': float(np.quantile(per_row_score, 0.99)),
                }
            ),
            'upper_state_margin_by_name': dict(
                zip(data.state_names, upper_margins.tolist())
            ),
            'lower_state_margin_by_name': dict(
                zip(data.state_names, lower_margins.tolist())
            ),
            'power_floor_w': float(parsed.power_floor_w),
        })
        return cls(
            config=parsed,
            state_constraints=state_constraints,
            power_names=data.power_names,
            state_names=data.state_names,
            disagreement_threshold=threshold,
            upper_state_margins=upper_margins,
            lower_state_margins=lower_margins,
            calibration_metadata=metadata,
        )

    def _unavailable_decision(self, vector: np.ndarray, reason: str):
        reject = bool(
            self.config.enabled
            and self.config.unavailable_model_policy == 'reject'
        )
        return ActionModelScreenDecision(
            encoded_control=tuple(np.rint(vector).astype(int).tolist()),
            eligible=not reject,
            rejection_reasons=(('independent_model_preview_unavailable',) if reject else ()),
            notes=(reason,),
            affine_power_components_w=None,
            evaluator_power_components_w=None,
            normalized_component_disagreement=None,
            maximum_normalized_component_disagreement=None,
            evaluator_next_state=None,
            state_safety={'status': 'not_evaluated'},
        )

    def _state_safety(self, predicted_state: np.ndarray) -> tuple[Dict[str, object], list[str]]:
        """Evaluate all conservative state bounds for one batched prediction."""
        state_safety: Dict[str, object] = {}
        reasons = []
        for position, specification in enumerate(self.state_constraints):
            name = str(specification['name'])
            lower = specification.get('lower')
            upper = specification.get('upper')
            lower_margin = float(self.lower_state_margins[position])
            upper_margin = float(self.upper_state_margins[position])
            lower_safe = (
                lower is None
                or predicted_state[position] >= float(lower) + lower_margin - 1e-12
            )
            upper_safe = (
                upper is None
                or predicted_state[position] <= float(upper) - upper_margin + 1e-12
            )
            state_safety[name] = {
                'predicted_next_state': float(predicted_state[position]),
                'lower': None if lower is None else float(lower),
                'upper': None if upper is None else float(upper),
                'lower_margin': lower_margin,
                'upper_margin': upper_margin,
                'safe_after_margin': bool(lower_safe and upper_safe),
            }
            if not (lower_safe and upper_safe):
                reasons.append(f'evaluator_state_safety_margin_failed:{name}')
        return state_safety, reasons

    @property
    def requires_reference_control(self) -> bool:
        return bool(
            self.config.enabled
            and self.config.state_safety_enabled
            and self.config.state_out_of_band_policy == 'non_worsening'
        )

    def _state_violation(self, predicted_state: np.ndarray) -> np.ndarray:
        """Per-state distance outside the margin-adjusted band (0 when safe)."""
        violation = np.zeros(len(self.state_constraints), dtype=float)
        for position, specification in enumerate(self.state_constraints):
            lower = specification.get('lower')
            upper = specification.get('upper')
            value = float(predicted_state[position])
            if lower is not None:
                violation[position] += max(
                    0.0, float(lower) + float(self.lower_state_margins[position]) - value
                )
            if upper is not None:
                violation[position] += max(
                    0.0, value - (float(upper) - float(self.upper_state_margins[position]))
                )
        return violation

    def evaluate_candidates(
        self,
        state,
        disturbance,
        controls,
        *,
        evaluator,
        reference_control=None,
    ) -> list[ActionModelScreenDecision]:
        """Screen a candidate matrix in one stateless evaluator batch.

        The previous implementation called two tree-model previews for every
        candidate.  Counterfactual evaluation has thousands of contexts, so
        batching is essential: it preserves the exact per-action decision but
        avoids turning a 33-action gate into 66 small estimator calls per row.
        """
        control_matrix = np.asarray(controls, dtype=float)
        if control_matrix.ndim == 1:
            control_matrix = control_matrix.reshape(1, -1)
        if control_matrix.ndim != 2:
            raise ValueError('Candidate controls must be a two-dimensional matrix')
        if not len(control_matrix):
            return []
        vectors = [row.reshape(-1) for row in control_matrix]
        keys = [tuple(np.rint(vector).astype(int).tolist()) for vector in vectors]
        count = len(vectors)

        def no_op(note: str, safety: Dict[str, object]):
            return [
                ActionModelScreenDecision(
                    encoded_control=key,
                    eligible=True,
                    rejection_reasons=(),
                    notes=(note,),
                    affine_power_components_w=None,
                    evaluator_power_components_w=None,
                    normalized_component_disagreement=None,
                    maximum_normalized_component_disagreement=None,
                    evaluator_next_state=None,
                    state_safety=safety,
                )
                for key in keys
            ]

        if not self.config.enabled:
            return no_op('action_model_screen_disabled', {'status': 'disabled'})
        if not self._requires_preview(self.config):
            return no_op('no_model_screen_checks_enabled', {'status': 'not_configured'})
        if not self._available(
            evaluator,
            require_power=self.config.power_disagreement_enabled,
            require_state=self.config.state_safety_enabled,
        ):
            return [
                self._unavailable_decision(
                    vector,
                    'evaluator_has_no_stateless_predict_state_and_predict_power',
                )
                for vector in vectors
            ]
        if self.calibration_metadata.get('status') != 'calibrated':
            return [
                self._unavailable_decision(
                    vector,
                    'screen_calibration_status='
                    f"{self.calibration_metadata.get('status')}",
                )
                for vector in vectors
            ]

        state_vector = np.asarray(state, dtype=float).reshape(-1)
        disturbance_vector = np.asarray(disturbance, dtype=float).reshape(-1)
        if state_vector.shape != (len(self.state_names),):
            raise ValueError(
                'State width does not match the action-model screen state names'
            )
        state_matrix = np.repeat(state_vector[None, :], count, axis=0)
        disturbance_matrix = np.repeat(
            disturbance_vector[None, :],
            count,
            axis=0,
        )
        affine = None
        evaluator_power = None
        evaluator_state = None
        normalized = None
        maximum = None
        if self.config.power_disagreement_enabled:
            try:
                affine = np.asarray(
                    self._affine_power_model.predict_components(
                        state_matrix,
                        control_matrix,
                        disturbance_matrix,
                    ),
                    dtype=float,
                )
            except AttributeError:
                return [
                    self._unavailable_decision(
                        vector,
                        'affine_power_model_not_bound',
                    )
                    for vector in vectors
                ]
            evaluator_power = np.asarray(
                evaluator.predict_power(
                    state_matrix,
                    control_matrix,
                    disturbance_matrix,
                ),
                dtype=float,
            )
            expected_power_shape = (count, len(self.power_names))
            if (
                affine.shape != expected_power_shape
                or evaluator_power.shape != expected_power_shape
                or not np.isfinite(affine).all()
                or not np.isfinite(evaluator_power).all()
            ):
                return [
                    self._unavailable_decision(
                        vector,
                        'nonfinite_or_misaligned_power_preview',
                    )
                    for vector in vectors
                ]
            normalized = np.abs(affine - evaluator_power) / np.maximum(
                np.abs(evaluator_power),
                float(self.config.power_floor_w),
            )
            maximum = normalized.max(axis=1)
        if self.config.state_safety_enabled:
            evaluator_state = np.asarray(
                evaluator.predict_state(
                    state_matrix,
                    control_matrix,
                    disturbance_matrix,
                ),
                dtype=float,
            )
            expected_state_shape = (count, len(self.state_names))
            if (
                evaluator_state.shape != expected_state_shape
                or not np.isfinite(evaluator_state).all()
            ):
                return [
                    self._unavailable_decision(
                        vector,
                        'nonfinite_or_misaligned_state_preview',
                    )
                    for vector in vectors
                ]

        state_checks = (
            None
            if evaluator_state is None
            else [self._state_safety(evaluator_state[position]) for position in range(count)]
        )
        relaxed = False
        if (
            state_checks is not None
            and self.requires_reference_control
            and reference_control is not None
            and not any(not state_reasons for _, state_reasons in state_checks)
        ):
            # No candidate can satisfy the in-band rule (the measured state is
            # already outside it).  Holding the reference action is what the
            # fallback would do, so accept only candidates that do not worsen
            # any state's predicted violation relative to that fallback.
            reference_state = np.asarray(
                evaluator.predict_state(
                    state_vector[None, :],
                    np.asarray(reference_control, dtype=float).reshape(1, -1),
                    disturbance_vector[None, :],
                ),
                dtype=float,
            )[0]
            reference_violation = self._state_violation(reference_state)
            relaxed = True
            relaxed_checks = []
            for position in range(count):
                safety, _ = state_checks[position]
                violation = self._state_violation(evaluator_state[position])
                state_reasons = [
                    f'evaluator_state_non_worsening_failed:{self.state_names[index]}'
                    for index in np.flatnonzero(
                        violation > reference_violation + 1e-9
                    )
                ]
                safety = {
                    **safety,
                    'out_of_band_policy': 'non_worsening',
                    'reference_violation': reference_violation.tolist(),
                    'candidate_violation': violation.tolist(),
                }
                relaxed_checks.append((safety, state_reasons))
            state_checks = relaxed_checks

        results = []
        for position, key in enumerate(keys):
            reasons = []
            if (
                maximum is not None
                and self.disagreement_threshold is not None
                and maximum[position] > self.disagreement_threshold + 1e-12
            ):
                reasons.append('affine_evaluator_power_disagreement_above_threshold')
            if state_checks is None:
                state_safety = {'status': 'disabled'}
            else:
                state_safety, state_reasons = state_checks[position]
                reasons.extend(state_reasons)
            results.append(
                ActionModelScreenDecision(
                    encoded_control=key,
                    eligible=not reasons,
                    rejection_reasons=tuple(reasons),
                    notes=(
                        ('batched_candidate_preview', 'state_safety_relaxed_non_worsening')
                        if relaxed
                        else ('batched_candidate_preview',)
                    ),
                    affine_power_components_w=(
                        None
                        if affine is None
                        else tuple(float(value) for value in affine[position])
                    ),
                    evaluator_power_components_w=(
                        None
                        if evaluator_power is None
                        else tuple(
                            float(value) for value in evaluator_power[position]
                        )
                    ),
                    normalized_component_disagreement=(
                        None
                        if normalized is None
                        else tuple(float(value) for value in normalized[position])
                    ),
                    maximum_normalized_component_disagreement=(
                        None
                        if maximum is None
                        else float(maximum[position])
                    ),
                    evaluator_next_state=(
                        None
                        if evaluator_state is None
                        else tuple(
                            float(value) for value in evaluator_state[position]
                        )
                    ),
                    state_safety=state_safety,
                )
            )
        return results

    def evaluate(
        self, state, disturbance, control, *, evaluator, reference_control=None,
    ) -> ActionModelScreenDecision:
        """Screen one action through the same batched implementation."""
        return self.evaluate_candidates(
            state,
            disturbance,
            np.asarray(control, dtype=float).reshape(1, -1),
            evaluator=evaluator,
            reference_control=reference_control,
        )[0]

    def bind_affine_power_model(self, affine_power_model) -> 'ActionModelScreen':
        """Bind the controller model used for candidate disagreement checks."""
        self._affine_power_model = affine_power_model
        return self

    def to_metadata(self) -> Dict[str, object]:
        return {
            'protocol': {
                'name': 'validation_calibrated_independent_model_action_screen',
                'action_selection_role': 'pre-MILP first-action eligibility only',
                'does_not_use_test_targets': True,
                'does_not_mutate_evaluator_rollout_session': True,
            },
            'config': self.config.to_metadata(),
            'calibration': self.calibration_metadata,
        }


__all__ = [
    'ActionModelScreen',
    'ActionModelScreenConfig',
    'ActionModelScreenDecision',
]
