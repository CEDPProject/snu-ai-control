"""Frozen-report-model guard for conservative first MPC actions.

The affine controller remains the only model inside the MILP.  This module
evaluates a finite, train-supported first-action pool *before* each solve and
can reject an action whose frozen report-model prediction is too far from the
reference applied action.  It is therefore an auditable trust-region gate,
not a nonlinear objective secretly embedded in the MILP.

The report model is selected on chronological validation and refit before any
test action is evaluated.  Test power targets are never consumed here.
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
    raise TypeError('Report action-effect gate configuration must be a mapping')


def _action_key(vector) -> tuple[int, ...]:
    return tuple(np.rint(np.asarray(vector, dtype=float)).astype(int).tolist())


@dataclass(frozen=True)
class ReportActionEffectGateConfig:
    """Strict, validation-fixed limits on reported action changes."""

    enabled: bool = False
    unavailable_model_policy: str = 'reject'
    minimum_reference_power_w: float = 0.1
    maximum_total_relative_change_percent: float | None = None
    maximum_component_relative_change_percent: float | None = None

    @classmethod
    def from_mapping(cls, config=None) -> 'ReportActionEffectGateConfig':
        spec = _plain(config)
        result = cls(
            enabled=bool(spec.get('enabled', False)),
            unavailable_model_policy=str(
                spec.get('unavailable_model_policy', 'reject')
            ),
            minimum_reference_power_w=float(
                spec.get('minimum_reference_power_w', 0.1)
            ),
            maximum_total_relative_change_percent=(
                None
                if spec.get('maximum_total_relative_change_percent') is None
                else float(spec['maximum_total_relative_change_percent'])
            ),
            maximum_component_relative_change_percent=(
                None
                if spec.get('maximum_component_relative_change_percent') is None
                else float(spec['maximum_component_relative_change_percent'])
            ),
        )
        result._validate()
        return result

    def _validate(self) -> None:
        if self.unavailable_model_policy not in ('reject', 'monitor'):
            raise ValueError("unavailable_model_policy must be 'reject' or 'monitor'")
        if not np.isfinite(self.minimum_reference_power_w) or self.minimum_reference_power_w <= 0.0:
            raise ValueError('minimum_reference_power_w must be finite and positive')
        for name, value in (
            ('maximum_total_relative_change_percent', self.maximum_total_relative_change_percent),
            ('maximum_component_relative_change_percent', self.maximum_component_relative_change_percent),
        ):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f'{name} must be null or finite/non-negative')
        if self.enabled and (
            self.maximum_total_relative_change_percent is None
            and self.maximum_component_relative_change_percent is None
        ):
            raise ValueError(
                'An enabled report action-effect gate needs at least one maximum change limit'
            )

    def to_metadata(self) -> Dict[str, object]:
        return {
            'enabled': bool(self.enabled),
            'unavailable_model_policy': self.unavailable_model_policy,
            'minimum_reference_power_w': float(self.minimum_reference_power_w),
            'maximum_total_relative_change_percent': self.maximum_total_relative_change_percent,
            'maximum_component_relative_change_percent': self.maximum_component_relative_change_percent,
        }


@dataclass(frozen=True)
class ReportActionEffectDecision:
    encoded_control: tuple[int, ...]
    eligible: bool
    rejection_reasons: tuple[str, ...]
    notes: tuple[str, ...]
    reference_total_power_w: float | None
    candidate_total_power_w: float | None
    total_relative_change_percent: float | None
    component_relative_change_percent: tuple[float, ...] | None

    def to_metadata(self) -> Dict[str, object]:
        return {
            'encoded_control': list(self.encoded_control),
            'eligible': bool(self.eligible),
            'rejection_reasons': list(self.rejection_reasons),
            'notes': list(self.notes),
            'reference_total_power_w': self.reference_total_power_w,
            'candidate_total_power_w': self.candidate_total_power_w,
            'total_relative_change_percent': self.total_relative_change_percent,
            'component_relative_change_percent': self.component_relative_change_percent,
        }


class ReportActionEffectGate:
    """Evaluate action-effect limits with a model frozen before test use."""

    def __init__(self, report_model, power_names: Sequence[str], config=None):
        self.config = (
            config
            if isinstance(config, ReportActionEffectGateConfig)
            else ReportActionEffectGateConfig.from_mapping(config)
        )
        self.report_model = report_model
        self.power_names = tuple(str(value) for value in power_names)

    def evaluate_candidates(
        self,
        state,
        disturbance,
        candidates,
        *,
        reference_control,
    ) -> list[ReportActionEffectDecision]:
        controls = np.asarray(candidates, dtype=float)
        if controls.ndim != 2:
            raise ValueError('Report action-effect candidates must be a 2-D matrix')
        if not len(controls):
            return []
        reference = np.asarray(reference_control, dtype=float).reshape(1, -1)
        if reference.shape[1] != controls.shape[1]:
            raise ValueError('Reference action width must match candidate action width')
        state_row = np.asarray(state, dtype=float).reshape(1, -1)
        disturbance_row = np.asarray(disturbance, dtype=float).reshape(1, -1)
        states = np.repeat(state_row, len(controls), axis=0)
        disturbances = np.repeat(disturbance_row, len(controls), axis=0)
        try:
            reference_components = np.asarray(
                self.report_model.predict_components(state_row, reference, disturbance_row),
                dtype=float,
            )
            candidate_components = np.asarray(
                self.report_model.predict_components(states, controls, disturbances),
                dtype=float,
            )
        except Exception as error:
            if self.config.unavailable_model_policy == 'reject':
                return [
                    ReportActionEffectDecision(
                        encoded_control=_action_key(control),
                        eligible=False,
                        rejection_reasons=('report_action_effect_model_unavailable',),
                        notes=(f'{type(error).__name__}: {error}',),
                        reference_total_power_w=None,
                        candidate_total_power_w=None,
                        total_relative_change_percent=None,
                        component_relative_change_percent=None,
                    )
                    for control in controls
                ]
            return [
                ReportActionEffectDecision(
                    encoded_control=_action_key(control),
                    eligible=True,
                    rejection_reasons=(),
                    notes=(f'report model unavailable; monitor only: {type(error).__name__}',),
                    reference_total_power_w=None,
                    candidate_total_power_w=None,
                    total_relative_change_percent=None,
                    component_relative_change_percent=None,
                )
                for control in controls
            ]
        if (
            reference_components.shape != (1, len(self.power_names))
            or candidate_components.shape != (len(controls), len(self.power_names))
            or not np.isfinite(reference_components).all()
            or not np.isfinite(candidate_components).all()
        ):
            raise ValueError('Report action-effect predictions must be finite component vectors')

        floor = float(self.config.minimum_reference_power_w)
        reference_component = reference_components[0]
        reference_total = float(reference_component.sum())
        candidate_total = candidate_components.sum(axis=1)
        component_change = 100.0 * np.abs(candidate_components - reference_component) / np.maximum(
            np.abs(reference_component), floor
        )
        total_change = 100.0 * np.abs(candidate_total - reference_total) / max(
            abs(reference_total), floor
        )
        decisions = []
        for control, total, total_ratio, component_ratio in zip(
            controls, candidate_total, total_change, component_change
        ):
            reasons = []
            if (
                self.config.maximum_total_relative_change_percent is not None
                and total_ratio > self.config.maximum_total_relative_change_percent + 1e-12
            ):
                reasons.append('report_total_action_effect_above_limit')
            if (
                self.config.maximum_component_relative_change_percent is not None
                and np.any(
                    component_ratio
                    > self.config.maximum_component_relative_change_percent + 1e-12
                )
            ):
                reasons.append('report_component_action_effect_above_limit')
            decisions.append(
                ReportActionEffectDecision(
                    encoded_control=_action_key(control),
                    eligible=not reasons,
                    rejection_reasons=tuple(reasons),
                    notes=(),
                    reference_total_power_w=reference_total,
                    candidate_total_power_w=float(total),
                    total_relative_change_percent=float(total_ratio),
                    component_relative_change_percent=tuple(
                        float(value) for value in component_ratio.tolist()
                    ),
                )
            )
        return decisions

    def to_metadata(self) -> Dict[str, object]:
        return {
            'protocol': {
                'name': 'frozen_report_model_first_action_effect_gate',
                'applies_to': 'first action only at each receding-horizon solve',
                'test_target_usage': False,
                'interpretation': (
                    'Limits model-predicted action effect relative to the current '
                    'applied action; it does not verify real post-intervention power.'
                ),
            },
            'config': self.config.to_metadata(),
            'power_names': list(self.power_names),
            'report_model': self.report_model.to_metadata(),
        }


__all__ = [
    'ReportActionEffectDecision',
    'ReportActionEffectGate',
    'ReportActionEffectGateConfig',
]
