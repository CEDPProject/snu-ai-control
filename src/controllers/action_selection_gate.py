"""Compose train-support and independent-model action eligibility checks.

This module is intentionally independent of the MILP formulation.  It turns
per-action audit decisions into an allowed first-action set; the caller then
translates rejected candidates into first-step MILP no-good cuts.  Keeping that
translation outside the gate makes it possible to use the same policy in both
closed-loop rollout and counterfactual report evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from controllers.action_model_screen import ActionModelScreenDecision
from controllers.action_trust import ActionTrustDecision, ActionTrustEligibility
from controllers.report_action_effect_gate import ReportActionEffectDecision


def _key(vector) -> tuple[int, ...]:
    return tuple(np.rint(np.asarray(vector, dtype=float)).astype(int).tolist())


@dataclass(frozen=True)
class ActionSelectionDecision:
    """Merged eligibility result for a candidate exact joint action."""

    encoded_control: tuple[int, ...]
    eligible: bool
    rejection_reasons: tuple[str, ...]
    trust: ActionTrustDecision
    model_screen: ActionModelScreenDecision | None
    report_action_effect: ReportActionEffectDecision | None

    def to_metadata(self) -> Dict[str, object]:
        return {
            'encoded_control': list(self.encoded_control),
            'action': self.trust.action,
            'eligible': bool(self.eligible),
            'rejection_reasons': list(self.rejection_reasons),
            'trust': self.trust.to_metadata(),
            'model_screen': (
                None if self.model_screen is None else self.model_screen.to_metadata()
            ),
            'report_action_effect': (
                None
                if self.report_action_effect is None
                else self.report_action_effect.to_metadata()
            ),
        }


class ActionSelectionGate:
    """Evaluate the exact same first-action candidate pool at every minute."""

    def __init__(
        self,
        trust: ActionTrustEligibility,
        model_screen=None,
        report_action_effect_gate=None,
    ):
        self.trust = trust
        self.model_screen = model_screen
        self.report_action_effect_gate = report_action_effect_gate

    @property
    def requires_reference_control(self) -> bool:
        return bool(
            self.report_action_effect_gate is not None
            or getattr(self.model_screen, 'requires_reference_control', False)
        )

    @property
    def candidate_controls(self) -> np.ndarray:
        return self.trust.candidate_controls.copy()

    def evaluate(self, state, disturbance, *, evaluator, reference_control=None):
        """Return allowed vectors plus exhaustive action-level audit rows."""
        trust_decisions = self.trust.evaluate_candidates(state, disturbance)
        trusted_vectors = [
            np.asarray(decision.encoded_control, dtype=float)
            for decision in trust_decisions
            if decision.encoded_control is not None and decision.eligible
        ]
        screen_by_key = {}
        if trusted_vectors and self.model_screen is not None:
            screen_kwargs = {'evaluator': evaluator}
            if getattr(self.model_screen, 'requires_reference_control', False):
                screen_kwargs['reference_control'] = reference_control
            screen_decisions = self.model_screen.evaluate_candidates(
                state,
                disturbance,
                np.asarray(trusted_vectors, dtype=float),
                **screen_kwargs,
            )
            if len(screen_decisions) != len(trusted_vectors):
                raise RuntimeError(
                    'Action model screen must return one decision per candidate'
                )
            screen_by_key = {
                _key(vector): screen
                for vector, screen in zip(trusted_vectors, screen_decisions)
            }
        report_effect_by_key = {}
        if trusted_vectors and self.report_action_effect_gate is not None:
            if reference_control is None:
                raise ValueError(
                    'Report action-effect gating requires a reference applied action'
                )
            report_decisions = self.report_action_effect_gate.evaluate_candidates(
                state,
                disturbance,
                np.asarray(trusted_vectors, dtype=float),
                reference_control=reference_control,
            )
            if len(report_decisions) != len(trusted_vectors):
                raise RuntimeError(
                    'Report action-effect gate must return one decision per candidate'
                )
            report_effect_by_key = {
                _key(vector): decision
                for vector, decision in zip(trusted_vectors, report_decisions)
            }
        decisions = []
        allowed = []
        for trust_decision in trust_decisions:
            if trust_decision.encoded_control is None:
                # This cannot occur for a configured candidate, but preserving
                # it makes the audit total rather than silently dropping it.
                continue
            vector = np.asarray(trust_decision.encoded_control, dtype=float)
            screen_decision = None
            report_effect_decision = None
            reasons = list(trust_decision.rejection_reasons)
            if trust_decision.eligible and self.model_screen is not None:
                screen_decision = screen_by_key[_key(vector)]
                reasons.extend(screen_decision.rejection_reasons)
            if trust_decision.eligible and self.report_action_effect_gate is not None:
                report_effect_decision = report_effect_by_key[_key(vector)]
                reasons.extend(report_effect_decision.rejection_reasons)
            eligible = not reasons
            decision = ActionSelectionDecision(
                encoded_control=_key(vector),
                eligible=bool(eligible),
                rejection_reasons=tuple(reasons),
                trust=trust_decision,
                model_screen=screen_decision,
                report_action_effect=report_effect_decision,
            )
            decisions.append(decision)
            if eligible:
                allowed.append(vector)
        allowed_matrix = (
            np.asarray(allowed, dtype=float)
            if allowed
            else np.empty((0, self.trust.data.encoder.n_features), dtype=float)
        )
        return allowed_matrix, decisions

    def to_metadata(self) -> Dict[str, object]:
        return {
            'protocol': {
                'name': 'train_local_support_plus_independent_model_first_action_gate',
                'applies_to': 'first action only at each receding-horizon solve',
                'test_target_usage': False,
            },
            'action_trust': self.trust.to_metadata(),
            'model_screen': (
                None
                if self.model_screen is None
                else self.model_screen.to_metadata()
            ),
            'report_action_effect_gate': (
                None
                if self.report_action_effect_gate is None
                else self.report_action_effect_gate.to_metadata()
            ),
        }


__all__ = ['ActionSelectionDecision', 'ActionSelectionGate']
