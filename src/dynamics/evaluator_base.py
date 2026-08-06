"""Common contracts for offline MPC plant evaluators.

The optimizer deliberately does not depend on these classes.  Evaluators are
counterfactual plant surrogates used by the backtest only, so they may be
nonlinear or history dependent without changing the affine MILP.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TransitionPredictions:
    """Measured targets and matching one-step evaluator predictions."""

    actual_states: np.ndarray
    predicted_states: np.ndarray
    actual_powers: np.ndarray
    predicted_powers: np.ndarray
    row_indices: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        pairs = (
            (self.actual_states, self.predicted_states, 'state'),
            (self.actual_powers, self.predicted_powers, 'power'),
        )
        sample_counts = set()
        for actual, predicted, label in pairs:
            actual_array = np.asarray(actual, dtype=float)
            predicted_array = np.asarray(predicted, dtype=float)
            if actual_array.ndim != 2 or actual_array.shape != predicted_array.shape:
                raise ValueError(
                    f'{label} transition arrays must be equal two-dimensional arrays'
                )
            if not (
                np.isfinite(actual_array).all()
                and np.isfinite(predicted_array).all()
            ):
                raise ValueError(f'{label} transition arrays must be finite')
            sample_counts.add(len(actual_array))
        if len(sample_counts) != 1 or not sample_counts or next(iter(sample_counts)) == 0:
            raise ValueError('Transition predictions must contain matching samples')
        if self.row_indices is not None:
            row_indices = np.asarray(self.row_indices)
            sample_count = next(iter(sample_counts))
            if row_indices.ndim != 1 or len(row_indices) != sample_count:
                raise ValueError(
                    'Transition row_indices must be one-dimensional and match samples'
                )
            if not np.issubdtype(row_indices.dtype, np.integer):
                raise ValueError('Transition row_indices must contain integers')
            if (row_indices < 0).any():
                raise ValueError('Transition row_indices cannot contain negatives')


@dataclass(frozen=True)
class RolloutHistory:
    """Measured observations strictly before a simulated rollout starts."""

    timestamps: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    disturbances: np.ndarray
    powers: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            len(self.timestamps),
            len(self.states),
            len(self.controls),
            len(self.disturbances),
            len(self.powers),
        }
        if len(lengths) != 1:
            raise ValueError('Every rollout-history array must have the same length')


class StatelessRolloutSession:
    """Give a stateless evaluator the same session API as sequence models."""

    def __init__(self, evaluator):
        self._evaluator = evaluator

    def step(
        self,
        state: np.ndarray,
        control: np.ndarray,
        disturbance: np.ndarray,
        *,
        timestamp=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del timestamp
        return self._evaluator.step(state, control, disturbance)


class StatelessEvaluatorMixin:
    """Default batch-validation and rollout behavior for memoryless models."""

    required_history_steps = 0

    def predict_transitions(self, data, start: int, end: int) -> TransitionPredictions:
        states, controls, disturbances, next_states, powers = data.model_arrays(
            start,
            end,
        )
        transition_mask = data.transition_mask(start, end)
        row_indices = np.arange(int(start), int(end), dtype=int)[transition_mask]
        return TransitionPredictions(
            actual_states=next_states,
            predicted_states=self.predict_state(states, controls, disturbances),
            actual_powers=powers,
            predicted_powers=self.predict_power(states, controls, disturbances),
            row_indices=row_indices,
        )

    def start_rollout(
        self,
        history: Optional[RolloutHistory] = None,
    ) -> StatelessRolloutSession:
        del history
        return StatelessRolloutSession(self)


def history_for_rollout(data, start: int, required_steps: int) -> Optional[RolloutHistory]:
    """Build a history containing only observations before ``start``.

    Sequence evaluators also use lagged power, so ``required_steps`` past rows
    are retained even though the current simulated row supplies the final
    element of the model's look-back window.
    """

    required_steps = int(required_steps)
    if required_steps <= 0:
        return None
    start = int(start)
    if start < 0 or start >= len(data.states):
        raise IndexError(f'Rollout start {start} is outside the data')
    history_start = max(0, start - required_steps)
    return RolloutHistory(
        timestamps=np.asarray(
            data.timestamps.iloc[history_start:start].to_numpy(),
        ),
        states=np.asarray(data.states[history_start:start], dtype=float).copy(),
        controls=np.asarray(data.controls[history_start:start], dtype=float).copy(),
        disturbances=np.asarray(
            data.disturbances[history_start:start],
            dtype=float,
        ).copy(),
        powers=np.asarray(data.powers[history_start:start], dtype=float).copy(),
    )


def start_rollout_session(evaluator, history: Optional[RolloutHistory] = None):
    """Return one isolated rollout session for any evaluator implementation."""

    start_rollout = getattr(evaluator, 'start_rollout', None)
    if start_rollout is None:
        return StatelessRolloutSession(evaluator)
    return start_rollout(history)
