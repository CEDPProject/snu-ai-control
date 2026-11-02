"""Physical-unit, causal contracts for interchangeable control models."""

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np


@dataclass
class ControlBatch:
    context: np.ndarray
    controls: np.ndarray
    disturbances: np.ndarray
    next_states: np.ndarray
    powers: np.ndarray
    rows: np.ndarray

    def take(self, indices):
        return ControlBatch(*(getattr(self, k)[indices] for k in (
            'context', 'controls', 'disturbances', 'next_states', 'powers', 'rows'
        )))


class ContextSpec:
    """[current x, previous x, previous action, EMA of previous actions].

    The memory contains strictly past commands, never the action being scored.
    """

    def __init__(self, data, decay=0.8):
        self.nx = len(data.state_names)
        self.nz = data.encoder.n_features
        self.nd = len(data.disturbance_names)
        self.np = len(data.power_names)
        self.width = 2 * self.nx + 2 * self.nz
        self.decay = float(decay)
        if not 0 <= self.decay < 1:
            raise ValueError('memory_decay must be in [0,1)')
        self.state_names = tuple(data.state_names)
        self.power_names = tuple(data.power_names)
        self.disturbance_names = tuple(data.disturbance_names)
        self.feature_names = tuple(data.encoder.feature_names)
        self.encoder = data.encoder

    def initial(self, state, previous_control=None):
        z = (self.encoder.encode_action() if previous_control is None
             else np.asarray(previous_control))
        return np.concatenate([state, state, z, z]).astype(float)

    def advance(self, context, control, next_state):
        context, control, next_state = map(np.asarray, (context, control, next_state))
        return np.concatenate([
            next_state, context[..., :self.nx], control,
            self.decay * context[..., -self.nz:] + (1 - self.decay) * control,
        ], axis=-1)

    def contexts(self, data):
        result = np.empty((len(data.states), self.width))
        result[0] = self.initial(data.states[0])
        valid = data.transition_mask(0, len(data.states) - 1)
        for i in range(1, len(result)):
            result[i] = (self.advance(result[i-1], data.controls[i-1], data.states[i])
                         if valid[i-1] else self.initial(data.states[i]))
        return result

    def batch(self, data, contexts, start, end):
        # end is an exclusive row boundary; targets must also be inside it.
        rows = data.valid_transition_rows(np.arange(start, end - 1))
        if not len(rows):
            raise ValueError(f'No contiguous training/evaluation transitions: {start}:{end}')
        return ControlBatch(contexts[rows], data.controls[rows],
                            data.disturbances[rows], data.states[rows+1],
                            data.powers[rows], rows)


class ControlModel:
    family = 'abstract'
    supports_milp = False

    def __init__(self, spec, parameters=None, seed=42):
        self.spec = spec
        self.parameters = dict(parameters or {})
        self.seed = int(seed)
        self.training_log = []
        self.fit_rows = None

    def predict(self, context, control, disturbance):
        raise NotImplementedError

    def scenario_predictions(self, context, control, disturbance):
        x, p = self.predict(context, control, disturbance)
        return x[None], p[None]

    def metadata(self):
        return {
            'family': self.family, 'supports_milp': self.supports_milp,
            'seed': self.seed, 'parameters': self.parameters,
            'context': '[x_t,x_previous,z_previous,EMA_previous_z]',
            'memory_decay': self.spec.decay,
            'state_names': self.spec.state_names,
            'power_names': self.spec.power_names,
            'control_feature_names': self.spec.feature_names,
            'disturbance_names': self.spec.disturbance_names,
            'training_transition_count': len(self.fit_rows),
            'training_log': self.training_log,
        }

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Only load locally trusted joblib files. CPU checkpoints are portable.
        if hasattr(self, 'to_cpu'):
            self.to_cpu()
        joblib.dump(self, directory / 'model.joblib')
        np.save(directory / 'training_rows.npy', self.fit_rows)
        (directory / 'model_metadata.json').write_text(
            json.dumps(self.metadata(), ensure_ascii=False, indent=2, allow_nan=False),
            encoding='utf-8',
        )


def load_control_model(directory):
    path = Path(directory)
    model = joblib.load(path / 'model.joblib' if path.is_dir() else path)
    if not isinstance(model, ControlModel):
        raise TypeError('Checkpoint is not a ControlModel')
    return model


def categorical_changes(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).sum(axis=-1) / 2


def stage_cost(next_states, powers, constraints):
    """Same Wh + physical-unit soft-constraint objective for every method."""
    value = np.maximum(powers, 0).sum(axis=-1) / 60
    for j, item in enumerate(constraints):
        penalty = float(item.get('penalty', 1000))
        if item.get('lower') is not None:
            value = value + penalty * np.maximum(float(item['lower']) - next_states[..., j], 0)
        if item.get('upper') is not None:
            value = value + penalty * np.maximum(next_states[..., j] - float(item['upper']), 0)
    return value
