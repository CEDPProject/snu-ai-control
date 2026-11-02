"""Block-bootstrap ensembles with individually persisted training lineage."""

import numpy as np
from .base import ControlModel


def block_bootstrap_indices(count, rng, block=128):
    block = min(int(block), count)
    starts = rng.integers(0, count - block + 1, int(np.ceil(count / block)))
    return np.concatenate([np.arange(i, i + block) for i in starts])[:count]


class EnsembleControlModel(ControlModel):
    def __init__(self, spec, parameters=None, seed=42, member_family='mode_history'):
        super().__init__(spec, parameters, seed)
        self.member_family = member_family
        self.family = 'scenario_ensemble' if member_family == 'mode_history' else 'neural_ensemble'
        self.supports_milp = member_family == 'mode_history'
        self.members = []

    def fit(self, train, validation=None):
        from .factory import create_control_model
        count = int(self.parameters.get('ensemble_size', 3))
        if count < 2:
            raise ValueError('An ensemble requires at least two independent members')
        rng = np.random.default_rng(self.seed)
        for i in range(count):
            indices = block_bootstrap_indices(len(train.rows), rng,
                                               self.parameters.get('bootstrap_block', 128))
            model = create_control_model(self.member_family, self.spec, self.parameters, self.seed + i + 1)
            model.fit(train.take(indices), validation)
            self.members.append(model)
        self.fit_rows = train.rows.copy()
        return self

    def scenario_predictions(self, context, control, disturbance):
        predictions = [m.predict(context, control, disturbance) for m in self.members]
        return np.stack([p[0] for p in predictions]), np.stack([p[1] for p in predictions])

    def predict(self, context, control, disturbance):
        x, p = self.scenario_predictions(context, control, disturbance)
        return x.mean(axis=0), p.mean(axis=0)

    def to_cpu(self):
        for m in self.members:
            if hasattr(m, 'to_cpu'):
                m.to_cpu()

    def metadata(self):
        return {**super().metadata(), 'member_family': self.member_family,
                'members': [m.metadata() for m in self.members],
                'uncertainty': 'block-bootstrap model disagreement; not calibrated probability'}
