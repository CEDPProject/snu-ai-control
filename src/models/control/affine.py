"""Baseline and mode-conditioned, action-history affine models."""

import numpy as np

from dynamics.affine import fit_affine_models, _fit_physical_ridge
from .base import ControlModel


class AffineControlModel(ControlModel):
    family = 'affine'
    supports_milp = True

    def fit(self, train, validation=None):
        s = self.spec
        self.state_model, self.power_model = fit_affine_models(
            train.context[:, :s.nx], train.controls, train.disturbances,
            train.next_states, train.powers,
            state_names=s.state_names, control_feature_names=s.feature_names,
            disturbance_names=s.disturbance_names, power_names=s.power_names,
            state_alpha=float(self.parameters.get('ridge_alpha', 1)),
            power_alpha=float(self.parameters.get('ridge_alpha', 1)),
        )
        self.fit_rows = train.rows.copy()
        return self

    def predict(self, context, control, disturbance):
        x = np.atleast_2d(context)[..., :self.spec.nx]
        return (self.state_model.predict(x, control, disturbance),
                self.power_model.predict_components(x, control, disturbance))

    def linear_coefficients(self, controls, disturbance):
        s, a, p = self.spec, self.state_model, self.power_model
        k = len(controls)
        A = np.zeros((k, s.nx, s.width))
        F = np.zeros((k, s.np, s.width))
        A[:, :, :s.nx] = a.A
        F[:, :, :s.nx] = p.F
        return A, controls @ a.B.T + disturbance @ a.E.T + a.c, F, (
            controls @ p.G.T + disturbance @ p.H.T + p.b)


class ModeHistoryModel(ControlModel):
    family = 'mode_history'
    supports_milp = True

    def _features(self, context, control, disturbance):
        x = context[:, :self.spec.nx]
        interactions = (control[:, self.mode_columns, None] * x[:, None, :]).reshape(len(x), -1)
        return np.concatenate([context, disturbance, control, interactions], axis=1)

    def fit(self, train, validation=None):
        s = self.spec
        self.mode_columns = [i for i, n in enumerate(s.feature_names)
                             if '운전모드=' in n or n.startswith('mode=')]
        if not self.mode_columns:
            self.mode_columns = list(range(s.nz))
        features = self._features(train.context, train.controls, train.disturbances)
        targets = np.concatenate([train.next_states - train.context[:, :s.nx], train.powers], axis=1)
        self.coefficients, self.intercept = _fit_physical_ridge(
            features, targets, float(self.parameters.get('ridge_alpha', 10)))
        self.fit_rows = train.rows.copy()
        return self

    def predict(self, context, control, disturbance):
        context, control, disturbance = map(np.atleast_2d, (context, control, disturbance))
        y = self._features(context, control, disturbance) @ self.coefficients.T + self.intercept
        return context[:, :self.spec.nx] + y[:, :self.spec.nx], np.maximum(y[:, self.spec.nx:], 0)

    def linear_coefficients(self, controls, disturbance):
        s = self.spec
        W = self.coefficients
        ns, nd = s.width, s.nd
        matrices = np.broadcast_to(W[:, :ns], (len(controls), len(W), ns)).copy()
        for pos, col in enumerate(self.mode_columns):
            interaction = W[:, ns+nd+s.nz+pos*s.nx:ns+nd+s.nz+(pos+1)*s.nx]
            matrices[:, :, :s.nx] += controls[:, col, None, None] * interaction
        offsets = controls @ W[:, ns+nd:ns+nd+s.nz].T + disturbance @ W[:, ns:ns+nd].T + self.intercept
        matrices[:, :s.nx, :s.nx] += np.eye(s.nx)
        return matrices[:, :s.nx], offsets[:, :s.nx], matrices[:, s.nx:], offsets[:, s.nx:]
