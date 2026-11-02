"""Smoothed decision-focused fine-tuning against an explicitly labelled surrogate.

This implements finite-candidate expected teacher cost, not differentiable MIP
or a reproduction of the 2026 paper. No counterfactual measurement is invented.
"""

from copy import deepcopy
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from .base import stage_cost
from .neural import NeuralHistoryModel


def torch_stage_cost(x, p, constraints):
    cost = p.sum(-1) / 60
    for j, item in enumerate(constraints):
        penalty = float(item.get('penalty', 1000))
        if item.get('lower') is not None:
            cost = cost + penalty * torch.relu(float(item['lower']) - x[:, j])
        if item.get('upper') is not None:
            cost = cost + penalty * torch.relu(x[:, j] - float(item['upper']))
    return cost


class DecisionFocusedModel(NeuralHistoryModel):
    family = 'decision_focused'

    def fit(self, train, validation=None):
        super().fit(train, validation)
        if not hasattr(self, 'decision_constraints'):
            raise ValueError('Decision-focused fit requires constraints and allowed actions')
        split = max(2, int(len(train.rows) * 0.7))
        if split >= len(train.rows) - 1:
            raise ValueError('Insufficient train data for disjoint DFL teacher/context split')
        teacher_data = train.take(np.arange(split))
        teacher = ExtraTreesRegressor(n_estimators=24, min_samples_leaf=5,
                                      random_state=self.seed+731, n_jobs=1)
        teacher.fit(np.concatenate([teacher_data.context, teacher_data.controls,
                                    teacher_data.disturbances], axis=1),
                    np.concatenate([teacher_data.next_states - teacher_data.context[:, :self.spec.nx],
                                    teacher_data.powers], axis=1))
        rng = np.random.default_rng(self.seed+732)
        ids = rng.choice(np.arange(split, len(train.rows)),
                         min(int(self.parameters.get('decision_samples', 512)),
                             len(train.rows)-split), replace=False)
        decision = train.take(np.sort(ids))
        actions = self.decision_actions
        n, k = len(ids), len(actions)
        h = int(self.parameters.get('decision_horizon', 5))
        if h < 1 or k < 2:
            raise ValueError('DFL requires positive horizon and at least two candidate actions')
        ctx = np.repeat(decision.context, k, axis=0)
        z = np.tile(actions, (n, 1))
        d = np.repeat(decision.disturbances, k, axis=0)
        costs = np.zeros(n*k)
        for _ in range(h):
            out = teacher.predict(np.concatenate([ctx, z, d], axis=1))
            nx = ctx[:, :self.spec.nx] + out[:, :self.spec.nx]
            costs += stage_cost(nx, np.maximum(out[:, self.spec.nx:], 0), self.decision_constraints)
            ctx = self.spec.advance(ctx, z, nx)
        costs = costs.reshape(n, k)
        # Center/scale each decision context; preserves teacher action ordering.
        norm = np.maximum(costs.std(axis=1, keepdims=True), 1.0)
        teacher_cost = (costs-costs.min(axis=1, keepdims=True))/norm
        ct = torch.as_tensor(decision.context, dtype=torch.float32, device=self.device)
        dt = torch.as_tensor(decision.disturbances, dtype=torch.float32, device=self.device)
        at = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        ref = torch.as_tensor(teacher_cost, dtype=torch.float32, device=self.device)
        scale = torch.as_tensor(norm, dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam(self.network.parameters(),
                                     lr=float(self.parameters.get('decision_learning_rate', 1e-4)))
        temperature = float(self.parameters.get('decision_temperature', 1.0))
        if temperature <= 0:
            raise ValueError('decision_temperature must be positive')
        batch_size = int(self.parameters.get('decision_batch_size', 32))
        # Supervised anchor and validation guard prevent unbounded cost hacking.
        val = validation if validation is not None else decision
        vi = np.linspace(0, len(val.rows)-1, min(4096, len(val.rows)), dtype=int)
        val = val.take(vi)
        vt = [torch.as_tensor(a, dtype=torch.float32, device=self.device)
              for a in (val.context, val.controls, val.disturbances)]
        vy = torch.as_tensor(np.concatenate([
            val.next_states-val.context[:, :self.spec.nx], val.powers], axis=1),
            dtype=torch.float32, device=self.device)
        with torch.no_grad():
            reference_validation = (((self._raw_torch(*vt)-vy)/self.target_scale)**2).mean().item()
        original = deepcopy(self.network.state_dict())
        best, best_loss = original, float('inf')
        for epoch in range(int(self.parameters.get('decision_epochs', 5))):
            losses = []
            for start in range(0, n, batch_size):
                end = min(start+batch_size, n)
                b = end-start
                c = ct[start:end].repeat_interleave(k, dim=0)
                dc = dt[start:end].repeat_interleave(k, dim=0)
                ac = at.repeat(b, 1)
                predicted_cost = torch.zeros(b*k, device=self.device)
                for _ in range(h):
                    xn, p = self.predict_torch(c, ac, dc)
                    predicted_cost = predicted_cost + torch_stage_cost(xn, p, self.decision_constraints)
                    c = torch.cat([xn, c[:, :self.spec.nx], ac,
                                   self.spec.decay*c[:, -self.spec.nz:] + (1-self.spec.decay)*ac], dim=1)
                logits = -predicted_cost.reshape(b, k) / scale[start:end] / temperature
                expected_cost = (torch.softmax(logits, dim=1) * ref[start:end]).sum(1).mean()
                rows = np.arange(start, end)
                obs = decision.take(rows)
                ot = [torch.as_tensor(a, dtype=torch.float32, device=self.device)
                      for a in (obs.context, obs.controls, obs.disturbances)]
                oy = torch.as_tensor(np.concatenate([
                    obs.next_states-obs.context[:, :self.spec.nx], obs.powers], axis=1),
                    dtype=torch.float32, device=self.device)
                anchor = (((self._raw_torch(*ot)-oy)/self.target_scale)**2).mean()
                loss = expected_cost + float(self.parameters.get('decision_anchor', 1.0))*anchor
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite DFL loss')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                losses.append(float(expected_cost.detach()))
            with torch.no_grad():
                vl = (((self._raw_torch(*vt)-vy)/self.target_scale)**2).mean().item()
            score = float(np.mean(losses))
            accepted = vl <= reference_validation * float(self.parameters.get('decision_validation_ratio', 1.25))
            self.training_log.append({'stage': 'decision_focused', 'epoch': epoch+1,
                                      'expected_teacher_cost_normalized': score,
                                      'validation_loss': vl, 'eligible_checkpoint': accepted})
            if accepted and score < best_loss:
                best_loss, best = score, deepcopy(self.network.state_dict())
        self.network.load_state_dict(best)
        self.dfl_metadata = {
            'objective': 'softmin expected held-action teacher cost + supervised anchor',
            'teacher': 'train-prefix-only ExtraTrees; not measured counterfactual truth',
            'teacher_max_row': int(teacher_data.rows.max()),
            'decision_min_row': int(decision.rows.min()),
            'decision_context_count': n, 'candidate_count': k, 'training_horizon': h,
            'fine_tuning_checkpoint_retained': best is not original,
            'not_an_exact_replication_of_differentiable_MIP': True,
        }
        self.teacher_rows = teacher_data.rows.copy()
        self.decision_rows = decision.rows.copy()
        self.network.eval()
        return self

    def metadata(self):
        return {**super().metadata(), 'decision_focused': self.dfl_metadata}
