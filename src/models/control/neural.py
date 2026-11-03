"""History-aware nonlinear dynamics and a transport-structured residual model."""

from copy import deepcopy

import numpy as np
from scipy.optimize import lsq_linear
import torch
from torch import nn

from .base import ControlModel


def contiguous_rollout_starts(rows, horizon):
    """학습 split 내부에서 gap/segment join을 건너지 않는 연속 transition만 허용."""
    rows = np.asarray(rows)
    if horizon < 1:
        raise ValueError('rollout horizon must be positive')
    starts = np.arange(max(0, len(rows)-horizon+1))
    if not len(starts):
        return starts
    return starts[np.all(
        rows[starts[:, None]+np.arange(horizon)] == rows[starts, None]+np.arange(horizon), axis=1)]


class SeparateResidualHeads(nn.Module):
    """상태와 각 전력 성분에 독립 MLP를 둔 ablation. 파라미터 수가 증가한다."""

    def __init__(self, width, hidden, nx, npower):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                          nn.GELU(), nn.Linear(hidden, output))
            for output in [nx, *([1]*npower)]])

    def forward(self, inputs):
        return torch.cat([head(inputs) for head in self.heads], dim=-1)


class TransportBackbone(nn.Module):
    """Nonnegative transport rates plus a learned source per state.

    This is a grey-box baseline, not a reproduction of PCNN or a proof of
    monotonicity for the unrestricted neural residual.
    """

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.register_buffer('state_coef', torch.zeros(spec.nx, 5))
        self.register_buffer('power_coef', torch.zeros(spec.np, 4))

    def _category(self, z, variable, category, default=0.0):
        name = f'{variable}={category}'
        if name not in self.spec.feature_names:
            return z[:, 0] * 0 + default
        return z[:, self.spec.feature_names.index(name)]

    def features(self, context, z, d):
        s = self.spec
        x = context[:, :s.nx]
        vent_on = self._category(z, '환기전원', 'ON')
        vent_fan = sum(float(v) / 6 * self._category(z, '환기풍량', str(v)) for v in (2, 4, 6))
        vent = vent_on * vent_fan
        clean = self._category(z, '환기운전모드', '공기청정') * vent
        fresh = vent * (1 - self._category(z, '환기운전모드', '공기청정'))
        hvac = self._category(z, '냉난방전원', 'ON')
        fan = sum(rate * self._category(z, '냉난방풍량', name)
                  for rate, name in ((0.4, '약풍'), (0.7, '중풍'), (1.0, '강풍')))
        setting = sum(v * self._category(z, '냉난방설정온도(℃)', str(v)) for v in (18, 22, 26))
        heating = self._category(z, '냉난방운전모드', '난방')
        temp_index = next((j for j, n in enumerate(s.state_names) if '온도' in n), None)
        temp = x[:, temp_index] if temp_index is not None else setting
        thermal = hvac * fan * (heating * torch.relu(setting - temp)
                              - (1 - heating) * torch.relu(temp - setting))
        terms = []
        for j, name in enumerate(s.state_names):
            outdoor_name = name.replace('실내', '실외')
            if outdoor_name in s.disturbance_names:
                gradient = d[:, s.disturbance_names.index(outdoor_name)] - x[:, j]
            elif '이산화탄소' in name:
                gradient = 420.0 - x[:, j]  # Explicit assumed outdoor CO2.
            else:
                gradient = -x[:, j] if '휘발성' in name else x[:, j] * 0
            removal = -clean * x[:, j] if ('미세먼지' in name or '휘발성' in name) else x[:, j] * 0
            conditioning = thermal if '온도' in name else x[:, j] * 0
            terms.append(torch.stack([gradient, fresh * gradient, removal,
                                      conditioning, torch.ones_like(gradient)], dim=1))
        state_features = torch.stack(terms, dim=1)
        power_features = []
        for name in s.power_names:
            is_vent = '환기' in name
            rate = vent if is_vent else hvac * fan
            demand = vent ** 3 if is_vent else torch.abs(thermal)
            power_features.append(torch.stack([torch.ones_like(rate), rate, rate**3, demand], dim=1))
        return state_features, torch.stack(power_features, dim=1)

    def fit(self, train):
        with torch.no_grad():
            sf, pf = self.features(*(torch.as_tensor(a, dtype=torch.float32) for a in
                                    (train.context, train.controls, train.disturbances)))
        targets = train.next_states - train.context[:, :self.spec.nx]
        for j in range(self.spec.nx):
            features = sf[:, j].numpy().astype(float)
            scale = np.maximum(np.linalg.norm(features, axis=0), 1e-6)
            lo = np.array([0, 0, 0, 0, -np.inf])
            fit = lsq_linear(features / scale, targets[:, j], bounds=(lo, np.inf))
            self.state_coef[j] = torch.as_tensor(fit.x / scale, dtype=torch.float32)
        for j in range(self.spec.np):
            features = pf[:, j].numpy().astype(float)
            scale = np.maximum(np.linalg.norm(features, axis=0), 1e-6)
            fit = lsq_linear(features / scale, train.powers[:, j], bounds=(0, np.inf))
            self.power_coef[j] = torch.as_tensor(fit.x / scale, dtype=torch.float32)

    def forward(self, context, control, disturbance):
        sf, pf = self.features(context, control, disturbance)
        return torch.cat([(sf * self.state_coef).sum(-1), (pf * self.power_coef).sum(-1)], dim=1)


class NeuralHistoryModel(ControlModel):
    family = 'neural_history'

    def _device(self):
        requested = str(self.parameters.get('device', 'cpu'))
        if requested.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError(f'Requested unavailable device: {requested}')
        return torch.device(requested)

    def _features_torch(self, context, controls, disturbances):
        return torch.cat([context, controls, disturbances], dim=-1)

    def _baseline_torch(self, context, controls, disturbances):
        if self.backbone is None:
            return torch.zeros((len(context), self.spec.nx + self.spec.np), device=context.device)
        return self.backbone(context, controls, disturbances)

    def _raw_torch(self, context, controls, disturbances):
        features = self._features_torch(context, controls, disturbances)
        residual = self.network((features - self.feature_mean) / self.feature_scale)
        return (self._baseline_torch(context, controls, disturbances)
                + self.target_mean + self.target_scale * residual)

    def predict_torch(self, context, controls, disturbances):
        y = self._raw_torch(context, controls, disturbances)
        return context[:, :self.spec.nx] + y[:, :self.spec.nx], torch.relu(y[:, self.spec.nx:])

    def fit(self, train, validation=None):
        torch.manual_seed(self.seed)
        self.fit_rows = train.rows.copy()
        self.device = self._device()
        self.backbone = None
        if self.family == 'physics_residual':
            self.backbone = TransportBackbone(self.spec)
            self.backbone.fit(train)
            self.backbone.to(self.device)
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        inputs = np.concatenate([train.context, train.controls, train.disturbances], axis=1)
        self.feature_mean = torch.as_tensor(inputs.mean(0), dtype=torch.float32, device=self.device)
        self.feature_scale = torch.as_tensor(np.maximum(inputs.std(0), 1e-3), dtype=torch.float32, device=self.device)
        tensors = [torch.as_tensor(a, dtype=torch.float32, device=self.device)
                   for a in (train.context, train.controls, train.disturbances)]
        targets = torch.as_tensor(np.concatenate([
            train.next_states - train.context[:, :self.spec.nx], train.powers
        ], axis=1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            residual = targets - self._baseline_torch(*tensors)
        self.target_mean = residual.mean(0)
        self.target_scale = residual.std(0, unbiased=False).clamp_min(1e-3)
        width = inputs.shape[1]
        hidden = int(self.parameters.get('hidden_size', 64))
        separate = bool(self.parameters.get('separate_power_heads', False))
        self.network = (SeparateResidualHeads(width, hidden, self.spec.nx, self.spec.np)
                        if separate else nn.Sequential(
                            nn.Linear(width, hidden), nn.GELU(),
                            nn.Linear(hidden, hidden), nn.GELU(),
                            nn.Linear(hidden, self.spec.nx+self.spec.np))).to(self.device)
        # Begin at the mean residual, avoiding huge random physical predictions.
        final_layers = [head[-1] for head in self.network.heads] if separate else [self.network[-1]]
        for layer in final_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        optimizer = torch.optim.AdamW(self.network.parameters(),
                                     lr=float(self.parameters.get('learning_rate', 1e-3)),
                                     weight_decay=float(self.parameters.get('weight_decay', 1e-4)))
        batch_size = int(self.parameters.get('batch_size', 512))
        patience = int(self.parameters.get('patience', 5))
        best_loss, stale, best = float('inf'), 0, None
        rollout_horizon = int(self.parameters.get('rollout_training_steps', 1))
        rollout_weight = float(self.parameters.get('rollout_training_weight', .1))
        if rollout_horizon < 1 or not np.isfinite(rollout_weight) or rollout_weight < 0:
            raise ValueError('Invalid rollout training settings')
        if rollout_horizon > 1:
            valid_starts = contiguous_rollout_starts(train.rows, rollout_horizon)
            if not len(valid_starts):
                raise ValueError('No contiguous train-only rollout sequences')
            valid_rollout = torch.zeros(len(train.rows), dtype=torch.bool, device=self.device)
            valid_rollout[valid_starts] = True
            next_states = torch.as_tensor(train.next_states, dtype=torch.float32, device=self.device)

        def rollout_loss(indices):
            ctx = tensors[0][indices]
            losses = []
            for lead in range(rollout_horizon):
                z, d = tensors[1][indices+lead], tensors[2][indices+lead]
                raw = self._raw_torch(ctx, z, d)
                x = ctx[:, :self.spec.nx]+raw[:, :self.spec.nx]
                state_error = (x-next_states[indices+lead])/self.target_scale[:self.spec.nx]
                power_error = (torch.relu(raw[:, self.spec.nx:])
                               -targets[indices+lead, self.spec.nx:])/self.target_scale[self.spec.nx:]
                losses.append(torch.cat([state_error, power_error], -1).square().mean())
                ctx = torch.cat([
                    x, ctx[:, :self.spec.nx], z,
                    self.spec.decay*ctx[:, -self.spec.nz:]+(1-self.spec.decay)*z], -1)
            return torch.stack(losses).mean()
        if validation is not None:
            # Fixed, outcome-blind subsample solely for early stopping.
            ids = np.linspace(0, len(validation.rows)-1, min(len(validation.rows), 10000), dtype=int)
            val = validation.take(ids)
            vt = [torch.as_tensor(a, dtype=torch.float32, device=self.device)
                  for a in (val.context, val.controls, val.disturbances)]
            vy = torch.as_tensor(np.concatenate([
                val.next_states-val.context[:, :self.spec.nx], val.powers], axis=1),
                dtype=torch.float32, device=self.device)
        for epoch in range(int(self.parameters.get('epochs', 20))):
            self.network.train()
            order = torch.randperm(len(targets), device=self.device)
            total = 0.0
            for indices in order.split(batch_size):
                prediction = self._raw_torch(*(a[indices] for a in tensors))
                loss = ((prediction - targets[indices]) / self.target_scale).square().mean()
                if rollout_horizon > 1 and rollout_weight:
                    starts = indices[valid_rollout[indices]][:128]
                    if len(starts):
                        loss = loss+rollout_weight*rollout_loss(starts)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite neural training loss')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                optimizer.step()
                total += loss.item() * len(indices)
            self.network.eval()
            with torch.no_grad():
                val_loss = (((self._raw_torch(*vt)-vy)/self.target_scale).square().mean().item()
                            if validation is not None else total / len(targets))
            self.training_log.append({'stage': 'supervised', 'epoch': epoch+1,
                                      'train_loss': total/len(targets), 'validation_loss': val_loss})
            if rollout_horizon > 1:
                self.training_log[-1].update(
                    rollout_training_steps=rollout_horizon,
                    rollout_training_weight=rollout_weight,
                    rollout_checkpoint_selection='one_step_validation_loss')
            if val_loss < best_loss - 1e-7:
                best_loss, best, stale = val_loss, deepcopy(self.network.state_dict()), 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best is None:
            raise RuntimeError('Neural training produced no checkpoint')
        self.network.load_state_dict(best)
        self.network.eval()
        return self

    def predict(self, context, control, disturbance):
        with torch.no_grad():
            result = self.predict_torch(*(torch.as_tensor(
                np.atleast_2d(a), dtype=torch.float32, device=self.device)
                for a in (context, control, disturbance)))
        return tuple(a.cpu().numpy().astype(float) for a in result)

    def to_cpu(self):
        self.device = torch.device('cpu')
        self.network.cpu()
        if self.backbone is not None:
            self.backbone.cpu()
        for name in ('feature_mean', 'feature_scale', 'target_mean', 'target_scale'):
            setattr(self, name, getattr(self, name).cpu())

    def metadata(self):
        return {**super().metadata(),
                'backbone': 'nonnegative transport + source' if self.backbone is not None else None,
                'physical_consistency_guarantee': False,
                'residual_is_unrestricted': True}


class PhysicsResidualModel(NeuralHistoryModel):
    family = 'physics_residual'
