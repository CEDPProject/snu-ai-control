from collections.abc import Mapping

import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


class EarlyStopping:
    """Stop after a meaningful validation improvement has not occurred."""

    def __init__(self, patience=8, min_delta=1e-4):
        if patience < 1:
            raise ValueError('patience must be at least 1')
        if min_delta < 0:
            raise ValueError('min_delta must be non-negative')
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = np.inf
        self.early_stop = False

    def __call__(self, value):
        value = float(value)
        improved = value < self.best_value - self.min_delta
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience
        return improved


def update_information(model_name, cfg, information_dict):
    """Add model and data metadata to the resolved experiment config."""
    cfg['MODEL']['modelname'] = model_name
    cfg['MODELSETTING']['METADATA'] = information_dict
    return cfg


def create_lr_scheduler(optimizer, scheduler_type, scheduler_config):
    """Create a validation-driven scheduler with an explicit lower LR bound."""
    normalized_type = str(scheduler_type).lower()
    if normalized_type in {'none', 'constant'}:
        return None
    if normalized_type not in {'reduce_on_plateau', 'plateau'}:
        raise ValueError(
            f'Unknown lr scheduler {scheduler_type!r}; '
            'use reduce_on_plateau or constant'
        )

    if scheduler_config is None:
        config = {}
    elif isinstance(scheduler_config, Mapping):
        config = dict(scheduler_config)
    else:
        raise TypeError('scheduler_config must be a mapping')

    factor = float(config.get('factor', 0.5))
    patience = int(config.get('patience', 2))
    min_lr = float(config.get('min_lr', 1e-6))
    if not 0.0 < factor < 1.0:
        raise ValueError('lr_scheduler.factor must be in (0, 1)')
    if patience < 0:
        raise ValueError('lr_scheduler.patience must be non-negative')
    if min_lr < 0:
        raise ValueError('lr_scheduler.min_lr must be non-negative')

    return ReduceLROnPlateau(
        optimizer,
        mode=str(config.get('mode', 'min')),
        factor=factor,
        patience=patience,
        threshold=float(config.get('threshold', 1e-3)),
        threshold_mode=str(config.get('threshold_mode', 'rel')),
        cooldown=int(config.get('cooldown', 0)),
        min_lr=min_lr,
    )
