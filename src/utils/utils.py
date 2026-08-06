import json
import logging
import os

import numpy as np
import torch


def version_build(accelerator, logdir: str, resume: int = None) -> tuple[str, str]:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)

    if resume is None:
        version = len(os.listdir(logdir))
        load_dir = os.path.join(logdir, f'version{version}')
        save_dir = load_dir
    else:
        load_dir = os.path.join(logdir, f'version{resume}')
        save_dir = os.path.join(logdir, f'version{len(os.listdir(logdir))}')

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(save_dir)
    return load_dir, save_dir


def load_resume_model(model, loaddir: str, fine_tuning_method: str = 'full'):
    """Load one explicit checkpoint and configure trainable parameters."""
    checkpoint = os.path.join(loaddir, 'best_model.pt')
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f'Resume checkpoint not found: {checkpoint}')
    weights = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(weights, strict=False)

    if fine_tuning_method == 'full':
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif fine_tuning_method == 'linear_probing':
        output_layer_tokens = ('projection', 'head', 'fc', 'regression', 'Linear_')
        selected = []
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(
                token in name for token in output_layer_tokens
            )
            if parameter.requires_grad:
                selected.append(name)
        if not selected:
            raise ValueError('No output layer matched the linear_probing policy')
    else:
        raise ValueError(
            f'Unknown fine_tuning_method {fine_tuning_method!r}; '
            'use full or linear_probing'
        )
    logging.getLogger('CHECKPOINT').info(
        'Loaded forecast checkpoint: %s (%s)', checkpoint, fine_tuning_method
    )


class Float32Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)
