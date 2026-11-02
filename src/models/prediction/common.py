"""Shared feature-index handling for forecasting models."""

import torch


def target_channel_indices(configs, categorical_embedding_dim: int) -> tuple[int, ...]:
    """Map target names to channels after categorical embeddings are concatenated."""
    metadata = configs['METADATA']
    numerical_names = list(metadata.get('input_numerical_features', []))
    target_names = list(metadata.get('target_features', []))
    if not numerical_names or not target_names:
        raise ValueError(
            'Forecast metadata must include input_numerical_features and target_features'
        )

    missing = [name for name in target_names if name not in numerical_names]
    if missing:
        raise ValueError(
            'Every forecast target must also be a numerical input for '
            f'channel-preserving models. Missing targets: {missing}'
        )

    n_cat = int(metadata['feature_types']['n_cat'])
    numerical_offset = n_cat * int(categorical_embedding_dim)
    return tuple(numerical_offset + numerical_names.index(name) for name in target_names)


def select_target_channels(
    tensor: torch.Tensor,
    indices: tuple[int, ...],
) -> torch.Tensor:
    index = torch.as_tensor(indices, device=tensor.device, dtype=torch.long)
    return tensor.index_select(-1, index)
