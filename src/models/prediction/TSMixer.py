from models.prediction.common import select_target_channels, target_channel_indices

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.TSMixer_layers import MixerLayer, TimeBatchNorm2d, feature_to_time, time_to_feature


class Model(nn.Module):
    """
    TSMixer: Multivariate Time Series Forecasting with Mixing
    Paper link: https://arxiv.org/abs/2303.06053
    """

    def __init__(self, configs):
        super().__init__()
        
        self.task_name = configs.get('task_name', 'long_term_forecast')
        self.seq_len = configs['METADATA']['seq_len']
        self.pred_len = configs['METADATA']['pred_len']
        
        # Feature dimensions
        self.n_cat = configs['METADATA']['feature_types']['n_cat']
        self.n_num = configs['METADATA']['feature_types']['n_num']
        self.n_target = configs['METADATA']['feature_types']['n_target']
        
        # TSMixer specific parameters
        self.output_channels = configs.get('output_channels', None)
        self.activation_fn_name = configs.get('activation_fn', 'relu')
        self.num_blocks = configs.get('num_blocks', 2)
        self.dropout_rate = configs.get('dropout', 0.1)
        self.ff_dim = configs.get('ff_dim', 64)
        self.normalize_before = configs.get('normalize_before', True)
        self.norm_type_name = configs.get('norm_type', 'batch')
        
        # Categorical embedding
        self.cat_emb_dim = configs.get('cat_emb_dim', 8)
        self.target_channel_indices = target_channel_indices(configs, self.cat_emb_dim)
        if self.n_cat > 0:
            self.cat_embedding = nn.Linear(self.n_cat, self.n_cat * self.cat_emb_dim)
        
        # Total input channels
        self.channels = self.n_cat * self.cat_emb_dim + self.n_num
        
        # Set output channels
        if self.output_channels is None:
            self.output_channels = self.channels
        
        # Transform activation_fn to callable
        self.activation_fn = getattr(F, self.activation_fn_name)
        
        # Transform norm_type to callable
        assert self.norm_type_name in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {self.norm_type_name}, must be one of batch, layer."
        self.norm_type = TimeBatchNorm2d if self.norm_type_name == "batch" else nn.LayerNorm
        
        # Build mixer layers
        self.mixer_layers = self._build_mixer(
            num_blocks=self.num_blocks,
            input_channels=self.channels,
            output_channels=self.output_channels,
            ff_dim=self.ff_dim,
            activation_fn=self.activation_fn,
            dropout_rate=self.dropout_rate,
            sequence_length=self.seq_len,
            normalize_before=self.normalize_before,
            norm_type=self.norm_type,
        )
        
        # Temporal projection layer
        self.temporal_projection = nn.Linear(self.seq_len, self.pred_len)

    def _build_mixer(
        self, num_blocks: int, input_channels: int, output_channels: int, **kwargs
    ):
        """Build the mixer blocks for the model.

        Args:
            num_blocks (int): Number of mixer blocks to be built.
            input_channels (int): Number of input channels for the first block.
            output_channels (int): Number of output channels for the last block.
            **kwargs: Additional keyword arguments for mixer layer configuration.

        Returns:
            nn.Sequential: Sequential container of mixer layers.
        """
        channels = [input_channels] * (num_blocks - 1) + [output_channels]

        return nn.Sequential(
            *[
                MixerLayer(input_channels=in_ch, output_channels=out_ch, **kwargs)
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forecast(self, x):
        """
        x: (batch, seq_len, channels)
        Returns: (batch, pred_len, channels)
        """
        # Mixer layers
        x = self.mixer_layers(x)
        
        # Temporal projection
        x_temp = feature_to_time(x)  # (batch, channels, seq_len)
        x_temp = self.temporal_projection(x_temp)  # (batch, channels, pred_len)
        x = time_to_feature(x_temp)  # (batch, pred_len, channels)
        
        return x

    def forward(self, input_cat, input_num, input_ts=None):
        """
        input_cat: (batch, seq_len, n_cat) or None
        input_num: (batch, seq_len, n_num) or None
        input_ts: timestamps (not used in TSMixer)
        
        Returns:
            (batch, pred_len, n_target)
        """
        features = []
        
        # Categorical embedding
        if input_cat is not None and self.n_cat > 0:
            cat_emb = self.cat_embedding(input_cat)
            features.append(cat_emb)
        
        # Numerical features
        if input_num is not None and self.n_num > 0:
            features.append(input_num)
        
        if len(features) == 0:
            raise ValueError("No input features provided")
        
        # Concatenate all features
        x = torch.cat(features, dim=-1)  # [B, seq_len, channels]
        
        # Forecast
        dec_out = self.forecast(x)
        
        # Select target features only
        return select_target_channels(dec_out, self.target_channel_indices)