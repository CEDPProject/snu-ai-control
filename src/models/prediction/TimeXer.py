from models.prediction.common import select_target_channels, target_channel_indices

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    TimeXer with categorical embedding support (simplified)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.get('task_name', 'long_term_forecast')
        self.seq_len = configs['METADATA']['seq_len']
        self.pred_len = configs['METADATA']['pred_len']
        self.use_norm = configs.get('use_norm', True)
        
        # Feature dimensions
        self.n_cat = configs['METADATA']['feature_types']['n_cat']
        self.n_num = configs['METADATA']['feature_types']['n_num']
        self.n_target = configs['METADATA']['feature_types']['n_target']
        
        # Categorical embedding
        self.cat_emb_dim = configs.get('cat_emb_dim', 8)
        self.target_channel_indices = target_channel_indices(configs, self.cat_emb_dim)
        
        if self.n_cat > 0:
            self.cat_embedding = nn.Linear(self.n_cat, self.n_cat * self.cat_emb_dim)
        
        # Total input channels
        self.channels = self.n_cat * self.cat_emb_dim + self.n_num
        
        # Simplified: use inverted embedding
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len, configs['d_model'], configs.get('embed', 'fixed'),
            configs.get('freq', 'h'), configs.get('dropout', 0.1)
        )
        
        # Projection head
        self.head_nf = configs['d_model']
        self.head = nn.Linear(self.head_nf, self.pred_len)

    def forecast(self, x):
        """
        x: (batch, seq_len, channels)
        """
        if self.use_norm:
            # Normalization
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        # Embedding
        enc_out = self.enc_embedding(x, None)  # (batch, channels, d_model)
        
        # Prediction
        dec_out = self.head(enc_out).permute(0, 2, 1)  # (batch, pred_len, channels)

        if self.use_norm:
            # De-Normalization - 전체 channels에 적용
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        # Select target features only
        return select_target_channels(dec_out, self.target_channel_indices)

    def forward(self, input_cat, input_num, input_ts=None):
        """
        input_cat: (batch, seq_len, n_cat) or None
        input_num: (batch, seq_len, n_num) or None
        input_ts: timestamps (not used)
        
        Returns:
            (batch, pred_len, n_target)
        """
        features = []
        
        if input_cat is not None and self.n_cat > 0:
            cat_emb = self.cat_embedding(input_cat)
            features.append(cat_emb)
        
        if input_num is not None and self.n_num > 0:
            features.append(input_num)
        
        if len(features) == 0:
            raise ValueError("No input features provided")
        
        x = torch.cat(features, dim=-1)
        
        dec_out = self.forecast(x)
        return dec_out