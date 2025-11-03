import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    FreTS: Frequency-domain Transformer for Time Series
    Paper link: https://arxiv.org/pdf/2311.06184.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.seq_len = configs['METADATA']['seq_len']
        self.pred_len = configs['METADATA']['pred_len']
        
        # Feature dimensions
        self.n_cat = configs['METADATA']['feature_types']['n_cat']
        self.n_num = configs['METADATA']['feature_types']['n_num']
        self.n_target = configs['METADATA']['feature_types']['n_target']
        
        # FreTS specific parameters
        self.embed_size = configs.get('embed_size', 128)
        self.hidden_size = configs.get('hidden_size', 256)
        self.channel_independence = configs.get('channel_independence', '1')
        self.sparsity_threshold = configs.get('sparsity_threshold', 0.01)
        self.scale = configs.get('scale', 0.02)
        
        # Categorical embedding
        self.cat_emb_dim = configs.get('cat_emb_dim', 8)
        if self.n_cat > 0:
            self.cat_embedding = nn.Linear(self.n_cat, self.n_cat * self.cat_emb_dim)
        
        # Total input channels
        self.channels = self.n_cat * self.cat_emb_dim + self.n_num
        
        # Learnable parameters for frequency-domain MLP
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

    def tokenEmb(self, x):
        """Dimension extension
        x: [Batch, Input length, Channel]
        Returns: [Batch, Channel, Input length, embed_size]
        """
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = x.unsqueeze(3)  # [B, C, L, 1]
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings  # [1, D]
        return x * y  # [B, C, L, D]

    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        """Frequency-domain MLPs
        dimension: FFT along the dimension
        r: the real part of weights, i: the imaginary part of weights
        rb: the real part of bias, ib: the imaginary part of bias
        """
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def MLP_temporal(self, x, B, N, L):
        """Frequency temporal learner
        x: [B, N, T, D]
        """
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")
        return x

    def MLP_channel(self, x, B, N, L):
        """Frequency channel learner
        x: [B, N, T, D]
        """
        x = x.permute(0, 2, 1, 3)  # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.channels, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)  # [B, N, T, D]
        return x

    def forecast(self, x):
        """
        x: [Batch, seq_len, channels]
        Returns: [Batch, pred_len, channels]
        """
        B, T, N = x.shape
        
        # Embedding x: [B, N, T, D]
        x = self.tokenEmb(x)
        bias = x
        
        # Channel mixing (optional)
        if self.channel_independence == '0':
            x = self.MLP_channel(x, B, N, T)
        
        # Temporal mixing
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        
        # Prediction
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)  # [B, pred_len, N]
        
        return x

    def forward(self, input_cat, input_num, input_ts=None):
        """
        input_cat: (batch, seq_len, n_cat) or None
        input_num: (batch, seq_len, n_num) or None
        input_ts: timestamps (not used)
        
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
        return dec_out[:, :, -self.n_target:]