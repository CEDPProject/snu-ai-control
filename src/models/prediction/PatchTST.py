from models.prediction.common import select_target_channels, target_channel_indices

import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


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
    PatchTST with categorical embedding support
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        
        self.seq_len = configs['METADATA']['seq_len']
        self.pred_len = configs['METADATA']['pred_len']
        padding = stride
        
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

        # Patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs['d_model'], patch_len, stride, padding, configs.get('dropout', 0.1))

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.get('factor', 1), 
                                    attention_dropout=configs.get('dropout', 0.1),
                                    output_attention=False), 
                        configs['d_model'], configs['n_heads']),
                    configs['d_model'],
                    configs['d_ff'],
                    dropout=configs.get('dropout', 0.1),
                    activation=configs.get('activation', 'gelu')
                ) for l in range(configs['e_layers'])
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs['d_model']), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = configs['d_model'] * int((self.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(self.channels, self.head_nf, self.pred_len,
                                head_dropout=configs.get('dropout', 0.1))

    def forecast(self, x):
        """
        x: (batch, seq_len, channels)
        """
        # Normalization
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # Patching and embedding
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        enc_out, n_vars = self.patch_embedding(x)

        # Encoder
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        # Select target features
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