from models.prediction.common import select_target_channels, target_channel_indices

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs['METADATA']['seq_len']
        self.pred_len = configs['METADATA']['pred_len']
        self.k = configs.get('top_k', 5)
        
        self.conv = nn.Sequential(
            Inception_Block_V1(configs['d_model'], configs['d_ff'],
                               num_kernels=configs.get('num_kernels', 6)),
            nn.GELU(),
            Inception_Block_V1(configs['d_ff'], configs['d_model'],
                               num_kernels=configs.get('num_kernels', 6))
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res


class Model(nn.Module):
    """
    TimesNet with categorical embedding support
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.get('task_name', 'long_term_forecast')
        self.seq_len = configs['METADATA']['seq_len']
        self.pred_len = configs['METADATA']['pred_len']
        
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
        
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs['e_layers'])])
        self.enc_embedding = DataEmbedding(
            self.channels, configs['d_model'], configs.get('embed', 'fixed'),
            configs.get('freq', 'h'), configs.get('dropout', 0.1)
        )
        self.layer = configs['e_layers']
        self.layer_norm = nn.LayerNorm(configs['d_model'])
        
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs['d_model'], self.n_target, bias=True)

    def forecast(self, x):
        """
        x: (batch, seq_len, channels)
        """
        # Normalization
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # Embedding
        enc_out = self.enc_embedding(x, None)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        dec_out = self.projection(enc_out)

        # De-Normalization
        target_stdev = select_target_channels(stdev, self.target_channel_indices)
        target_means = select_target_channels(means, self.target_channel_indices)
        dec_out = dec_out * target_stdev.repeat(1, self.pred_len + self.seq_len, 1)
        dec_out = dec_out + target_means.repeat(1, self.pred_len + self.seq_len, 1)
        
        return dec_out[:, -self.pred_len:, :]

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