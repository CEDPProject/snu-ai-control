import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi
import torch.nn.functional as F


class MIC(nn.Module):
    """MIC layer to extract local and global features"""

    def __init__(self, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24],
                 isometric_kernel=[18, 6], device='cuda'):
        super(MIC, self).__init__()
        self.conv_kernel = conv_kernel
        self.device = device

        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                                       kernel_size=i, padding=0, stride=1)
                                             for i in isometric_kernel])

        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])

        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in conv_kernel])

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = torch.nn.Conv2d(in_channels=feature_size, out_channels=feature_size,
                                     kernel_size=(len(self.conv_kernel), 1))

        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)

        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1), device=self.device)
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)

        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]

        x = self.norm(x.permute(0, 2, 1) + input)
        return x

    def forward(self, src):
        self.device = src.device
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, trend1 = self.decomp[i](src)
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)

        mg = torch.tensor([], device=self.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1).to(self.device)), dim=1)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)

        y = self.norm1(mg)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)

        return self.norm2(mg + y)


class SeasonalPrediction(nn.Module):
    def __init__(self, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1,
                 conv_kernel=[2, 4], isometric_kernel=[18, 6], device='cuda'):
        super(SeasonalPrediction, self).__init__()

        self.mic = nn.ModuleList([MIC(feature_size=embedding_size, n_heads=n_heads,
                                      decomp_kernel=decomp_kernel, conv_kernel=conv_kernel,
                                      isometric_kernel=isometric_kernel, device=device)
                                  for i in range(d_layers)])

        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)


class Model(nn.Module):
    """
    MICN with categorical embedding support
    Paper link: https://openreview.net/pdf?id=zt53IDUR1U
    """
    def __init__(self, configs, conv_kernel=[12, 16]):
        super(Model, self).__init__()

        decomp_kernel = []
        isometric_kernel = []
        for ii in conv_kernel:
            if ii % 2 == 0:
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((configs['METADATA']['seq_len'] + configs['METADATA']['pred_len'] + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((configs['METADATA']['seq_len'] + configs['METADATA']['pred_len'] + ii - 1) // ii)

        self.task_name = configs.get('task_name', 'long_term_forecast')
        self.pred_len = configs['METADATA']['pred_len']
        self.seq_len = configs['METADATA']['seq_len']
        
        # Feature dimensions
        self.n_cat = configs['METADATA']['feature_types']['n_cat']
        self.n_num = configs['METADATA']['feature_types']['n_num']
        self.n_target = configs['METADATA']['feature_types']['n_target']
        
        # Categorical embedding
        self.cat_emb_dim = configs.get('cat_emb_dim', 8)
        
        if self.n_cat > 0:
            self.cat_embedding = nn.Linear(self.n_cat, self.n_cat * self.cat_emb_dim)
        
        # Total input channels
        self.channels = self.n_cat * self.cat_emb_dim + self.n_num

        self.decomp_multi = series_decomp_multi(decomp_kernel)

        self.dec_embedding = DataEmbedding(
            self.channels, configs['d_model'], configs.get('embed', 'fixed'),
            configs.get('freq', 'h'), configs.get('dropout', 0.1)
        )

        self.conv_trans = SeasonalPrediction(
            embedding_size=configs['d_model'], n_heads=configs['n_heads'],
            dropout=configs.get('dropout', 0.1),
            d_layers=configs['d_layers'], decomp_kernel=decomp_kernel,
            c_out=self.n_target, conv_kernel=conv_kernel,
            isometric_kernel=isometric_kernel, device=torch.device('cuda:0')
        )
        
        self.regression = nn.Linear(self.seq_len, self.pred_len)
        self.regression_dim = nn.Linear(self.channels, self.n_target)
        self.regression.weight = nn.Parameter(
            (1 / self.pred_len) * torch.ones([self.pred_len, self.seq_len]),
            requires_grad=True)

    def forecast(self, x):
        """
        x: (batch, seq_len, channels)
        """
        # Multi-scale Hybrid Decomposition
        seasonal_init_enc, trend = self.decomp_multi(x)
        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)
        trend = self.regression_dim(trend)

        # Embedding
        zeros = torch.zeros([x.shape[0], self.pred_len, x.shape[2]], device=x.device)
        seasonal_init_dec = torch.cat([seasonal_init_enc[:, -self.seq_len:, :], zeros], dim=1)
        dec_out = self.dec_embedding(seasonal_init_dec, None)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
        return dec_out

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