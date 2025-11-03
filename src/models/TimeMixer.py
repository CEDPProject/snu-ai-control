import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block using DFT
    """
    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(down_sampling_layers))
            ])

    def forward(self, trend_list):
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, seq_len, pred_len, down_sampling_window, d_model, d_ff, 
                 dropout, channel_independence, decomp_method, moving_avg, top_k, 
                 down_sampling_layers):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.channel_independence = channel_independence

        if decomp_method == 'moving_avg':
            self.decompsition = series_decomp(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(top_k)
        else:
            raise ValueError('decompsition is error')

        if not channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.GELU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            seq_len, down_sampling_window, down_sampling_layers
        )

        # Mixing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            seq_len, down_sampling_window, down_sampling_layers
        )

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):
    """
    TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting
    Paper link: https://arxiv.org/abs/2405.14616
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.get('task_name', 'long_term_forecast')
        self.seq_len = configs['METADATA']['seq_len']
        self.pred_len = configs['METADATA']['pred_len']
        self.label_len = configs.get('label_len', 48)
        
        # Feature dimensions
        self.n_cat = configs['METADATA']['feature_types']['n_cat']
        self.n_num = configs['METADATA']['feature_types']['n_num']
        self.n_target = configs['METADATA']['feature_types']['n_target']
        
        # TimeMixer specific parameters
        self.d_model = configs.get('d_model', 128)
        self.d_ff = configs.get('d_ff', 256)
        self.e_layers = configs.get('e_layers', 2)
        self.dropout = configs.get('dropout', 0.1)
        self.embed = configs.get('embed', 'timeF')
        self.freq = configs.get('freq', 'h')
        
        # Multi-scale parameters
        self.down_sampling_window = configs.get('down_sampling_window', 2)
        self.down_sampling_layers = configs.get('down_sampling_layers', 3)
        self.down_sampling_method = configs.get('down_sampling_method', 'avg')
        
        # Decomposition parameters
        self.decomp_method = configs.get('decomp_method', 'moving_avg')
        self.moving_avg = configs.get('moving_avg', 25)
        self.top_k = configs.get('top_k', 5)
        
        # Channel independence
        self.channel_independence = configs.get('channel_independence', True)
        
        # Normalization
        self.use_norm = configs.get('use_norm', 1)
        
        # Categorical embedding
        self.cat_emb_dim = configs.get('cat_emb_dim', 8)
        if self.n_cat > 0:
            self.cat_embedding = nn.Linear(self.n_cat, self.n_cat * self.cat_emb_dim)
        
        # Total input channels
        self.channels = self.n_cat * self.cat_emb_dim + self.n_num
        
        # PDM blocks
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                self.seq_len, self.pred_len, self.down_sampling_window,
                self.d_model, self.d_ff, self.dropout, self.channel_independence,
                self.decomp_method, self.moving_avg, self.top_k, self.down_sampling_layers
            )
            for _ in range(self.e_layers)
        ])

        self.preprocess = series_decomp(self.moving_avg)

        # Embedding (timestamp 없이 처리)
        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(
                1, self.d_model, self.embed, self.freq, self.dropout
            )
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                self.channels, self.d_model, self.embed, self.freq, self.dropout
            )

        # Normalization layers
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.channels, affine=True, 
                         non_norm=True if self.use_norm == 0 else False)
                for i in range(self.down_sampling_layers + 1)
            ]
        )

        # Prediction layers
        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.seq_len // (self.down_sampling_window ** i),
                    self.pred_len,
                )
                for i in range(self.down_sampling_layers + 1)
            ]
        )

        if self.channel_independence:
            self.projection_layer = nn.Linear(self.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(self.d_model, self.channels, bias=True)

            self.out_res_layers = torch.nn.ModuleList([
                torch.nn.Linear(
                    self.seq_len // (self.down_sampling_window ** i),
                    self.seq_len // (self.down_sampling_window ** i),
                )
                for i in range(self.down_sampling_layers + 1)
            ])

            self.regression_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window ** i),
                        self.pred_len,
                    )
                    for i in range(self.down_sampling_layers + 1)
                ]
            )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc):
        """
        Multi-scale down-sampling without timestamp features
        x_enc: [Batch, seq_len, channels]
        Returns: list of [Batch, seq_len_i, channels] for each scale
        """
        # 이미 처리된 경우 (리스트인 경우) 바로 반환
        if isinstance(x_enc, list):
            return x_enc
        
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(
                in_channels=self.channels, out_channels=self.channels,
                kernel_size=3, padding=padding,
                stride=self.down_sampling_window,
                padding_mode='circular',
                bias=False
            )
        else:
            # No down-sampling, return as list
            return [x_enc]
        
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_enc_sampling_list = []
        
        # 첫 번째 스케일 추가
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        
        # Down-sampling loop
        for i in range(self.down_sampling_layers):
            # Down-sample x_enc
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        return x_enc_sampling_list

    def forecast(self, x_enc):
        """
        x_enc: [Batch, seq_len, channels]
        Returns: [Batch, pred_len, channels]
        """
        # Multi-scale processing (timestamp 없이)
        x_enc = self.__multi_scale_process_inputs(x_enc)

        x_list = []
        
        # Normalization and reshaping for each scale
        for i, x in enumerate(x_enc):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # Embedding (timestamp 없이)
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        
        for i, x in enumerate(x_list[0]):
            enc_out = self.enc_embedding(x, None)  # timestamp=None
            enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.e_layers):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.channels, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)
        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, input_cat, input_num, input_ts=None):
        """
        input_cat: (batch, seq_len, n_cat) or None
        input_num: (batch, seq_len, n_num) or None
        input_ts: timestamps (NOT USED - ignored)
        
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
        x_enc = torch.cat(features, dim=-1)  # [B, seq_len, channels]
        
        # Forecast (input_ts는 사용하지 않음)
        dec_out = self.forecast(x_enc)
        
        # Select target features only
        return dec_out[:, :, -self.n_target:]