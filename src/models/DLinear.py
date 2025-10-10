import torch
import torch.nn as nn

class Model(nn.Module):
    """
    DLinear with categorical embedding support
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.seq_len = configs['METADATA']['seq_len']
        self.pred_len = configs['METADATA']['pred_len']
        
        self.decompsition = series_decomp(configs['moving_avg'])
        self.individual = configs['individual']
        
        # Feature dimensions
        self.n_cat = configs['METADATA']['feature_types']['n_cat']
        self.n_num = configs['METADATA']['feature_types']['n_num']
        self.n_target = configs['METADATA']['feature_types']['n_target']
        
        # Categorical embedding
        self.cat_emb_dim = configs.get('cat_emb_dim', 8)  # embedding dimension per categorical feature
        
        if self.n_cat > 0:
            # Each categorical feature gets projected to embedding dimension
            self.cat_embedding = nn.Linear(self.n_cat, self.n_cat * self.cat_emb_dim)
        
        # Total input channels after embedding
        self.channels = self.n_cat * self.cat_emb_dim + self.n_num

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.n_target):  # 출력 채널 수
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            # Input projection (all features -> target features)
            self.input_projection = nn.Linear(self.channels, self.n_target)
            
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def encode(self, x):
        """
        x: (batch, seq_len, channels)
        """
        seasonal_init, trend_init = self.decompsition(x)  # (batch, seq_len, channels)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)  # (batch, channels, seq_len)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), self.n_target, self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), self.n_target, self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.n_target):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # Project to target dimension first
            seasonal_init = self.input_projection(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, n_target, seq_len)
            trend_init = self.input_projection(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
            
            seasonal_output = self.Linear_Seasonal(seasonal_init)  # (batch, n_target, pred_len)
            trend_output = self.Linear_Trend(trend_init)
            
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # (batch, pred_len, n_target)

    def forward(self, input_cat, input_num, input_ts=None):
        """
        input_cat: (batch, seq_len, n_cat) or None
        input_num: (batch, seq_len, n_num) or None
        input_ts: timestamps (not used in this model)
        
        Returns:
            (batch, pred_len, n_target)
        """
        features = []
        
        # 1. Categorical features -> embedding
        if input_cat is not None and self.n_cat > 0:
            # (batch, seq_len, n_cat) -> (batch, seq_len, n_cat * emb_dim)
            cat_emb = self.cat_embedding(input_cat)
            features.append(cat_emb)
        
        # 2. Numerical features (그대로 사용)
        if input_num is not None and self.n_num > 0:
            features.append(input_num)
            
        # 3. Concatenate
        if len(features) == 0:
            raise ValueError("No input features provided")
        
        x = torch.cat(features, dim=-1)  # (batch, seq_len, channels)
        
        # 4. DLinear encoding
        dec_out = self.encode(x)
        
        return dec_out  # (batch, pred_len, n_target)

    
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean