from models.common import select_target_channels, target_channel_indices

import math
import torch
import torch.nn as nn


class FourierFilter(nn.Module):
    """
    Fourier Filter: to time-variant and time-invariant term
    """
    def __init__(self, mask_spectrum):
        super(FourierFilter, self).__init__()
        self.mask_spectrum = mask_spectrum
        
    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf*mask, dim=1)
        x_inv = x - x_var
        
        return x_var, x_inv
    

class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=128, 
                 hidden_layers=2, 
                 dropout=0.05,
                 activation='tanh'): 
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError
        
        layers = [nn.Linear(self.f_in, self.hidden_dim), 
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]
        
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y
    

class KPLayer(nn.Module):
    """
    A demonstration of finding one step transition of linear system by DMD iteratively
    """
    def __init__(self): 
        super(KPLayer, self).__init__()
        
        self.K = None # B E E

    def one_step_forward(self, z, return_rec=False, return_K=False):
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution # B E E
        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_pred = torch.bmm(z[:, -1:], self.K)
        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred

        return z_pred
    
    def forward(self, z, pred_len=1):
        assert pred_len >= 1, 'prediction length should not be less than 1'
        z_rec, z_pred= self.one_step_forward(z, return_rec=True)
        z_preds = [z_pred]
        for i in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            z_preds.append(z_pred)
        z_preds = torch.cat(z_preds, dim=1)
        return z_rec, z_preds


class KPLayerApprox(nn.Module):
    """
    Find koopman transition of linear system by DMD with multistep K approximation
    """
    def __init__(self): 
        super(KPLayerApprox, self).__init__()
        
        self.K = None # B E E
        self.K_step = None # B E E

    def forward(self, z, pred_len=1):
        # z:       B L E, koopman invariance space representation
        # z_rec:   B L E, reconstructed representation
        # z_pred:  B S E, forecasting representation
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution # B E E

        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1) # B L E
        
        if pred_len <= input_len:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            temp_z_pred, all_pred = z, []
            for _ in range(math.ceil(pred_len / input_len)):
                temp_z_pred = torch.bmm(temp_z_pred, self.K_step)
                all_pred.append(temp_z_pred)
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]

        return z_rec, z_pred
    

class TimeVarKP(nn.Module):
    """
    Koopman Predictor with DMD (analysitical solution of Koopman operator)
    Utilize local variations within individual sliding window to predict the future of time-variant term
    """
    def __init__(self,
                 enc_in=8,
                 input_len=96,
                 pred_len=96,
                 seg_len=24,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None,
                 multistep=False,
                ):
        super(TimeVarKP, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep
        self.encoder, self.decoder = encoder, decoder            
        self.freq = math.ceil(self.input_len / self.seg_len)  # segment number of input
        self.step = math.ceil(self.pred_len / self.seg_len)   # segment number of output
        self.padding_len = self.seg_len * self.freq - self.input_len
        # Approximate mulitstep K by KPLayerApprox when pred_len is large
        self.dynamics = KPLayerApprox() if self.multistep else KPLayer() 

    def forward(self, x):
        # x: B L C
        B, L, C = x.shape

        res = torch.cat((x[:, L-self.padding_len:, :], x) ,dim=1)

        res = res.chunk(self.freq, dim=1)     # F x B P C, P means seg_len
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)   # B F PC

        res = self.encoder(res) # B F H
        x_rec, x_pred = self.dynamics(res, self.step) # B F H, B S H

        x_rec = self.decoder(x_rec) # B F PC
        x_rec = x_rec.reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # B L C
        
        x_pred = self.decoder(x_pred)     # B S PC
        x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :] # B S C

        return x_rec, x_pred


class TimeInvKP(nn.Module):
    """
    Koopman Predictor with learnable Koopman operator
    Utilize lookback and forecast window snapshots to predict the future of time-invariant term
    """
    def __init__(self,
                 input_len=96,
                 pred_len=96,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None):
        super(TimeInvKP, self).__init__()
        self.dynamic_dim = dynamic_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.encoder = encoder
        self.decoder = decoder

        K_init = torch.randn(self.dynamic_dim, self.dynamic_dim)
        U, _, V = torch.svd(K_init) # stable initialization
        self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())
    
    def forward(self, x):
        # x: B L C
        res = x.transpose(1, 2) # B C L
        res = self.encoder(res) # B C H
        res = self.K(res) # B C H
        res = self.decoder(res) # B C S
        res = res.transpose(1, 2) # B S C

        return res


class Model(nn.Module):
    '''
    Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors
    Paper link: https://arxiv.org/pdf/2305.18803.pdf
    '''
    def __init__(self, configs):
        super(Model, self).__init__()
        
        
        self.seq_len = configs['METADATA']['seq_len']
        self.pred_len = configs['METADATA']['pred_len']
        
        # Feature dimensions
        self.n_cat = configs['METADATA']['feature_types']['n_cat']
        self.n_num = configs['METADATA']['feature_types']['n_num']
        self.n_target = configs['METADATA']['feature_types']['n_target']
        
        # Koopa specific parameters
        self.seg_len = self.seq_len // 4
        self.dynamic_dim = configs.get('dynamic_dim', 128)
        self.hidden_dim = configs.get('hidden_dim', 64)
        self.hidden_layers = configs.get('hidden_layers', 2)
        self.num_blocks = configs.get('num_blocks', 3)
        self.multistep = configs.get('multistep', False)
        self.alpha = configs.get('alpha', 0.2)
        
        # Categorical embedding
        self.cat_emb_dim = configs.get('cat_emb_dim', 8)
        self.target_channel_indices = target_channel_indices(configs, self.cat_emb_dim)
        if self.n_cat > 0:
            self.cat_embedding = nn.Linear(self.n_cat, self.n_cat * self.cat_emb_dim)
        
        # Total input channels
        self.channels = self.n_cat * self.cat_emb_dim + self.n_num
        
        # Fourier filter mask
        self.mask_spectrum = self._get_mask_spectrum()
        self.disentanglement = FourierFilter(self.mask_spectrum)

        # Time-invariant encoder/decoder
        self.time_inv_encoder = MLP(
            f_in=self.seq_len, 
            f_out=self.dynamic_dim, 
            activation='relu',
            hidden_dim=self.hidden_dim, 
            hidden_layers=self.hidden_layers
        )
        self.time_inv_decoder = MLP(
            f_in=self.dynamic_dim, 
            f_out=self.pred_len, 
            activation='relu',
            hidden_dim=self.hidden_dim, 
            hidden_layers=self.hidden_layers
        )
        self.time_inv_kps = nn.ModuleList([
            TimeInvKP(
                input_len=self.seq_len,
                pred_len=self.pred_len, 
                dynamic_dim=self.dynamic_dim,
                encoder=self.time_inv_encoder, 
                decoder=self.time_inv_decoder
            )
            for _ in range(self.num_blocks)
        ])
        
        # Time-variant encoder/decoder
        self.time_var_encoder = MLP(
            f_in=self.seg_len * self.channels, 
            f_out=self.dynamic_dim, 
            activation='tanh',
            hidden_dim=self.hidden_dim, 
            hidden_layers=self.hidden_layers
        )
        self.time_var_decoder = MLP(
            f_in=self.dynamic_dim, 
            f_out=self.seg_len * self.channels, 
            activation='tanh',
            hidden_dim=self.hidden_dim, 
            hidden_layers=self.hidden_layers
        )
        self.time_var_kps = nn.ModuleList([
            TimeVarKP(
                enc_in=self.channels,
                input_len=self.seq_len,
                pred_len=self.pred_len,
                seg_len=self.seg_len,
                dynamic_dim=self.dynamic_dim,
                encoder=self.time_var_encoder,
                decoder=self.time_var_decoder,
                multistep=self.multistep
            )
            for _ in range(self.num_blocks)
        ])

    def _get_mask_spectrum(self):
        """
        Uniformly spaced frequency components
        """
        spectrum_len = self.seq_len // 2 + 1
        num_masks = int(spectrum_len * self.alpha)
        
        # 균등하게 분산된 주파수 선택
        mask_spectrum = torch.linspace(0, spectrum_len-1, num_masks).long()
        return mask_spectrum
    
    def forecast(self, x_enc):
        """
        x_enc: [Batch, seq_len, channels]
        Returns: [Batch, pred_len, channels]
        """
        # Series Stationarization adopted from NSformer
        mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # Koopman Forecasting
        residual, forecast = x_enc, None
        for i in range(self.num_blocks):
            time_var_input, time_inv_input = self.disentanglement(residual)
            time_inv_output = self.time_inv_kps[i](time_inv_input)
            time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)
            residual = residual - time_var_backcast
            if forecast is None:
                forecast = (time_inv_output + time_var_output)
            else:
                forecast += (time_inv_output + time_var_output)

        # De-normalization
        res = forecast * std_enc + mean_enc

        return res        
    
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
        return select_target_channels(dec_out, self.target_channel_indices)