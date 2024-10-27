from layers.Data_Embedding import DataEmbedding_wo_temp
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.Self_Attention import FullAttention, AttentionLayer
from layers.FTtransformer_layers import *

import torch



class FTTransformer(torch.nn.Module):
    def __init__(self, configs):
        super(FTTransformer, self).__init__()
        self.configs = configs
        tab_config = configs.tab_model_args
        tsa_config = configs.tsa_model_args
        tsb_config = configs.tsb_model_args
        #! Tabular data
        self.tftransformer = FTTransformer_module(tab_config.d_numerical,
                                categories = tab_config.categories,
                                token_bias = tab_config.token_bias,
                                n_layers = tab_config.n_layers,
                                d_token = tab_config.d_token,
                                n_heads = tab_config.n_heads,
                                d_ffn_factor = tab_config.d_ffn_factor,
                                attention_dropout = tab_config.attention_dropout,
                                ffn_dropout = tab_config.ffn_dropout,
                                residual_dropout = tab_config.residual_dropout,
                                activation = tab_config.activation,
                                prenormalization = tab_config.prenormalization,
                                initialization = tab_config.initialization,
                                kv_compression = tab_config.kv_compression,
                                kv_compression_sharing = tab_config.kv_compression_sharing,
                                d_out = tab_config.d_out)
        
        self.embed_output_dim = tab_config.d_out + 2 * tsa_config.d_model + 6 * tsb_config.d_model
        self.output_dim = len(tab_config.categories) + tab_config.d_numerical +\
                                2 * tsa_config.enc_in +\
                                6 * tsb_config.enc_in
        
        #! Time-series A
        self.tsa_enc_embedding = DataEmbedding_wo_temp(tsa_config.enc_in, tsa_config.d_model, tsa_config.embed, tsa_config.freq,
                                                tsa_config.dropout)
        
        self.tsa_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, tsa_config.factor, attention_dropout=tsa_config.dropout,
                                      output_attention=tsa_config.output_attention), tsa_config.d_model, tsa_config.n_heads),
                    tsa_config.d_model,
                    tsa_config.d_ff,
                    dropout=tsa_config.dropout,
                    activation=tsa_config.activation
                ) for l in range(tsa_config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(tsa_config.d_model)
        )
        
        #! Time-series B
        self.tsb_enc_embedding = DataEmbedding_wo_temp(tsb_config.enc_in, tsb_config.d_model, tsb_config.embed, tsb_config.freq,
                                                tsb_config.dropout)

        self.tsb_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, tsb_config.factor, attention_dropout=tsb_config.dropout,
                                        output_attention=tsb_config.output_attention), tsb_config.d_model, tsb_config.n_heads),
                    tsb_config.d_model,
                    tsb_config.d_ff,
                    dropout=tsb_config.dropout,
                    activation=tsb_config.activation
                ) for l in range(tsb_config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(tsb_config.d_model)
        )
        
        #! Output MLP
        if configs.pre_training == False:
            self.final_layer = MultiLayerPerceptron(self.embed_output_dim, 
                                            layer_dims=tab_config.mlp_dims, 
                                            dropout=tab_config.mlp_dropout,
                                            output_dims=276)
        elif configs.pre_training == True:
            self.final_layer = MultiLayerPerceptron(self.embed_output_dim, 
                                            layer_dims=tab_config.mlp_dims, 
                                            dropout=tab_config.mlp_dropout,
                                            output_dims=self.output_dim)
            
        self.mlp = torch.nn.Sequential(self.final_layer)
        
        
    def forward(self, x_cat, x_cons, x_tsa, x_tsb, enc_self_mask=None):
        #! Tabular data
        tabular_out = self.tftransformer(x_cat = x_cat, x_num = x_cons)

        #! Time-series A
        tsa_enc_out = self.tsa_enc_embedding(x_tsa)
        tsa_enc_out, attns = self.tsa_encoder(tsa_enc_out, attn_mask=enc_self_mask)        
        tsa_enc_out = tsa_enc_out.view(x_tsa.size(0), -1)

        #! Time-series B
        tsb_enc_out = self.tsb_enc_embedding(x_tsb)
        tsb_enc_out, attns = self.tsb_encoder(tsb_enc_out, attn_mask=enc_self_mask)
        tsb_enc_out = tsb_enc_out.view(x_tsb.size(0), -1)

        all_output = torch.cat([tabular_out, tsa_enc_out, tsb_enc_out], dim=-1)
        
        out = self.mlp(all_output)

        return torch.sigmoid(out.squeeze(1))