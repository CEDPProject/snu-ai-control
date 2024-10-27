import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

from layers.TabTransformer_layers import *
from layers.Gated_Tab_Transformer_layers import *
from layers.Data_Embedding import DataEmbedding_wo_temp
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.Self_Attention import FullAttention, AttentionLayer


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class HeadAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, HeadAttention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim, dropout=ff_dropout))),
            ]))

    def forward(self, x):
        x = self.embeds(x)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x


class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class GatedTabTransformer_module(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            transformer_dim,
            transformer_depth,
            transformer_heads,
            transformer_dim_head=16,
            dim_out=1,
            mlp_depth=2,
            mlp_act=None,
            num_special_tokens=2,
            continuous_mean_std=None,
            attn_dropout=0.,
            ff_dropout=0.,
            gmlp_enabled=False,
            mlp_dimension=32,
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        # continuous

        if exists(continuous_mean_std):
            assert continuous_mean_std.shape == (num_continuous,
                                                 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
        self.register_buffer('continuous_mean_std', continuous_mean_std)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous

        # transformer

        self.transformer = Transformer(
            num_tokens=total_tokens,
            dim=transformer_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=transformer_dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        # mlp to logits

        input_size = (transformer_dim * self.num_categories) + num_continuous

        if gmlp_enabled:
            self.mlp = gMLPClassification(
                patch_width=1,
                seq_len=input_size,
                num_classes=dim_out,
                dim=mlp_dimension,
                depth=mlp_depth
            )
        else:
            hidden_dimensions = []

            for i in range(mlp_depth):
                if mlp_dimension == -1:
                    hidden_dimensions.append((input_size // 8) * (2 ** (mlp_depth - i)))
                else:
                    hidden_dimensions.append(mlp_dimension)

            all_dimensions = [input_size, *hidden_dimensions, dim_out]
            self.mlp = MLP(all_dimensions, act=mlp_act)

    def forward(self, x_categ, x_cont=None):
        assert x_categ.shape[
                   -1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        x_categ += self.categories_offset

        x = self.transformer(x_categ)

        flat_categ = x.flatten(1)

        if self.num_continuous != 0:
            assert x_cont.shape[
                       1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

            if exists(self.continuous_mean_std):
                mean, std = self.continuous_mean_std.unbind(dim=-1)
                x_cont = (x_cont - mean) / std

            normed_cont = self.norm(x_cont)

            x = torch.cat((flat_categ, normed_cont), dim=-1)
        else:
            x = flat_categ

        return self.mlp(x)


# class GatedTabTransformer(nn.Module):
#     def __init__(
#             self,
#             *,
#             categories,
#             num_continuous,
#             input_tsa_dim,
#             input_tsb_dim,
#             transformer_dim,
#             transformer_depth,
#             transformer_heads,
#             transformer_dim_head=16,
#             dim_out=1,
#             mlp_depth=2,
#             mlp_act=None,
#             num_special_tokens=2,
#             continuous_mean_std=None,
#             attn_dropout=0.,
#             ff_dropout=0.,
#             gmlp_enabled=False,
#             mlp_dimension=32,
#     ):
#         super().__init__()
#         assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
#
#         # categories related calculations
#
#         self.num_categories = len(categories)
#         self.num_unique_categories = sum(categories)
#
#         # create category embeddings table
#
#         self.num_special_tokens = num_special_tokens
#         total_tokens = self.num_unique_categories + num_special_tokens
#
#         # for automatically offsetting unique category ids to the correct position in the categories embedding table
#
#         categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
#         categories_offset = categories_offset.cumsum(dim=-1)[:-1]
#         self.register_buffer('categories_offset', categories_offset)
#
#         # continuous
#
#         if exists(continuous_mean_std):
#             assert continuous_mean_std.shape == (num_continuous,
#                                                  2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
#         self.register_buffer('continuous_mean_std', continuous_mean_std)
#
#         self.norm = nn.LayerNorm(num_continuous)
#         self.num_continuous = num_continuous
#
#         # transformer
#
#         self.tab_transformer = Transformer(
#             num_tokens=total_tokens,
#             dim=transformer_dim,
#             depth=transformer_depth,
#             heads=transformer_heads,
#             dim_head=transformer_dim_head,
#             attn_dropout=attn_dropout,
#             ff_dropout=ff_dropout
#         )
#
#         self.tsa_transformer = Transformer(
#             num_tokens=total_tokens,
#             dim=transformer_dim,
#             depth=transformer_depth,
#             heads=transformer_heads,
#             dim_head=transformer_dim_head,
#             attn_dropout=attn_dropout,
#             ff_dropout=ff_dropout
#         )
#
#         self.tsb_transformer = Transformer(
#             num_tokens=total_tokens,
#             dim=transformer_dim,
#             depth=transformer_depth,
#             heads=transformer_heads,
#             dim_head=transformer_dim_head,
#             attn_dropout=attn_dropout,
#             ff_dropout=ff_dropout
#         )
#
#         # mlp to regression
#
#         input_size = (transformer_dim * self.num_categories) + num_continuous + () * transformer_dim
#
#         if gmlp_enabled:
#             self.mlp = gMLPRegression(
#                 patch_width=1,
#                 seq_len=input_size,
#                 mlp_dims=dim_out,
#                 dim=mlp_dimension,
#                 depth=mlp_depth
#             )
#         else:
#             hidden_dimensions = []
#
#             for i in range(mlp_depth):
#                 if mlp_dimension == -1:
#                     hidden_dimensions.append((input_size // 8) * (2 ** (mlp_depth - i)))
#                 else:
#                     hidden_dimensions.append(mlp_dimension)
#
#             all_dimensions = [input_size, *hidden_dimensions, dim_out]
#             self.mlp = MLP(all_dimensions, act=mlp_act)
#
#     def forward(self, x_categ, x_cont=None):
#         assert x_categ.shape[
#                    -1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
#         x_categ += self.categories_offset
#
#         x = self.transformer(x_categ)
#
#         flat_categ = x.flatten(1)
#
#         if self.num_continuous != 0:
#             assert x_cont.shape[
#                        1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'
#
#             if exists(self.continuous_mean_std):
#                 mean, std = self.continuous_mean_std.unbind(dim=-1)
#                 x_cont = (x_cont - mean) / std
#
#             normed_cont = self.norm(x_cont)
#
#             x = torch.cat((flat_categ, normed_cont), dim=-1)
#         else:
#             x = flat_categ
#
#         return self.mlp(x)


class GatedTabTransformer(torch.nn.Module):
    def __init__(self, configs):
        super(GatedTabTransformer, self).__init__()
        tab_config = configs.tab_model_args
        tsa_config = configs.tsa_model_args
        tsb_config = configs.tsb_model_args
    
        # ! Tabular data
        self.embedding = FeaturesEmbedding(field_dims=tab_config.cat_field_dims, embed_dim=tab_config.embed_dim)
        self.transformer = TabTransformerEncoder(tab_config.embed_dim, tab_config.depth, tab_config.n_heads,
                                                 tab_config.att_dropout, tab_config.ffn_mult, tab_config.ffn_dropout,
                                                 tab_config.ffn_act, tab_config.an_dropout)

        self.embed_tab_dim = len(tab_config.cat_field_dims) * tab_config.embed_dim + tab_config.cons_dims

        self.embed_output_dim = tab_config.embed_dim + 2 * tsa_config.d_model + 6 * tsb_config.d_model

        self.norm = torch.nn.LayerNorm(tab_config.cons_dims)
        
        self.output_dim = len(tab_config.cat_field_dims) + tab_config.cons_dims +\
                                2 * tsa_config.enc_in +\
                                6 * tsb_config.enc_in
        # ! Time-series A
        self.tsa_enc_embedding = DataEmbedding_wo_temp(tsa_config.enc_in, tsa_config.d_model, tsa_config.embed,
                                                       tsa_config.freq,
                                                       tsa_config.dropout)

        self.tsa_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, tsa_config.factor, attention_dropout=tsa_config.dropout,
                                      output_attention=tsa_config.output_attention), tsa_config.d_model,
                        tsa_config.n_heads),
                    tsa_config.d_model,
                    tsa_config.d_ff,
                    dropout=tsa_config.dropout,
                    activation=tsa_config.activation
                ) for l in range(tsa_config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(tsa_config.d_model)
        )

        # ! Time-series B
        self.tsb_enc_embedding = DataEmbedding_wo_temp(tsb_config.enc_in, tsb_config.d_model, tsb_config.embed,
                                                       tsb_config.freq,
                                                       tsb_config.dropout)

        self.tsb_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, tsb_config.factor, attention_dropout=tsb_config.dropout,
                                      output_attention=tsb_config.output_attention), tsb_config.d_model,
                        tsb_config.n_heads),
                    tsb_config.d_model,
                    tsb_config.d_ff,
                    dropout=tsb_config.dropout,
                    activation=tsb_config.activation
                ) for l in range(tsb_config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(tsb_config.d_model)
        )

        # ! Output MLP
        self.gmlp = gMLPRegression(
            patch_width=1,
            seq_len=self.embed_tab_dim,
            output_dims=tab_config.embed_dim,
            dim=self.embed_tab_dim,
            depth=tab_config.gmlp_depth
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
        # ! Tabular data
        embed_x = self.embedding(x_cat)
        # B, Category 변수 수, d_model -> Transformer 거쳐 나온 Categoryrical 변수들의 Embedding
        trans_out = self.transformer(embed_x)
        # B, Cons dimension -> continuous value embedding
        cons_x = self.norm(x_cons)
        # B, Category 변수 수 * d_model + Cons dimension
        all_x = torch.cat([trans_out.flatten(1), cons_x], dim=-1)
        all_x = self.gmlp(all_x)
        all_x = all_x.view(all_x.size(0), -1)

        # ! Time-series A
        tsa_enc_out = self.tsa_enc_embedding(x_tsa)
        tsa_enc_out, attns = self.tsa_encoder(tsa_enc_out, attn_mask=enc_self_mask)
        tsa_enc_out = tsa_enc_out.view(x_tsa.size(0), -1)

        # ! Time-series B
        tsb_enc_out = self.tsb_enc_embedding(x_tsb)
        tsb_enc_out, attns = self.tsb_encoder(tsb_enc_out, attn_mask=enc_self_mask)
        tsb_enc_out = tsb_enc_out.view(x_tsb.size(0), -1)

        all_output = torch.cat([all_x, tsa_enc_out, tsb_enc_out], dim=-1)

        out = self.mlp(all_output)

        return torch.sigmoid(out.squeeze(1))
