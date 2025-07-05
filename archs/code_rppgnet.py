import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from einops.layers.torch import Rearrange

from archs.transformer_layer import DeformConv2d, CDC_T

class EmbedLayer(nn.Module):
    def __init__(self, hidden_dim, hidden_inner_dim, layer_norm_size, dropout_ratio) -> None:
        super(EmbedLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_inner_dim = hidden_inner_dim
        self.layer_norm_size = layer_norm_size
        self.dropout_ratio = dropout_ratio

        self.up_sample = nn.Conv1d(self.hidden_dim, self.hidden_inner_dim, kernel_size=3, stride=1, padding=1)
        self.layer_norm1 = nn.LayerNorm(self.layer_norm_size, elementwise_affine=False)
        self.affine1 = nn.Conv1d(self.hidden_inner_dim, self.hidden_inner_dim, kernel_size=1, stride=1, padding=0)
        self.down_sample = nn.ConvTranspose1d(self.hidden_inner_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
        self.layer_norm2 = nn.LayerNorm(self.layer_norm_size, elementwise_affine=False)
        self.affine2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.up_sample(x)
        x = self.layer_norm1(x)
        x = x + self.affine1(x)
        x = nn.GELU()(x)
        x = nn.Dropout(self.dropout_ratio)(x)
        x = self.down_sample(x)
        x = self.layer_norm2(x)
        x = x + self.affine2(x)
        # x = nn.GELU()(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.hidden_dim = args.enc_hidden_dim
        self.hidden_inner_dim = args.enc_hidden_inner_dim
        self.num_layers = args.enc_num_layers
        self.num_rppg = args.num_rppg
        self.vq_dim = args.vq_token_dim
        self.vq_num = args.num_rppg // 4
        self.dropout_ratio = args.enc_dropout_ratio

        self.start_embed = nn.Conv1d(1, self.hidden_dim, kernel_size=1, stride=1, padding=0)
        self.embed_layer = EmbedLayer(self.hidden_dim, self.hidden_inner_dim, self.num_rppg, self.dropout_ratio)
        self.layers = nn.ModuleList([
            self.embed_layer for _ in range(self.num_layers)
        ])
        self.feature_token = nn.Conv1d(self.hidden_dim, self.vq_dim, kernel_size=5, stride=4, padding=2) # /4 downsample
        self.norm_layer = nn.LayerNorm(self.vq_num, elementwise_affine=False)
        self.affine = nn.Conv1d(self.vq_dim, self.vq_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x : torch.Tensor):   # x : [B, T] 
        x = x.unsqueeze(1)  # x : [B, 1, T]
        x = self.start_embed(x) # x : [bs, enc_hidden_dim, num_rppg]
        for layer in self.layers:
            x = layer(x)
        x = self.feature_token(x) # x : [bs, vq_dim, vq_num]
        x = self.norm_layer(x)   # x : [bs, vq_dim, vq_num]
        x = x + self.affine(x)   # x : [bs, vq_dim, vq_num]
        return x.transpose(1, 2).flatten(0,1)     # x : [bs*vq_num, vq_dim]

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.hidden_dim = args.dec_hidden_dim
        self.hidden_inner_dim = args.dec_hidden_inner_dim
        self.num_layers = args.dec_num_layers
        self.num_rppg = args.num_rppg
        self.vq_dim = args.vq_token_dim
        self.vq_num = args.num_rppg
        self.dropout_ratio = args.dec_dropout_ratio

        self.start_embed = nn.Conv1d(self.vq_dim, self.hidden_dim, kernel_size=1, stride=1, padding=0)
        self.embed_layer = EmbedLayer(self.hidden_dim, self.hidden_inner_dim, self.vq_num // 4, self.dropout_ratio)
        self.layers = nn.ModuleList([
            self.embed_layer for _ in range(self.num_layers)
        ])
        self.recover_token = nn.ConvTranspose1d(self.hidden_dim, 1, kernel_size=5, stride=4, padding=2, output_padding=3) # *4 upsample
        self.norm_layer = nn.LayerNorm(self.num_rppg, elementwise_affine=False)
        self.affine = nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x : torch.Tensor): # x : [bs * $vq_token_num$, vq_token_dim]
        x = x.view(-1, self.vq_num // 4, self.vq_dim) # x : [bs, vq_token_num, vq_token_dim]
        x = x.transpose(1, 2)  # x : [bs, vq_token_dim, vq_token_num]
        x = self.start_embed(x) # x : [bs, dec_hidden_dim, vq_token_num]
        for layer in self.layers:
            x = layer(x)
        x = self.recover_token(x) # x : [bs, 1, num_rppg]
        x = self.norm_layer(x)  # x : [bs, 1, num_rppg]
        x = x + self.affine(x)  # x : [bs, 1, num_rppg]
        return x.squeeze(1) # x : [bs, num_rppg]

class LocalStream(nn.Module):
    def __init__(self, args):
        super(LocalStream, self).__init__()
        self.args = args
        self.hidden_dim = args.ada_hidden_dim
        self.hidden_inner_dim = args.ada_hidden_inner_dim
        self.num_layers = args.ada_num_layers
        self.vq_dim = args.vq_token_dim
        self.vq_num = args.num_rppg
        self.dropout_ratio = args.ada_dropout_ratio
        self.feature_size = 8

        self.tdc_conv_1 = nn.Sequential(
            CDC_T(self.vq_dim, self.vq_dim, kernel_size=3, stride=(1, 2, 2), padding=1, theta=0.7),
            nn.LayerNorm([self.feature_size // 2, self.feature_size // 2]),
            nn.GELU()
        )

        self.tdc_conv_2 = nn.Sequential(
            CDC_T(self.vq_dim, self.vq_dim, kernel_size=3, stride=(2, 2, 2), padding=1, theta=0.7),
            nn.LayerNorm([self.feature_size // 4, self.feature_size // 4]),
            nn.GELU()
        )

        self.tdc_conv_3 = nn.Sequential(
            CDC_T(self.vq_dim, self.vq_dim, kernel_size=3, stride=(2, 2, 2), padding=1, theta=0.7),
            nn.LayerNorm([self.feature_size // 8, self.feature_size // 8]),
            nn.GELU()
        )

        self.deform_conv_1 = nn.Sequential(
            Rearrange('b c n h w -> (b n) c h w'),
            DeformConv2d(self.vq_dim, self.vq_dim, kernel_size=3, stride=2, padding=1),
            Rearrange('(b n) c h w -> b c n h w', n=self.vq_num),
            nn.LayerNorm([self.feature_size // 2, self.feature_size // 2]),
            nn.GELU()
        )

        self.deform_conv_2 = nn.Sequential(
            Rearrange('b c n h w -> (b n) c h w'),
            DeformConv2d(self.vq_dim, self.vq_dim, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([self.feature_size // 4, self.feature_size // 4]),
            Rearrange('(b n) c h w -> (b h w) c n', n=self.vq_num),
            nn.Conv1d(self.vq_dim, self.vq_dim, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(self.vq_num // 2, elementwise_affine=False),
            nn.Conv1d(self.vq_dim, self.vq_dim, kernel_size=1, stride=1, padding=0),
            Rearrange('(b h w) c n -> b c n h w', h=self.feature_size // 4, w=self.feature_size // 4),
            nn.GELU()
        )

        self.deform_conv_3 = nn.Sequential(
            Rearrange('b c n h w -> (b n) c h w'),
            DeformConv2d(self.vq_dim, self.vq_dim, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([self.feature_size // 8, self.feature_size // 8]),
            Rearrange('(b n) c h w -> (b h w) c n', n=self.vq_num // 2),
            nn.Conv1d(self.vq_dim, self.vq_dim, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(self.vq_num // 4, elementwise_affine=False),
            nn.Conv1d(self.vq_dim, self.vq_dim, kernel_size=1, stride=1, padding=0),
            Rearrange('(b h w) c n -> b c n h w', h=self.feature_size // 8, w=self.feature_size // 8),
            nn.GELU()
        )

        self.embed_layer = EmbedLayer(self.hidden_dim, self.hidden_inner_dim, self.vq_num // 4, self.dropout_ratio)
        self.layers = nn.ModuleList([
            self.embed_layer for _ in range(self.num_layers)
        ])

        self.norm_layer = nn.LayerNorm(self.vq_num // 4, elementwise_affine=False)
        self.affine = nn.Conv1d(self.hidden_dim, self.vq_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):   # [bs, vq_token_dim, vq_token_num, 8, 8]
        time_feature_1 = self.tdc_conv_1(x) # [bs, vq_token_dim, vq_token_num, 4, 4]
        spatial_feature_1 = self.deform_conv_1(x) # [bs, vq_token_dim, vq_token_num, 4, 4]
        fusion_feature_1 = time_feature_1 + spatial_feature_1 # [bs, vq_token_dim, vq_token_num, 4, 4]

        time_feature_2 = self.tdc_conv_2(fusion_feature_1) # [bs, vq_token_dim, vq_token_num, 2, 2]
        spatial_feature_2 = self.deform_conv_2(fusion_feature_1) # [bs, vq_token_dim, vq_token_num // 2, 2, 2]
        fusion_feature_2 = time_feature_2 + spatial_feature_2 # [bs, vq_token_dim, vq_token_num, 2, 2]

        time_feature_3 = self.tdc_conv_3(fusion_feature_2) # [bs, vq_token_dim, vq_token_num, 1, 1]
        spatial_feature_3 = self.deform_conv_3(fusion_feature_2) # [bs, vq_token_dim, vq_token_num // 4, 1, 1]
        fusion_feature_3 = time_feature_3 + spatial_feature_3 # [bs, vq_token_dim, vq_token_num, 1, 1]

        x = fusion_feature_3.squeeze(-1).squeeze(-1)   # [bs, hidden_dim, vq_token_num]
        for layer in self.layers:
            x = layer(x)
        x = self.norm_layer(x)   # [bs, hidden_dim, vq_token_num]
        x = self.affine(x)   # [bs, vq_token_dim, vq_token_num]
        return x

class LocalGlobalMixer(nn.Module):
    def __init__(self, args):
        super(LocalGlobalMixer, self).__init__()
        self.args = args
        
    def forward(self, local_rppg, global_rppg): # local_roog: [bs*vq_token_num, vq_token_dim]
        # AdaIN
        local_rppg_mean = torch.mean(local_rppg, dim=1, keepdim=True) # [bs*vq_token_num, 1]
        local_rppg_std = torch.std(local_rppg, dim=1, keepdim=True) # [bs*vq_token_num, 1]
        local_rppg = (local_rppg - local_rppg_mean) / (local_rppg_std + 1e-8) # [bs*vq_token_num, vq_token_dim]
        global_rppg_mean = torch.mean(global_rppg, dim=1, keepdim=True) # [bs, 1]
        global_rppg_std = torch.std(global_rppg, dim=1, keepdim=True) # [bs, 1]
        new_global_rppg = global_rppg_std * local_rppg + global_rppg_mean # [bs*vq_token_num, vq_token_dim]

        # mix
        mix_rppg = local_rppg + new_global_rppg # [bs*vq_token_num, vq_token_dim]
        return mix_rppg

class CodeRppgNet(nn.Module):
    def __init__(self, args):
        super(CodeRppgNet, self).__init__()
        ## hyperparameters
        self.args = args
        self.token_class_num = args.token_class_num
        self.vq_token_dim = args.vq_token_dim
        self.fmix_rate = args.fmix_rate

        ## network module
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.rppg_local_stream = LocalStream(args)
        self.local_global_mixer = LocalGlobalMixer(args)

        ## codebook module
        self.codebook = nn.Embedding(self.token_class_num, self.vq_token_dim)

        ## init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, rppg, rppg_feature = None): # rppg: [bs, seq_len]

        encode_feat = self.encoder(rppg) # [bs*vq_token_num, vq_token_dim]

        if self.args.fmix_rate > 0: # use rppg feature  [bs, vq_token_dim, vq_token_num, 8, 8]
            encode_rppg_feat = self.rppg_local_stream(rppg_feature) # [bs, vq_token_dim, vq_token_num]
            encode_rppg_feat = encode_rppg_feat.transpose(2, 1).flatten(0,1) # [bs*vq_token_num, vq_token_dim]
            encode_feat = self.local_global_mixer(encode_feat, encode_rppg_feat) # [bs*vq_token_num, vq_token_dim]
        
        distances = torch.sum(encode_feat**2, dim=1, keepdim=True) +\
              torch.sum(self.codebook.weight**2, dim=1) - 2 * torch.matmul(encode_feat, self.codebook.weight.t())
            
        encoding_indices = torch.argmin(distances, dim=1)   # [bs * $vq_token_num$]
        encodings = torch.zeros(encoding_indices.shape[0], self.token_class_num, device=encode_feat.device) # [bs * $vq_token_num$, token_class_num]
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1) # [bs * $vq_token_num$, token_class_num]

        part_token_feat = torch.matmul(encodings, self.codebook.weight) # [bs * $vq_token_num$, vq_token_dim]

        e_latent_loss = 0.25 * F.mse_loss(part_token_feat.detach(), encode_feat) + F.mse_loss(encode_feat.detach(), part_token_feat)
        part_token_feat = encode_feat + (part_token_feat - encode_feat).detach()

        
        # Decoder of Tokenizer, Recover the rppg.
        recoverd_rppg = self.decoder(part_token_feat) # [bs, seq_len]

        return recoverd_rppg, encoding_indices, e_latent_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## general parameters
    parser.add_argument('--num_rppg', type=int, default=160)

    ## encoder parameters
    parser.add_argument('--enc_hidden_dim', type=int, default=512)
    parser.add_argument('--enc_hidden_inner_dim', type=int, default=512)
    parser.add_argument('--enc_num_layers', type=int, default=4)
    parser.add_argument('--enc_dropout_ratio', type=float, default=0.0)
    
    ## decoder parameters
    parser.add_argument('--dec_hidden_dim', type=int, default=32)
    parser.add_argument('--dec_hidden_inner_dim', type=int, default=64)
    parser.add_argument('--dec_num_layers', type=int, default=1)
    parser.add_argument('--dec_dropout_ratio', type=float, default=0.0)

    ## adapter parameters
    parser.add_argument('--ada_hidden_dim', type=int, default=64, help='equal to vq_token_dim')
    parser.add_argument('--ada_hidden_inner_dim', type=int, default=128)
    parser.add_argument('--ada_num_layers', type=int, default=2)
    parser.add_argument('--ada_dropout_ratio', type=float, default=0.0)

    ## codebook parameters
    parser.add_argument('--vq_token_dim', type=int, default=64, help='codebook embedding dim')
    parser.add_argument('--token_class_num', type=int, default=64, help='codebook size')
    parser.add_argument('--fmix_rate', type=float, default=0.1)

    args = parser.parse_args()

    rppg = torch.randn(2, 160)
    rppg_feature = torch.randn(2, 64, 160, 8, 8)

    encoder = Encoder(args).cuda()
    decoder = Decoder(args)
    rppg_local_stream = LocalStream(args)
    local_global_mixer = LocalGlobalMixer(args)
