import torch
import random
import numpy as np
import os
import torch.nn as nn
import julius

from einops import rearrange

from archs.code_rppgnet import EmbedLayer

def set_seed(seed=92):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VideoFeatureExtractor(nn.Module):
    '''
    this extract the video feature from the input video
    the network structure contains first 7 Conv layers of PhysNet
    '''
    def __init__(self):
        super(VideoFeatureExtractor, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1], stride=[2, 1, 1],
                               padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1], stride=[2, 1, 1],
                               padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

    def forward(self, x):  # x [3, T, 128,128]
        
        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        x = self.MaxpoolSpaTem(x_visual6464)  # x [32, T/2, 32,32]    Temporal halve

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x) # x [64, T/4, 8,8]
        
        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]
       
        return x


class PPGFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(PPGFeatureExtractor, self).__init__()
        self.args = args
        self.hidden_dim = args.enc_hidden_dim
        self.hidden_inner_dim = args.enc_hidden_inner_dim
        self.num_layers = args.enc_num_layers
        self.num_rppg = args.num_rppg
        self.vq_dim = args.vq_token_dim
        self.vq_num = args.num_rppg
        self.dropout_ratio = args.enc_dropout_ratio

        self.start_embed = nn.Conv1d(1, self.hidden_dim, kernel_size=1, stride=1, padding=0)
        self.embed_layer = EmbedLayer(self.hidden_dim, self.hidden_inner_dim, self.num_rppg, self.dropout_ratio)
        self.layers = nn.ModuleList([
            self.embed_layer for _ in range(self.num_layers)
        ])

        self.feature_token = nn.Conv1d(self.hidden_dim, self.vq_dim, kernel_size=3, stride=1, padding=1)
        self.norm_layer = nn.LayerNorm(self.vq_num, elementwise_affine=True)

    def forward(self, x : torch.Tensor):   # x : [B, T]
        B, T = x.shape
        x = x.unsqueeze(1)  # x : [B, 1, T]
        x = self.start_embed(x) # x : [bs, enc_hidden_dim, num_rppg]
        for layer in self.layers:
            x = layer(x)
        x = self.feature_token(x) # x : [bs, vq_dim, vq_num]
        x = self.norm_layer(x)   # x : [bs, vq_dim, vq_num]

        x = x.transpose(1, 2).flatten(0,1) 

        x = rearrange(x, '(b t) c -> b c t', b = B, t=T)
        x = x.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, 8, 8)
        return x  # x : [bs*vq_num, vq_dim]

class RppgEstimator(nn.Module):
    '''
    this estimate the rPPG signal from the input video feature (or ecg feature)
    the network structure contains last 3 Conv layers of PhysNet
    '''
    def __init__(self, frames=160):
        super(RppgEstimator, self).__init__()

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x): # x [64, T, 8, 8]
        

        # print(x.shape)
        
        x = self.poolspa(x)  # x [64, T, 1,1]    -->  groundtruth left and right - 7
        # print(x.shape)
        x = self.ConvBlock10(x)  # x [1, T, 1,1]
        # print(x.shape)
        rPPG_peak = x.squeeze(-1).squeeze(-1).squeeze(1)  # [Batch, T]
        return rPPG_peak

class WeightMapGenerator(nn.Module):
    '''
    this generate the weight map from the input video feature
    '''
    def __init__(self):
        super(WeightMapGenerator, self).__init__()
        # spatial weight map
        self.spatial_attention = nn.Conv2d(2, 1, 7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
                    

    def forward(self, x):
        
        # Input: x [B, 64, 160, 8, 8]
        # Output: spatial map [B, 8, 8]
        
        spatial_map = rearrange(x, 'b c t h w -> b (c t) h w')
        spatial_map_avg = torch.mean(spatial_map, dim=1, keepdim=True) # [B, 1, 8, 8]
        spatial_map_max = torch.max(spatial_map, dim=1, keepdim=True)[0] # [B, 1, 8, 8]
        spatial_map = torch.cat([spatial_map_avg, spatial_map_max], dim=1) # [B, 2, 8, 8]
        spatial_map = self.spatial_attention(spatial_map).squeeze(1) # [B, 8, 8]
        spatial_map = spatial_map.squeeze(1) # [B, 8*8]
        spatial_map = self.sigmoid(spatial_map)
        spatial_map = spatial_map.view(-1, 8, 8) # x [B, 8, 8]
        
        return spatial_map
 
