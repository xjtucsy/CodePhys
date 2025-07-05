import torch
import torch.nn as nn
    
class FeatureLossComputerSpatial(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(FeatureLossComputerSpatial, self).__init__()
        self.distance_func = torch.nn.SmoothL1Loss(beta=2.0)
        return
    def forward(self, video_feature, ecg_feature, spatial_weight_map):  # all variable operation
        # feature_video: [B, 64, T, 8, 8]
        # feature_ecg: [B, 64, T]
        # spatial_weight_map: [B, 8, 8]
        
        # spatial_feature_loss: [B, 4, 4]
        spatial_feature_loss = torch.zeros_like(spatial_weight_map)
        for i in range(spatial_weight_map.shape[0]):
            for j in range(spatial_weight_map.shape[1]):
                for k in range(spatial_weight_map.shape[2]):
                    spatial_feature_loss[i,j,k] = self.distance_func(video_feature[i,:,:,j,k], ecg_feature[i,:,:,j,k])  # [64, T] with [64, T]

        spatial_feature_loss = spatial_feature_loss * spatial_weight_map
        spatial_feature_loss = torch.mean(spatial_feature_loss)
        
        return spatial_feature_loss
