import torch
import math 
import numpy as np

from torch import nn
from torchvision.ops import deform_conv2d
from torch.nn import functional as F

'''
Temporal Center-difference based Convolutional layer (3D version)
theta: control the percentage of original convolution and centeral-difference convolution
'''
class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):

        super(CDC_T, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, t, kernel_size, kernel_size] = self.conv.weight.shape

            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2] > 1:
                kernel_diff = self.conv.weight[:, :, 0, :, :].sum(2).sum(2) + self.conv.weight[:, :, 2, :, :].sum(
                    2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                    padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
                return out_normal - self.theta * out_diff

            else:
                return out_normal

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=1):
        super(DeformConv2d, self).__init__()
        self.padding = padding
        self.stride = stride
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=stride, padding=padding) # vanilla conv
 
        self.conv_offset = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_offset.weight.data.zero_()
 
        self.conv_mask = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_mask.weight.data.fill_(0.5) # initialize to 0.5
 
    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x)) # ensure mask in [0, 1]
        out = deform_conv2d(input=x, offset=offset, 
                            weight=self.conv.weight, 
                            mask=mask, stride=self.stride, padding=self.padding)
        return out
