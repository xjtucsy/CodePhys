import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

# std = 2
def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k-mean) ** 2 / (2*std**2) ) / (math.sqrt(2 * math.pi) * std)

def kl_loss(inputs : torch.Tensor, labels : torch.Tensor):
    criterion = nn.KLDivLoss(reduction='none')
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    loss = loss.sum()
    return loss

class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output : torch.Tensor, k : torch.Tensor, N : int):
        device = output.device
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        k = k.type(torch.FloatTensor).to(device)
        two_pi_n_over_N = two_pi_n_over_N.to(device)
        hanning = hanning.to(device)
            
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output : torch.Tensor, Fs : float, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0 # 转换到频率
        k = feasible_bpm / unit_per_hz
        
        # only calculate feasible PSD range [0.7,4]Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)

        return (1.0 / complex_absolute.sum()) * complex_absolute	# Analogous Softmax operator
        
        
    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs): # inputs: 160, target:1
        device = inputs.device
        inputs = inputs.view(1, -1) # 160
        target = target.view(1, -1) # 1
        bpm_range = torch.arange(40, 180, dtype=torch.float).to(device) # 140

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range) # [140,1]，对应140个类别

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0) # max返回（values, indices）
        whole_max_idx = whole_max_idx.type(torch.float) # 功率谱密度的峰值对应频率即为心率
        
        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

    
    @staticmethod
    def cross_entropy_power_spectrum_DLDL_softmax2(inputs, target, Fs, std):
        device = inputs.device
        
        # 生成目标的心率分布
        target_distribution = [normal_sampling(int(target), i, std) for i in range(140)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(device)
                
        rank = torch.Tensor([i for i in range(140)]).to(device)
        
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        
        bpm_range = torch.arange(40, 180, dtype=torch.float).to(device)

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)    # 计算pred的功率谱密度
        
        fre_distribution = F.softmax(complex_absolute.view(-1), dim=0)   # 计算pred的心率分布
        loss_distribution_kl = kl_loss(fre_distribution, target_distribution)   # 计算两个分布之间的距离
                
        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return loss_distribution_kl, F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)
    
    @staticmethod
    def cross_entropy_power_spectrum_HRV_softmax2(inputs, target, Fs, std):
        device = inputs.device
        
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        
        bpm_range = torch.arange(2.4, 24, dtype=torch.float).to(device)

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)    # 计算pred的功率谱密度
        target_complex_absolute = TorchLossComputer.complex_absolute(target, Fs, bpm_range)    # 计算target的功率谱密度
        
        fre_distribution = F.softmax(complex_absolute.view(-1), dim=0)   # 计算pred的心率分布
        target_fre_distribution = F.softmax(target_complex_absolute.view(-1), dim=0)   # 计算target的心率分布
        loss_distribution_kl = kl_loss(fre_distribution, target_fre_distribution)   # 计算两个分布之间的距离
                
        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        target_max_val, target_max_idx = target_complex_absolute.view(-1).max(0)
        whole_max_idx = (whole_max_idx.type(torch.float) + 2) / 60
        target_max_idx = (target_max_idx.type(torch.float) +2) / 60

        return loss_distribution_kl, F.cross_entropy(complex_absolute, target_complex_absolute),  torch.abs(target_max_idx - whole_max_idx)
        