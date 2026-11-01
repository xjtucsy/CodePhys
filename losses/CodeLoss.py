import torch
import torch.nn.functional as F
from archs.code_rppgnet import CodeRppgNet

def code_loss(code_logits: torch.Tensor, gt_rppg: torch.Tensor, codebook_module_blocked: CodeRppgNet):
    
    rppg, bs = gt_rppg, gt_rppg.shape[0]

    encode_feat = codebook_module_blocked.encoder(rppg)
    distances = torch.sum(encode_feat**2, dim=1, keepdim=True) +\
            torch.sum(codebook_module_blocked.codebook.weight**2, dim=1) - 2 * torch.matmul(encode_feat, codebook_module_blocked.codebook.weight.t())
    
    gt_indices = torch.argmin(distances, dim=1).detach()
    return F.cross_entropy(code_logits, gt_indices, reduction='mean')
