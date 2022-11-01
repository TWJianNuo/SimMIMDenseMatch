import torch
import torch.nn as nn
from .backbone.resnet_swin_fpn import ResNetSwinFPN

class LoFTR(nn.Module):
    def __init__(self, config_simmim, config, outputfeature='concatenated'):
        super().__init__()
        # Misc
        self.config = config
        self.config_simmim = config_simmim

        # Modules
        self.backbone = ResNetSwinFPN(config_simmim=config_simmim, config_resfpn=config['resnetfpn'])

        self.outputfeature = outputfeature
        assert self.outputfeature in ['transformer', 'fpn', 'concatenated']

    def format_output_feature(self, image0, feat_c0, feat_c0_fpn, feat_m0, feat_f0, image1, feat_c1, feat_c1_fpn, feat_m1, feat_f1):
        if self.outputfeature == 'transformer':
            f_q_pyramid = {
                8: feat_c0, 4: feat_m0, 2: feat_f0, 1: image0
            }
            f_s_pyramid = {
                8: feat_c1, 4: feat_m1, 2: feat_f1, 1: image1
            }
        elif self.outputfeature == 'fpn':
            f_q_pyramid = {
                8: feat_c0_fpn, 4: feat_m0, 2: feat_f0, 1: image0
            }
            f_s_pyramid = {
                8: feat_c1_fpn, 4: feat_m1, 2: feat_f1, 1: image1
            }
        elif self.outputfeature == 'concatenated':
            f_q_pyramid = {
                8: torch.cat([feat_c0, feat_c0_fpn], dim=1), 4: feat_m0, 2: feat_f0, 1: image0
            }
            f_s_pyramid = {
                8: torch.cat([feat_c1, feat_c1_fpn], dim=1), 4: feat_m1, 2: feat_f1, 1: image1
            }
        else:
            raise NotImplementedError()

        return f_q_pyramid, f_s_pyramid

    def forward(self, img, mask):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        x3_out, x2_out = self.backbone(img, mask)
        f_pyramid = {8: x3_out, 4: x2_out}
        return f_pyramid

    def coarse_alignment(self, feat_c0, feat_c1, mask_c0=None, mask_c1=None):
        # Compute Initial Coarse Alignment
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5, [feat_c0, feat_c1])

        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
        if mask_c0 is not None:
            sim_matrix.masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -self.INF)
        # conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        return sim_matrix

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
