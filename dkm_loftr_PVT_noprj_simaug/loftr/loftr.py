import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer

from dkm_loftr_PVT_noprj_simaug.loftr.utils.supervision import spvs_coarse
from dkm_loftr_PVT_noprj_simaug.loftr.losses.loftr_loss import LoFTRLoss
from dkm_loftr_PVT_noprj_simaug.loftr.config.default import get_cfg_defaults

from tools.tools import lower_config

class LoFTR(nn.Module):
    def __init__(self, config, outputfeature='concatenated'):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])

        self.outputfeature = outputfeature
        assert self.outputfeature in ['transformer', 'fpn', 'concatenated']

        self.temperature = 0.1
        self.INF = 1e9

        self.loftr_loss = LoFTRLoss(lower_config(get_cfg_defaults()))

    def format_output_feature(self, feat_c0, feat_c0_fpn, feat_m0, feat_c1, feat_c1_fpn, feat_m1):
        if self.outputfeature == 'transformer':
            f_q_pyramid = {
                8: feat_c0, 4: feat_m0
            }
            f_s_pyramid = {
                8: feat_c1, 4: feat_m1
            }
        elif self.outputfeature == 'fpn':
            f_q_pyramid = {
                8: feat_c0_fpn, 4: feat_m0
            }
            f_s_pyramid = {
                8: feat_c1_fpn, 4: feat_m1
            }
        elif self.outputfeature == 'concatenated':
            f_q_pyramid = {
                8: torch.cat([feat_c0, feat_c0_fpn], dim=1), 4: feat_m0
            }
            f_s_pyramid = {
                8: torch.cat([feat_c1, feat_c1_fpn], dim=1), 4: feat_m1
            }
        else:
            raise NotImplementedError()

        return f_q_pyramid, f_s_pyramid

    def forward(self, data):
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
        data.update({
            'bs': data['image0_color'].size(0),
            'hw0_i': data['image0_color'].shape[2:], 'hw1_i': data['image1_color'].shape[2:]
        })
        a = 1
        assert data['hw0_i'] == data['hw1_i']

        images = torch.cat([data['image0_color'], data['image1_color']], dim=0)
        masks = torch.cat([data['mask0_'], data['mask1_']], dim=0)
        feats_c, feats_m = self.backbone(images, masks)

        (feat_c0, feat_c1), (feat_m0, feat_m1) = feats_c.split(data['bs']), feats_m.split(data['bs'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:]
        })

        # Preserve FPN Feature
        feat_c0_fpn = feat_c0
        feat_c1_fpn = feat_c1

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        f_q_pyramid, f_s_pyramid = self.format_output_feature(rearrange(feat_c0, 'n (h w) c -> n c h w', h=data['hw0_c'][0], w=data['hw0_c'][1]),
                                                              feat_c0_fpn, feat_m0,
                                                              rearrange(feat_c1, 'n (h w) c -> n c h w', h=data['hw1_c'][0], w=data['hw1_c'][1]),
                                                              feat_c1_fpn, feat_m1)


        # Compute Supervision
        sim_matrix = self.coarse_alignment(feat_c0, feat_c1)
        spvs_coarse(data)
        conf_matrix_sprv = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        data['conf_matrix_sprv'] = conf_matrix_sprv

        loftr_loss = self.loftr_loss(data) * 0.002

        return f_q_pyramid, f_s_pyramid, loftr_loss

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