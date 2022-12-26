import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer

class LoFTR(nn.Module):
    def __init__(self, config, outputfeature='concatenated'):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])

        self.outputfeature = outputfeature
        assert self.outputfeature in ['transformer', 'fpn', 'concatenated']

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=320,
                out_channels=8 ** 2 * 3,
                kernel_size=1),
            nn.PixelShuffle(8),
        )

        self.out_conv_pair = nn.Sequential(
            nn.Conv2d(
                in_channels=320,
                out_channels=8 ** 2 * 3,
                kernel_size=1),
            nn.PixelShuffle(8),
        )

    def forward(self, img, mask, img2=None):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        feats_c = self.backbone(img, mask)

        if img2 is not None:
            feats_c1 = self.backbone(img2, torch.zeros_like(mask))

            n, c, h, w = feats_c.shape

            feats_c = rearrange(self.pos_encoding(feats_c), 'n c h w -> n (h w) c')
            feats_c1 = rearrange(self.pos_encoding(feats_c1), 'n c h w -> n (h w) c')

            feats_c, feats_c1 = self.loftr_coarse(feats_c, feats_c1)

            feats_c = rearrange(feats_c, 'n (h w) c -> n c h w', h=h, w=w)
            feats_c1 = rearrange(feats_c1, 'n (h w) c -> n c h w', h=h, w=w)

            rgb_recon = self.out_conv_pair(feats_c)
        else:
            rgb_recon = self.out_conv(feats_c)

        patch_sizeh = int(img.shape[2] / mask.shape[1])
        patch_sizew = int(img.shape[3] / mask.shape[2])
        assert patch_sizeh == patch_sizew
        mask = mask.repeat_interleave(patch_sizeh, 1).repeat_interleave(patch_sizew, 2).unsqueeze(1).contiguous()

        in_chans = 3
        loss_recon = F.l1_loss(img, rgb_recon, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / in_chans
        loss2 = (loss_recon * (1 - mask)).sum() / ((1 - mask).sum() + 1e-5) / in_chans
        loss = loss + loss2 * 0.05
        return loss, rgb_recon

    # def forward(self, data):
    #     """
    #     Update:
    #         data (dict): {
    #             'image0': (torch.Tensor): (N, 1, H, W)
    #             'image1': (torch.Tensor): (N, 1, H, W)
    #             'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
    #             'mask1'(optional) : (torch.Tensor): (N, H, W)
    #         }
    #     """
    #     # 1. Local Feature CNN
    #     data.update({
    #         'bs': data['query'].size(0),
    #         'hw0_i': data['query'].shape[2:], 'hw1_i': data['support'].shape[2:]
    #     })
    #
    #     if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
    #         feats_c, feats_m, feats_f = self.backbone(
    #             torch.cat([data['query'], data['support']], dim=0)
    #         )
    #         (feat_c0, feat_c1), (feat_m0, feat_m1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_m.split(data['bs']), feats_f.split(data['bs'])
    #     else:  # handle different input shapes
    #         (feat_c0, feat_m0, feat_f0), (feat_c1, feat_m1, feat_f1) = self.backbone(data['query']), self.backbone(data['support'])
    #
    #     data.update({
    #         'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
    #         'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
    #     })
    #
    #     # Preserve FPN Feature
    #     feat_c0_fpn = feat_c0
    #     feat_c1_fpn = feat_c1
    #
    #     # 2. coarse-level loftr module
    #     # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
    #     feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
    #     feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
    #
    #     mask_c0 = mask_c1 = None  # mask is useful in training
    #     if 'query_mask' in data:
    #         mask_c0, mask_c1 = data['query_mask'].flatten(-2), data['support_mask'].flatten(-2)
    #     feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
    #
    #     f_q_pyramid, f_s_pyramid = self.format_output_feature(data['query'],
    #                                                           rearrange(feat_c0, 'n (h w) c -> n c h w', h=data['hw0_c'][0], w=data['hw0_c'][1]),
    #                                                           feat_c0_fpn, feat_m0, feat_f0,
    #                                                           data['support'],
    #                                                           rearrange(feat_c1, 'n (h w) c -> n c h w', h=data['hw1_c'][0], w=data['hw1_c'][1]),
    #                                                           feat_c1_fpn, feat_m1, feat_f1)
    #
    #     sim_matrix = self.coarse_alignment(feat_c0, feat_c1)
    #
    #     if self.training:
    #         # Compute Supervision
    #         conf_matrix_sprv = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
    #         # data['conf_matrix_sprv'] = conf_matrix_sprv
    #     else:
    #         conf_matrix_sprv = None
    #
    #     return f_q_pyramid, f_s_pyramid, sim_matrix, conf_matrix_sprv