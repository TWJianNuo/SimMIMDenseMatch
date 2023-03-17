import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

import numpy as np

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

        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=320,
                out_channels=8 ** 2 * 3,
                kernel_size=1),
            nn.PixelShuffle(8),
        )

        self.temperature = 0.1

    def forward(self, img1, mask1, img2):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # assert masksup is None

        # Change to Evaluation Mode for Batch Statistics
        mode = self.training
        self.backbone.train()
        feats1_mask = self.backbone(img1, mask1, maskedin=True)
        self.backbone.train(mode)

        bz = img1.shape[0]
        maskzero = torch.zeros_like(mask1)
        feats2 = self.backbone(img2, maskzero, maskedin=False)

        n, ch, h, w = feats2.shape
        feats2 = rearrange(self.pos_encoding(feats2), 'n c h w -> n (h w) c')
        feats1_mask = rearrange(self.pos_encoding(feats1_mask), 'n c h w -> n (h w) c')

        feats1_mask_, _ = self.loftr_coarse(feats1_mask, feats2)
        feats1_mask__ = rearrange(feats1_mask_, 'n (h w) c -> n c h w', h=h, w=w)
        recon_initial = self.out(feats1_mask__)

        patch_sizeh = int(img1.shape[2] / mask1.shape[1])
        patch_sizew = int(img1.shape[3] / mask1.shape[2])
        assert patch_sizeh == patch_sizew
        mask1 = mask1.repeat_interleave(patch_sizeh, 1).repeat_interleave(patch_sizew, 2).unsqueeze(1).contiguous()

        in_chans = 3
        loss_recon = F.l1_loss(img1, recon_initial, reduction='none')

        mask_pos = mask1
        loss_pos_initial = (loss_recon * mask_pos).sum() / (mask_pos.sum() + 1e-5) / in_chans

        mask_neg = 1 - mask_pos
        loss_neg = (loss_recon * mask_neg).sum() / (mask_neg.sum() + 1e-5) / in_chans

        loss = loss_pos_initial + loss_neg * 0.05

        return loss, recon_initial

    def extrac_feature(self, img, img2):
        # Change to Evaluation Mode for Batch Statistics
        n, c, h, w = img.shape
        mask = torch.zeros([n, int(h / 2), int(w / 2)], device=img.device)
        feats_c = self.backbone(img, mask, maskedin=False)
        feats_c1 = self.backbone(img2, mask, maskedin=False)

        n, c, h, w = feats_c.shape

        feats_c = rearrange(self.pos_encoding(feats_c), 'n c h w -> n (h w) c')
        feats_c1 = rearrange(self.pos_encoding(feats_c1), 'n c h w -> n (h w) c')

        feats_c, feats_c1 = self.loftr_coarse(feats_c, feats_c1, None, None, detach_left=False, detach_right=False)

        feats_c = rearrange(feats_c, 'n (h w) c -> n c h w', h=h, w=w)
        feats_c1 = rearrange(feats_c1, 'n (h w) c -> n c h w', h=h, w=w)

        return feats_c, feats_c1