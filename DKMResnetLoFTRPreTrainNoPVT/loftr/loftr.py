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

        self.out_conv_pair = nn.Sequential(
            nn.Conv2d(
                in_channels=320,
                out_channels=8 ** 2 * 3,
                kernel_size=1),
            nn.PixelShuffle(8),
        )

    def forward(self, img, mask, img2=None, masksup=None):
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

        if masksup is None:
            masksup = torch.ones_like(mask).squeeze(1)

        in_chans = 3
        loss_recon = F.l1_loss(img, rgb_recon, reduction='none')

        mask_pos = mask * masksup.unsqueeze(1)

        # from tools.tools import tensor2rgb, tensor2disp
        # tensor2disp(mask, viewind=0, vmax=1).show()
        # tensor2disp(masksup.unsqueeze(1), viewind=0, vmax=1).show()

        loss = (loss_recon * mask_pos).sum() / (mask_pos.sum() + 1e-5) / in_chans

        mask_neg = (1 - mask)
        loss2 = (loss_recon * mask_neg).sum() / (mask_neg.sum() + 1e-5) / in_chans
        loss = loss + loss2 * 0.05
        return loss, rgb_recon
