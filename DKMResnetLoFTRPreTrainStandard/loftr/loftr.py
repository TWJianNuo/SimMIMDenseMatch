import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer

class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=320,
                out_channels=8 ** 2 * 3,
                kernel_size=1),
            nn.PixelShuffle(8),
        )


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
        feats_c = self.backbone(img, mask)
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
