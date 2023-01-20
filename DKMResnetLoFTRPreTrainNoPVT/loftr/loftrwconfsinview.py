import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer

class LoFTRWConfSinview(nn.Module):
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

        self.out_conv_pair_refiner = nn.Sequential(
            nn.Conv2d(
                in_channels=640,
                out_channels=8 ** 2 * (3 + 1),
                kernel_size=1),
            nn.PixelShuffle(8),
        )

        self.temperature = 0.1
        self.sigmoid = torch.nn.Sigmoid()

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
        # Change to Evaluation Mode for Batch Statistics
        mode = self.training
        self.backbone.train()
        feats_c = self.backbone(img, mask, maskedin=True)
        self.backbone.train(mode)

        feats_c1 = self.backbone(img2, torch.zeros_like(mask), maskedin=False)
        n, c, h, w = feats_c.shape
        feats_c = rearrange(self.pos_encoding(feats_c), 'n c h w -> n (h w) c')
        feats_c1 = rearrange(self.pos_encoding(feats_c1), 'n c h w -> n (h w) c')

        feats_c, feats_c1 = self.loftr_coarse(feats_c, feats_c1)
        feats_c = rearrange(feats_c, 'n (h w) c -> n c h w', h=h, w=w)
        feats_c1 = rearrange(feats_c1, 'n (h w) c -> n c h w', h=h, w=w)

        rgb_recon = self.out_conv_pair(feats_c)

        patch_sizeh = int(img.shape[2] / mask.shape[1])
        patch_sizew = int(img.shape[3] / mask.shape[2])
        assert patch_sizeh == patch_sizew
        mask = mask.repeat_interleave(patch_sizeh, 1).repeat_interleave(patch_sizew, 2).unsqueeze(1).contiguous()

        if masksup is None:
            masksup = torch.ones_like(mask).squeeze(1)

        in_chans = 3
        loss_recon = F.l1_loss(img, rgb_recon, reduction='none')

        mask_pos = mask * masksup.unsqueeze(1)

        loss = (loss_recon * mask_pos).sum() / (mask_pos.sum() + 1e-5) / in_chans

        mask_neg = (1 - mask)
        loss2 = (loss_recon * mask_neg).sum() / (mask_neg.sum() + 1e-5) / in_chans
        loss = loss + loss2 * 0.05

        # Additional Stuff
        bz, ch, h, w = feats_c.shape
        feats_c = rearrange(feats_c, 'n c h w -> n (h w) c')
        feats_c1 = rearrange(feats_c1, 'n c h w -> n (h w) c')
        sim_matrix = torch.einsum("nlc,nsc->nls", feats_c, feats_c1) / self.temperature
        A = torch.softmax(sim_matrix, dim=2)
        if torch.sum(torch.isnan(A)) > 0:
            A = torch.softmax(sim_matrix.type(torch.FloatTensor), dim=2)
            A = A.type_as(feats_c)

        feats_c1 = rearrange(feats_c1, 'n (h w) c -> n c h w', h=h, w=w)
        f2_scale8_aligned = A.view([bz, 1, h*w, h*w]) @ feats_c1.view([bz, ch, h*w, 1])
        f2_scale8_aligned = f2_scale8_aligned.view([bz, ch, h, w])
        f2_scale8_aligned = rearrange(f2_scale8_aligned, 'n c h w -> n (h w) c')

        feats_c = rearrange(feats_c, 'n (h w) c -> n c h w', h=h, w=w, c=c)
        f2_scale8_aligned = rearrange(f2_scale8_aligned, 'n (h w) c -> n c h w', h=h, w=w, c=c)

        catted = torch.cat([feats_c, f2_scale8_aligned], dim=1)
        d = self.out_conv_pair_refiner(catted)
        reliability, delta_rgb = d[:, :-3], d[:, -3:]
        reliability = self.sigmoid(reliability)

        rgb_recon_refined = rgb_recon.detach() * (1 - reliability) + delta_rgb * reliability
        loss3 = (F.l1_loss(img, rgb_recon_refined, reduction='none') * mask_pos).sum() / (mask_pos.sum() + 1e-5) / in_chans

        loss = loss + loss3 * 0.25
        return loss, rgb_recon, rgb_recon_refined, reliability