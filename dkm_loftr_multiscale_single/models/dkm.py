import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from tools.tools import get_tuple_transform_ops, RGB2Gray
from dkm_loftr_multiscale_single.loftr.loftr import LoFTR


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb=None,
        displacement_emb_dim=None,
        encoder_stride=None
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=encoder_stride ** 2 * 3,
                kernel_size=1),
            nn.PixelShuffle(encoder_stride),
        )

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
        )
        norm = nn.BatchNorm2d(out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, x):
        """Computes the relative refining displacement in pixels for a given image x,y and a coarse flow-field between them

        Args:
            x ([type]): [description]
            y ([type]): [description]
            flow ([type]): [description]

        Returns:
            [type]: [description]
        """
        d = self.block1(x)
        d = self.hidden_blocks(d)
        d = self.out_conv(d)
        return d

class Decoder(nn.Module):
    def __init__(
        self, conv_refiner, scales="all"
    ):
        super().__init__()
        self.conv_refiner = conv_refiner
        if scales == "all":
            self.scales = ["32", "16", "8", "4", "2", "1"]
        else:
            self.scales = scales

    def forward(self, img, mask, f_pyramid):
        in_chans = 3.0

        patch_sizeh = int(img.shape[2] / mask.shape[1])
        patch_sizew = int(img.shape[3] / mask.shape[2])
        assert patch_sizeh == patch_sizew

        mask = mask.repeat_interleave(patch_sizeh, 1).repeat_interleave(patch_sizew, 2).unsqueeze(1).contiguous()

        rgb_recons = dict()
        losses = dict()
        totloss = 0
        for new_scale in self.scales:
            ins = int(new_scale)
            feature = f_pyramid[ins]

            rgb_recon = self.conv_refiner[new_scale](feature)
            rgb_recons[ins] = rgb_recon

            loss_recon = F.l1_loss(img, rgb_recon, reduction='none')
            loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / in_chans
            losses[new_scale] = loss

            totloss += loss
        totloss = totloss / len(self.scales)
        return rgb_recons, losses, totloss


class RegressionMatcher(nn.Module):
    def __init__(
        self,
        loftr,
        decoder,
    ):
        super().__init__()
        self.loftr = loftr
        self.decoder = decoder

    def forward(self, img, mask):
        f_pyramid = self.loftr(img, mask)
        rgb_recons, losses, totloss = self.decoder(img, mask, f_pyramid)
        return rgb_recons, losses, totloss