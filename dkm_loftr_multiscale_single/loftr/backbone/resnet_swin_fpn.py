import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial

from dkm_loftr_multiscale_single.models.simmim_rect import SimMIMPVT

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class ResNetSwinFPN(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config_resfpn, config_simmim):
        super().__init__()

        # Config FPN
        block = BasicBlock
        initial_dim = config_resfpn['initial_dim']
        block_dims = config_resfpn['block_dims']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(3, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 3. FPN upsample
        block_dims[2] = 320
        block_dims[1] = 128
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Configurate PVT Transformer
        encoder = SimMIMPVT(
            img_size=config_simmim.DATA.IMG_SIZE,
            patch_size=4,
            in_chans=3,
            embed_dims=[128, 320],
            depths=[4, 8],
            num_heads=[2, 5],
            mlp_ratios=[8, 4],
            sr_ratios=[4, 2],
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        self.encoder = encoder

        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder.embed_dims[0]))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, mask):
        # ResNet Backbone
        B, _, rawh, raww = x.shape

        # Scale 1
        x1 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x1)  # 1/2

        x2 = self.maxpool(x1)
        # Apply Mask
        x2 = x2.flatten(2).transpose(1, 2)
        assert mask is not None
        B, L, _ = x2.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x2 = x2 * (1. - w) + mask_tokens * w

        # add position embedding
        x2 = x2 + self.encoder.pos_embed

        # apply Transformer blocks
        H = int(rawh / self.encoder.patch_size)
        W = int(raww / self.encoder.patch_size)
        for blk in self.encoder.blocks[0:4]:
            x2, (H, W) = blk(x2, H, W)
        x2_out = x2.transpose(1, 2)
        B, C, L = x2_out.shape
        x2_out = x2_out.reshape(B, C, H, W)

        x3, (H, W) = self.encoder.blocks[4](x2, H, W)
        for blk in self.encoder.blocks[5:12]:
            x3, (H, W) = blk(x3, H, W)
        x3 = self.encoder.norm(x3)
        x3_out = x3.transpose(1, 2)
        B, C, L = x3_out.shape
        x3_out = x3_out.reshape(B, C, H, W)

        # FPN
        x3_out = self.layer3_outconv(x3_out)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2_out)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        return [x3_out, x2_out]


