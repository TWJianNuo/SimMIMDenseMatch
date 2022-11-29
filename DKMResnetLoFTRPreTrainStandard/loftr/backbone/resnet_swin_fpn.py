import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from functools import partial
from einops.einops import rearrange
from .simmim_rect import SimMIMPVT
from loguru import logger

from torchvision.models import resnet as tv_resnet

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

    def __init__(self, config):
        super().__init__()

        # Config FPN
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']
        block_dims[2] = 320

        self.prj3 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=block_dims[2], kernel_size=1, padding=0))

        # Init Networks
        if config['pretrained_resnet']:
            logger.info("Loaded from Pretrained ResNet50")
        else:
            logger.info("Loaded from Random ResNet50")

        # encoder = tv_resnet.resnet50(pretrained=config['pretrained_resnet'])
        encoder = tv_resnet.resnet50(pretrained=False)

        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu

        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3

        self.pvt_depth = config['pvt_depth']
        # Configurate PVT Transformer
        encoder = SimMIMPVT(
            img_size=config['IMG_SIZE'],
            patch_size=8,
            in_chans=3,
            embed_dims=[320],
            depths=[self.pvt_depth],
            num_heads=[5],
            mlp_ratios=[4],
            sr_ratios=[2],
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        self.encoder = encoder

        self.img_size = config['IMG_SIZE']
        logger.info("PVT_DEPTH: %d, IMGH: %d, IMGW: %d" % (self.pvt_depth, self.img_size[0], self.img_size[1]))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 256))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mimmask):
        # ResNet Backbone
        B, _, rawh, raww = x.shape

        # Scale 1
        x1 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x1)  # 1/2

        # Apply MIM Masking
        w = mimmask.unsqueeze(1).type_as(x1)
        mask_tokens = self.mask_token.view([1, -1, 1, 1]).expand(B, -1, x1.shape[2], x1.shape[3])
        x1 = x1 * (1. - w) + mask_tokens * w

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x3 = self.prj3(x3)

        # add position embedding
        spawnh, spawnw = self.img_size
        posembedh, posembedw = int(spawnh / 8), int(spawnw / 8)
        pos_embed = rearrange(self.encoder.pos_embed, 'b (h w) c -> b c h w', h=posembedh, w=posembedw)
        pos_embed = torch.nn.functional.interpolate(pos_embed, (int(x3.shape[2]), int(x3.shape[3])), mode='bilinear', align_corners=True)
        pos_embed = rearrange(pos_embed, 'b c h w -> b (h w) c')

        x3 = x3.flatten(2).transpose(1, 2)
        x3 = x3 + pos_embed

        # apply Transformer blocks
        H = int(rawh / self.encoder.patch_size)
        W = int(raww / self.encoder.patch_size)

        for blk in self.encoder.blocks:
            x3, (H, W) = blk(x3, H, W)
        x3 = self.encoder.norm(x3)
        x3_spatial = x3.transpose(1, 2)
        B, C, L = x3_spatial.shape
        x3_spatial = x3_spatial.reshape(B, C, H, W)

        return x3_spatial