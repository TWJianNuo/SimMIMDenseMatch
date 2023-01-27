import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from functools import partial
from einops.einops import rearrange
from .simmim_rect import SimMIMPVT
from loguru import logger

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, twobn=False):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)

        self.twobn = twobn
        if self.twobn:
            self.bn1_mask = nn.BatchNorm2d(planes)
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2_mask = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
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
        if self.twobn:
            maskedin = x['maskedin']
            x = x['x']

            y = x

            if maskedin:
                y = self.relu(self.bn1_mask(self.conv1(y)))
                y = self.bn2_mask(self.conv2(y))
            else:
                y = self.relu(self.bn1(self.conv1(y)))
                y = self.bn2(self.conv2(y))

            if self.downsample is not None:
                x = self.downsample(x)

            return {'x': self.relu(x+y), 'maskedin': maskedin}

        else:
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

        if 'two_bn' in config:
            self.twobn = config['two_bn']
        else:
            self.twobn = False

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(3, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)

        if self.twobn:
            self.bn1_mask = nn.BatchNorm2d(initial_dim)
            self.bn1 = nn.BatchNorm2d(initial_dim)
        else:
            self.bn1 = nn.BatchNorm2d(initial_dim)

        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.img_size = config['IMG_SIZE']

        logger.info("IMGH: %d, IMGW: %d" % (self.img_size[0], self.img_size[1]))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 128))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride, twobn=self.twobn)
        layer2 = block(dim, dim, stride=1, twobn=self.twobn)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, mimmask, maskedin=False):
        # ResNet Backbone
        B, _, rawh, raww = x.shape

        if self.twobn:
            if maskedin:
                x1 = self.relu(self.bn1_mask(self.conv1(x)))
            else:
                x1 = self.relu(self.bn1(self.conv1(x)))
            x1 = self.layer1({'x': x1, 'maskedin': maskedin})['x']  # 1/2

            # Apply MIM Masking
            w = mimmask.unsqueeze(1).type_as(x1)
            mask_tokens = self.mask_token.view([1, -1, 1, 1]).expand(B, -1, x1.shape[2], x1.shape[3])
            x1 = x1 * (1. - w) + mask_tokens * w

            x2 = self.layer2({'x': x1, 'maskedin': maskedin})['x']
            x3 = self.layer3({'x': x2, 'maskedin': maskedin})['x']
            x3_spatial = x3
        else:
            x1 = self.relu(self.bn1(self.conv1(x)))
            x1 = self.layer1(x1)  # 1/2

            # Apply MIM Masking
            w = mimmask.unsqueeze(1).type_as(x1)
            mask_tokens = self.mask_token.view([1, -1, 1, 1]).expand(B, -1, x1.shape[2], x1.shape[3])
            x1 = x1 * (1. - w) + mask_tokens * w

            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x3_spatial = x3

        return x3_spatial