import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from functools import partial
from einops.einops import rearrange
from .simmim_rect import SimMIMPVT

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
        # print(block_dims)

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(3, initial_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
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

        print("PVT_DEPTH: %d, IMGH: %d, IMGW: %d" % (self.pvt_depth, self.img_size[0], self.img_size[1]))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, block_dims[0]))
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

        w = F.interpolate(mask.unsqueeze(1).type_as(x1), [int(rawh / 2), int(raww / 2)], mode='nearest')
        mask_tokens = self.mask_token.view([1, -1, 1, 1]).expand(B, -1, int(rawh / 2), int(raww / 2))

        # Apply Mask Tokens
        x1 = self.layer1(x1)  # 1/2
        x1 = x1 * (1. - w) + mask_tokens * w
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)


        # add position embedding
        spawnh, spawnw = self.img_size
        posembedh, posembedw = int(spawnh / 8), int(spawnw / 8)
        assert (rawh / spawnh) == (raww / spawnw)
        ratio = ((rawh / spawnh) + (raww / spawnw)) / 2
        pos_embed = rearrange(self.encoder.pos_embed, 'b (h w) c -> b c h w', h=posembedh, w=posembedw)
        pos_embed = torch.nn.functional.interpolate(pos_embed, (int(posembedh * ratio), int(posembedw * ratio)), mode='bilinear', align_corners=True)
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

        # FPN
        x3_out = self.layer3_outconv(x3_spatial)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        return [x3_out, x2_out]