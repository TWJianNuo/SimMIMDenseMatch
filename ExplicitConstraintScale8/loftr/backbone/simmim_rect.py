# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio

        self.sr = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_ = F.avg_pool2d(x_, self.sr_ratio)
        x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class PVTBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        self.downsample = downsample

    def forward(self, x, H, W):
        if self.downsample:
            x, (H, W) = self.downsample(x, H, W)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, (H, W)


class PatchMerge(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, H, W):
        N, L, C = x.shape
        assert L == H * W
        x = x.permute(0, 2, 1).reshape(N, C, H, W)
        # FIXME look at relaxing size constraints
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x):
        raise NotImplementedError()


class SimMIMPVT(nn.Module):
    """ SimMIM with Pyramid Vision Transformer backbone
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, stride=16,
                 embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3], num_heads=[1, 2, 5, 8],
                 mlp_ratios=[8, 8, 4, 4], sr_ratios=[8, 4, 2, 1],
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.embed_dims = embed_dims
        self.stride = stride
        self.kernel_stride = stride // patch_size
        self.in_chans = 3

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size

        self.embed_h, self.embed_w = self.patch_embed.patches_resolution
        self.patches_resolution = self.patch_embed.patches_resolution
        self.num_layers = len(depths)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]), requires_grad=False)  # fixed sin-cos embedding
        self.kernel = torch.ones(embed_dims[0], 1, 2, 2)

        self.blocks = nn.ModuleList()
        for i_layer in range(self.num_layers):
            for dep in range(depths[i_layer]):
                downsample_flag = (i_layer > 0) and (dep == 0)
                layer = PVTBlock(dim=embed_dims[i_layer],
                                 num_heads=num_heads[i_layer],
                                 sr_ratio=sr_ratios[i_layer],
                                 mlp_ratio=mlp_ratios[i_layer],
                                 qkv_bias=True, qk_scale=None,
                                 drop_path=0.,
                                 downsample=PatchMerge(
                                     patch_size=2,
                                     in_chans=embed_dims[i_layer - 1],
                                     embed_dim=embed_dims[i_layer]
                                 ) if downsample_flag else None
                                 )
                self.blocks.append(layer)
        self.norm = norm_layer(embed_dims[-1])
        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.patches_resolution[0]), int(self.patch_embed.patches_resolution[1]), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def unpatchify(self, x, stride=16):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = stride
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def patchify(self, imgs, stride=16):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = stride
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

class PVTTransformerForSimMIM(SimMIMPVT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims[0]))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        B, _, rawh, raww = x.shape
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        # add position embedding
        x = x + self.pos_embed

        # apply Transformer blocks
        H = int(rawh / self.patch_size)
        W = int(raww / self.patch_size)
        for blk in self.blocks:
            # print(x.shape)
            x, (H, W) = blk(x, H, W)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        x = x.reshape(B, C, H, W)
        return x


class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.embed_dims[-1],
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss, x_rec

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(config):
    model_type = config.MODEL.TYPE
    if model_type == 'pvt_small':
        encoder = PVTTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=4,
            in_chans=3,
            embed_dims=[64, 128, 320, 512],
            depths=[3, 4, 6, 3],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            sr_ratios=[8, 4, 2, 1],
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        encoder_stride = 32
        print("============================")
        print("Construct Pvit Small")
        print("============================")
    elif model_type == 'pvt_medium':
        encoder = PVTTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=4,
            in_chans=3,
            embed_dims=[64, 128, 320, 512],
            depths=[3, 4, 18, 3],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            sr_ratios=[8, 4, 2, 1],
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        encoder_stride = 32
        print("============================")
        print("Construct Pvit Medium")
        print("============================")
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")
    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)
    return model
