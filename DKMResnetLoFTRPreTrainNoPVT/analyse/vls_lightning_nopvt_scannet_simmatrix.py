# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------
from __future__ import print_function, division
import os, sys, inspect, time, tqdm
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_root)

import argparse
import datetime
import numpy as np
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from config import get_config
from tools.tools import tensor2rgb, tensor2disp
from utils import get_grad_norm

from einops.einops import rearrange
from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from DKMResnetLoFTRPreTrainNoPVT.models.build_model import DKMv2
from DKMResnetLoFTRPreTrainNoPVT.analyse.scannet_vls import build_loader_scannet

def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--data-path-scannet', type=str, help='path to dataset scannet')
    parser.add_argument('--minoverlap-scannet', type=float, default=0.0)
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--adjustscaler', type=float, default=1.0)
    parser.add_argument('--usefullattention', action='store_true')

    parser.add_argument('--mask-patch-size', type=int, default=32)
    parser.add_argument('--mask-ratio-scannet', type=float, default=0.85)

    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def vls_conf(vls1, vsl2, feat_src, feat_sup, temperature=0.1, seed=0):
    np.random.seed(seed)
    # Coarse Alignment
    feat_src = rearrange(feat_src, 'n c h w -> n (h w) c')
    feat_sup = rearrange(feat_sup, 'n c h w -> n (h w) c')
    feat_src, feat_sup = map(lambda feat: feat / feat.shape[-1] ** .5, [feat_src, feat_sup])
    sim_matrix = torch.einsum("nlc,nsc->nls", feat_src, feat_sup) / temperature
    conf_matrix = torch.softmax(sim_matrix, dim=2)

    bz, d16h, d16w = conf_matrix.shape[0], 20, 24
    conf_matrix = conf_matrix.view([bz, d16h, d16w, d16h, d16w])
    rndsh, rndsw, rndsph, rndspw = np.random.randint(0, d16h), np.random.randint(0, d16w), np.random.randint(0, d16h), np.random.randint(0, d16w)
    conf_matrix_sampled = conf_matrix[0, rndsh, rndsw]
    dummy_source = torch.zeros_like(conf_matrix_sampled)
    dummy_source[rndsh, rndsw] = 1

    conf_matrix_sampled = F.interpolate(conf_matrix_sampled.unsqueeze(0).unsqueeze(0), [160, 192], mode='bilinear', align_corners=True)
    dummy_source = F.interpolate(dummy_source.unsqueeze(0).unsqueeze(0), [160, 192], mode='bilinear', align_corners=True)
    conf_matrix_sampled = tensor2disp(conf_matrix_sampled, vmax=0.2, viewind=0)
    dummy_source = tensor2disp(dummy_source, vmax=0.2, viewind=0)

    combuned1 = Image.fromarray((np.array(vls1) * 0.3 + np.array(dummy_source) * 0.7).astype(np.uint8))
    combuned2 = Image.fromarray((np.array(vsl2) * 0.3 + np.array(conf_matrix_sampled) * 0.7).astype(np.uint8))

    combined = np.concatenate([np.array(vls1), np.array(vsl2), np.array(combuned1), np.array(combuned2)], axis=0)
    return combined


class GP(nn.Module):
    def __init__(
        self,
        kernel,
        T=1,
        learn_temperature=False,
        sigma_noise=0.1,
        softmaxT=1
    ):
        super().__init__()
        self.K = kernel(T=T, learn_temperature=learn_temperature)
        self.sigma_noise = sigma_noise
        self.softmaxT = softmaxT

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def forward(self, x, y):
        b, c, h1, w1 = x.shape
        b, c, h2, w2 = y.shape

        x, y = self.reshape(x), self.reshape(y)
        K_yy = self.K(y, y)
        K_xy = self.K(x, y)
        sigma_noise = self.sigma_noise * torch.eye(h2 * w2, device=x.device)[None, :, :]
        # Due to https://github.com/pytorch/pytorch/issues/16963 annoying warnings, remove batch if N large
        if len(K_yy[0]) > 2000:
            K_yy_inv = torch.cat([torch.linalg.inv(K_yy[k:k+1] + sigma_noise[k:k+1]) for k in range(b)])
        else:
            K_yy_inv = torch.linalg.inv(K_yy + sigma_noise)

        simmatrix_source = K_xy.matmul(K_yy_inv)

        return simmatrix_source


class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T=0.2, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K

def vls_gpconf(vls1, vsl2, feat_src, feat_sup, temperature=0.2, seed=0):
    gp = GP(kernel=CosKernel, T=temperature)
    np.random.seed(seed)

    # Coarse Alignment
    conf_matrix = gp(feat_src, feat_sup)

    bz, d16h, d16w = conf_matrix.shape[0], 20, 24
    conf_matrix = conf_matrix.view([bz, d16h, d16w, d16h, d16w])
    rndsh, rndsw, rndsph, rndspw = np.random.randint(0, d16h), np.random.randint(0, d16w), np.random.randint(0, d16h), np.random.randint(0, d16w)
    conf_matrix_sampled = conf_matrix[0, rndsh, rndsw]
    dummy_source = torch.zeros_like(conf_matrix_sampled)
    dummy_source[rndsh, rndsw] = 1

    conf_matrix_sampled = F.interpolate(conf_matrix_sampled.unsqueeze(0).unsqueeze(0), [160, 192], mode='bilinear', align_corners=True)
    dummy_source = F.interpolate(dummy_source.unsqueeze(0).unsqueeze(0), [160, 192], mode='bilinear', align_corners=True)
    conf_matrix_sampled = tensor2disp(conf_matrix_sampled, vmax=0.2, viewind=0)
    dummy_source = tensor2disp(dummy_source, vmax=0.2, viewind=0)

    combuned1 = Image.fromarray((np.array(vls1) * 0.3 + np.array(dummy_source) * 0.7).astype(np.uint8))
    combuned2 = Image.fromarray((np.array(vsl2) * 0.3 + np.array(conf_matrix_sampled) * 0.7).astype(np.uint8))

    combined = np.concatenate([np.array(vls1), np.array(vsl2), np.array(combuned1), np.array(combuned2)], axis=0)
    return combined

@torch.no_grad()
def main(config):
    temperature = 0.3
    gptemperature = 0.2

    scannet = build_loader_scannet(config)
    model = DKMv2()
    model.cuda()

    data_loader_train_scannet = DataLoader(scannet, 1, num_workers=0, pin_memory=True, drop_last=True, shuffle=False)
    data_loader_train_scannet = iter(data_loader_train_scannet)

    ckpt_path = '/home/shengjie/Documents/MultiFlow/SimMIMDenseMatch/checkpoints/simmim_pretrain/AblatePretrain/lightning_nopvt_scannet_pz4_p099_dm/ckpt_epoch_40.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    incompactible = model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    scannet_batch = next(data_loader_train_scannet)

    mask_scannet = scannet_batch['mask']
    mask_scannet = mask_scannet.cuda(non_blocking=True)
    img1_scannet = scannet_batch[0]
    img1_scannet = img1_scannet.cuda(non_blocking=True)

    img_keys = list(scannet_batch.keys())[0:-1]
    for x in tqdm.tqdm(img_keys):
        img2_scannet = scannet_batch[x]
        img2_scannet = img2_scannet.cuda(non_blocking=True)
        feat_src, feat_sup = model.extrac_feature(img1_scannet, img2_scannet)

        # Visualize the Sim Matrix
        img1_vls = img1_scannet * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view(
            [1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view(
            [1, 3, 1, 1]).cuda().float()
        img2_vls = img2_scannet * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view(
            [1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view(
            [1, 3, 1, 1]).cuda().float()

        b, _, h, w = img1_scannet.shape

        vls1 = tensor2rgb(img1_vls)
        vls4 = tensor2rgb(img2_vls)
        vls3 = tensor2disp(F.interpolate(mask_scannet.unsqueeze(1).float(), [h, w]), vmax=1, viewind=0)
        vls = np.concatenate([vls1, vls4, vls3], axis=0)

        vls_fold = os.path.join('/media/shengjie/disk1/visualization/EMAwareFlow/pmim_ablate_confvls', '{}'.format(str(x)))
        os.makedirs(vls_fold, exist_ok=True)

        Image.fromarray(vls).save(os.path.join(vls_fold, 'overview.jpg'))
        for _ in range(30):
            # Random Visualziation of the Confidence Matrix
            combined = vls_conf(vls1, vls4, feat_src, feat_sup, temperature=temperature, seed=_)
            gpcombined = vls_gpconf(vls1, vls4, feat_src, feat_sup, temperature=gptemperature, seed=_)
            vls = np.concatenate([np.array(combined), np.array(gpcombined)], axis=1)
            Image.fromarray(vls).save(os.path.join(vls_fold, '{}.png'.format(str(_))))

if __name__ == '__main__':
    args, config = parse_option()

    config.defrost()
    config.DATA.IMG_SIZE = (160, 192)  # 192
    config['DATA']['MASK_RATIO'] = 0.75
    config['DATA']['DATA_PATH_SCANNET'] = args.data_path_scannet
    config['DATA']['MINOVERLAP_SCANNET'] = args.minoverlap_scannet
    config['MODEL']['SWIN']['PATCH_SIZE'] = 2
    config['MODEL']['VIT']['PATCH_SIZE'] = 2

    config['DATA']['MASK_RATIO_SCANNET'] = args.mask_ratio_scannet
    config['DATA']['MASK_PATCH_SIZE'] = args.mask_patch_size
    config.freeze()
    main(config)