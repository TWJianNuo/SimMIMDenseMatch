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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from config import get_config
from tools.tools import tensor2rgb, tensor2disp
from utils import get_grad_norm

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

@torch.no_grad()
def main(config):
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
        _, x_rec = model(img1_scannet, mask_scannet, img2_scannet)

        img1_vls = img1_scannet * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view(
            [1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view(
            [1, 3, 1, 1]).cuda().float()
        img2_vls = img2_scannet * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view(
            [1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view(
            [1, 3, 1, 1]).cuda().float()
        rec_vls = x_rec * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view(
            [1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view(
            [1, 3, 1, 1]).cuda().float()

        b, _, h, w = img1_scannet.shape

        vls1 = tensor2rgb(img1_vls)
        vls4 = tensor2rgb(img2_vls)
        vls2 = tensor2rgb(rec_vls)
        vls3 = tensor2disp(F.interpolate(mask_scannet.unsqueeze(1).float(), [h, w]), vmax=1, viewind=0)
        vls = np.concatenate([vls1, vls4, vls2, vls3], axis=0)

        Image.fromarray(vls).save(os.path.join('/media/shengjie/disk1/visualization/EMAwareFlow/pmim_ablate', '{}.png'.format(str(x))))


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