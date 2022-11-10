from __future__ import print_function, division
import os, sys, inspect, time
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_root)

import tqdm
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from datasets.megadepth_pair import MegadepthBuilder
from data.data_simmim_mega import SimMIMTransform
from config import get_config
from dkm_loftr_PVT_noprj_new.models.build_model import DKMv2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tools.tools import tensor2rgb, tensor2disp

import PIL.Image as Image

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def unnormrgb(rgb):
    mean = torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view([1, 3, 1, 1]).cuda().float()
    std = torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view([1, 3, 1, 1]).cuda().float()

    rgb = rgb * std + mean
    return rgb

parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument('--data_root', type=str, required=True,)
parser.add_argument('--local_rank', type=int, default=0, )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)

args = parser.parse_args()

config = get_config(args)
config.defrost()
config.DATA.IMG_SIZE = (192, 256)
config.MODEL.TYPE = 'pvt_medium'
config.DATA.MASK_RATIO = 0.75
config.freeze()

transform = SimMIMTransform(config)

h, w = 384, 512
mega = MegadepthBuilder(data_root=args.data_root)
megadepth_train1 = mega.build_scenes(
    split="test_loftr", min_overlap=0.01, ht=h, wt=w, shake_t=32, transform=transform
)
megadepth_train2 = mega.build_scenes(
    split="test_loftr", min_overlap=0.35, ht=h, wt=w, shake_t=32, transform=transform
)

megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)

dataloader = DataLoader(megadepth_train, 12, num_workers=0, pin_memory=True, drop_last=True)

model = DKMv2(config, version="outdoor", outputfeature='concatenated', h=config.DATA.IMG_SIZE[0], w=config.DATA.IMG_SIZE[1])
model.cuda()
model.train()

vls_root = '/media/shengjie/disk1/visualization/EMAwareFlow/dense_match/pretext'
os.makedirs(vls_root, exist_ok=True)


ckpt_path = '/home/shengjie/Documents/MultiFlow/SimMIMDenseMatch/checkpoints/simmim_pretrain/mega_twoview_fix075/ckpt_epoch_45.pth'
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['model'], strict=True)

# for p in model.parameters():
#     assert (p.min() > -1e3)

np.random.seed(2022)
vls_idx = np.random.randint(0, len(megadepth_train), 1000)
print(len(megadepth_train), vls_idx)

darkratio1 = 0.0
darkratio2 = 1.0
with torch.no_grad():
    for idx in tqdm.tqdm(vls_idx):
        img1, mask1, img2, mask2 = megadepth_train.__getitem__(idx)

        mask1 = torch.from_numpy(mask1).int()
        mask2 = torch.from_numpy(mask2).int()

        img1 = img1.unsqueeze(0)
        mask1 = mask1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        mask2 = mask2.unsqueeze(0)

        img1 = img1.cuda(non_blocking=True)
        mask1 = mask1.cuda(non_blocking=True)

        img2 = img2.cuda(non_blocking=True)
        mask2 = mask2.cuda(non_blocking=True)

        rgb1_recons, rgb2_recons, losses, totloss = model(img1, mask1, img2, mask2)

        mask1 = F.interpolate(mask1.unsqueeze(1).float(), config.DATA.IMG_SIZE, mode='nearest')
        mask2 = F.interpolate(mask2.unsqueeze(1).float(), config.DATA.IMG_SIZE, mode='nearest')

        img1vls = unnormrgb(img1)
        img2vls = unnormrgb(img2)
        img1vls = img1vls * (1 - mask1) + img1vls * mask1 * darkratio1
        img2vls = img2vls * (1 - mask2) + img2vls * mask2 * darkratio1

        img1vls = tensor2rgb(img1vls)
        img2vls = tensor2rgb(img2vls)

        reconvls1 = unnormrgb(rgb1_recons[4])
        reconvls2 = unnormrgb(rgb2_recons[4])

        reconvls1 = reconvls1 * mask1 + reconvls1 * (1 - mask1) * darkratio2
        reconvls2 = reconvls2 * mask2 + reconvls2 * (1 - mask2) * darkratio2

        reconvls1 = tensor2rgb(reconvls1)
        reconvls2 = tensor2rgb(reconvls2)

        vlscomb1 = np.concatenate([np.array(img1vls), np.array(img2vls)], axis=1)
        vlscomb2 = np.concatenate([np.array(reconvls1), np.array(reconvls2)], axis=1)
        vlscom = np.concatenate([np.array(vlscomb1), np.array(vlscomb2)], axis=0)

        Image.fromarray(vlscom).save(os.path.join(vls_root, "{}.jpg".format(str(idx).zfill(10))))
        a = 1