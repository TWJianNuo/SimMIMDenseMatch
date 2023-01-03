import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from datasets.imagenet import ImageNetDataset
from datasets.imagenet_aug import ImangeNetAug
from datasets.scannet import ScanNetBuilder
from data.data_simmim_scannet_twoview import SimMIMTransform
from config import get_config
from DKMResnetLoFTRPreTrainNoPVT.models.build_model import DKMv2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument('--data_root', type=str, required=True,)
parser.add_argument('--scannet_root', type=str, required=True,)
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
config['DATA']['MASK_RATIO'] = 0.75
config['DATA']['MASK_RATIO_SCANNET'] = 0.9
config['MODEL']['SWIN']['PATCH_SIZE'] = 2
config['MODEL']['VIT']['PATCH_SIZE'] = 2
config['DATA']['IMG_SIZE'] = (160, 192)
config.freeze()
transform = SimMIMTransform(config)

imagenet = ImageNetDataset(data_root=args.data_root, split='val', transform=transform)

scannet = ScanNetBuilder(data_root=args.scannet_root, progress_bar=False, debug=True)
scannet = scannet.build_scenes(split="train", transform=transform)
scannet = ConcatDataset(scannet)

dataloader_imagenet = DataLoader(imagenet, 12, num_workers=0, pin_memory=True, drop_last=True)
dataloader_scannet = DataLoader(scannet, 12, num_workers=0, pin_memory=True, drop_last=True)

imagenetaug = ImangeNetAug(data_root=args.data_root, auscale=1.0, ht=config.DATA.IMG_SIZE[0], wd=config.DATA.IMG_SIZE[1],
                           homography_augmentation=True, transform=transform)
dataloader_imagenetaug = DataLoader(imagenetaug, 12, num_workers=0, pin_memory=True, drop_last=True)
model = DKMv2(pvt_depth=4, resolution='extrasmall')
model.cuda()
model.eval()

# torch.save(model.state_dict(), '/home/shengjie/Documents/supporting_projects/EMAwareFlow/checkpoints/MIMAug/model.ckpt')
with torch.no_grad():
    for idx, (imagenet_batch, scannet_batch, imagenetaug_batch) in enumerate(zip(dataloader_imagenet, dataloader_scannet, dataloader_imagenetaug)):
        img_imagenet, mask_imagenet = imagenet_batch
        img1_scannetnet, mask_scannet, img2_scannetnet, _ = scannet_batch
        img1_imagenet, mask1_imagenet, img2_imagenet, _, masksup1_imagenet, _ = imagenetaug_batch
        imagenetaug_batch

        img_imagenet = img_imagenet.cuda(non_blocking=True)
        mask_imagenet = mask_imagenet.cuda(non_blocking=True)

        img1_scannetnet = img1_scannetnet.cuda(non_blocking=True)
        img2_scannetnet = img2_scannetnet.cuda(non_blocking=True)
        mask_scannet = mask_scannet.cuda(non_blocking=True)

        img1_imagenet = img1_imagenet.cuda(non_blocking=True)
        mask1_imagenet = mask1_imagenet.cuda(non_blocking=True)
        img2_imagenet = img2_imagenet.cuda(non_blocking=True)
        masksup1_imagenet = masksup1_imagenet.cuda(non_blocking=True)

        # loss, x_rec = model(img_imagenet, mask_imagenet)
        loss, x_rec = model(img1_imagenet, mask1_imagenet, img2_imagenet, masksup1_imagenet)

        # img_vls = img * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view([1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view([1, 3, 1, 1]).cuda().float()
        # rec_vls = x_rec * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view([1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view([1, 3, 1, 1]).cuda().float()
        img_vls = img1_scannetnet * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view(
            [1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view(
            [1, 3, 1, 1]).cuda().float()

        from tools.tools import tensor2rgb, tensor2disp
        tensor2rgb(img_vls).show()
        tensor2rgb(rec_vls).show()
        tensor2disp(F.interpolate(mask.unsqueeze(1).float(), [192, 192]), vmax=1, viewind=0).show()
        a = 1