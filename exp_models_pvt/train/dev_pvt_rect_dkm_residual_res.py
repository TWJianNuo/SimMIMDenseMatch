from __future__ import print_function, division
import os, sys, inspect, time
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_root)

import argparse
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from datasets.megadepth import MegadepthBuilder
from data.data_simmim_mega import SimMIMTransform
from config import get_config
from dkm_loftr_multiscale_single_residual_res.models.build_model import DKMv2


try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

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
config.freeze()

transform = SimMIMTransform(config)

h, w = 384, 512
mega = MegadepthBuilder(data_root=args.data_root)
megadepth_train1 = mega.build_scenes(
    split="train_loftr", min_overlap=0.01, ht=h, wt=w, shake_t=32, transform=transform
)
megadepth_train2 = mega.build_scenes(
    split="train_loftr", min_overlap=0.35, ht=h, wt=w, shake_t=32, transform=transform
)

megadepth_train = ConcatDataset(megadepth_train1 + megadepth_train2)
mega_ws = mega.weight_scenes(megadepth_train, alpha=0.75)

dataloader = DataLoader(megadepth_train, 12, num_workers=0, pin_memory=True, drop_last=True)

model = DKMv2(config, version="outdoor", outputfeature='concatenated')
model.cuda()
model.eval()

with torch.no_grad():
    for idx, (img, mask, _) in enumerate(dataloader):
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        rgb_recons, losses, totloss = model(img, mask)

        # img_vls = img * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view([1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view([1, 3, 1, 1]).cuda().float()
        # rec_vls = x_rec * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view([1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view([1, 3, 1, 1]).cuda().float()
        #
        # from tools.tools import tensor2rgb, tensor2disp
        # tensor2rgb(img_vls).show()
        # tensor2rgb(rec_vls).show()
        # tensor2disp(F.interpolate(mask.unsqueeze(1).float(), [192, 192]), vmax=1, viewind=0).show()
        a = 1