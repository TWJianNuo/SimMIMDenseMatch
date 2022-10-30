import argparse

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, DistributedSampler
from datasets.megadepth import MegadepthBuilder
from data.data_simmim import SimMIMTransform
from config import get_config
from models import build_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

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

model = build_model(config, is_pretrain=True)
model.cuda()
model.eval()

checkpoint = torch.load('/home/shengjie/Documents/MultiFlow/SimMIMDenseMatch/checkpoints/simmim_pretrain__swin_base__img192_window6__800ep.pth', map_location='cpu')
msg = model.load_state_dict(checkpoint['model'], strict=True)

with torch.no_grad():
    for idx, (img, mask, _) in enumerate(dataloader):
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        loss, x_rec = model(img, mask)

        img_vls = img * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view([1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view([1, 3, 1, 1]).cuda().float()
        rec_vls = x_rec * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view([1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view([1, 3, 1, 1]).cuda().float()


        from tools.tools import tensor2rgb, tensor2disp
        tensor2rgb(img_vls).show()
        tensor2rgb(rec_vls).show()
        tensor2disp(F.interpolate(mask.unsqueeze(1).float(), [192, 192]), vmax=1, viewind=0).show()
        a = 1