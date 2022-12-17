import argparse

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, DistributedSampler
from datasets.imagenet import ImageNetDataset
from data.data_simmim_mega import SimMIMTransform
from config import get_config
from DKMResnetLoFTRPreTrainStandard.models.build_model import DKMv2
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
config.defrost()
config['DATA']['MASK_RATIO'] = 0.75
config['MODEL']['SWIN']['PATCH_SIZE'] = 2
config['MODEL']['VIT']['PATCH_SIZE'] = 2
config.freeze()
transform = SimMIMTransform(config)

imagenet = ImageNetDataset(data_root=args.data_root, split='val', transform=transform)
dataloader = DataLoader(imagenet, 12, num_workers=0, pin_memory=True, drop_last=True)

model = DKMv2(pvt_depth=4, resolution='extrasmall')
model.cuda()
model.eval()

torch.save(model.state_dict(), '/home/shengjie/Documents/supporting_projects/EMAwareFlow/checkpoints/MIMAug/model.ckpt')
with torch.no_grad():
    for idx, (img, mask) in enumerate(dataloader):
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