# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import ConcatDataset
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import to_2tuple

from datasets.scannet import ScanNetBuilder


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):

        self.input_size_h, self.input_size_w = to_2tuple(input_size)

        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert (self.input_size_h % self.mask_patch_size == 0) and (self.input_size_w % self.mask_patch_size == 0)
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size_h, self.rand_size_w = self.input_size_h // self.mask_patch_size, self.input_size_w // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size_h * self.rand_size_w
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

        self.pixel_unshuffle = torch.nn.PixelUnshuffle(self.mask_patch_size)

    def __call__(self, idx=None, validmask=None):
        if idx is not None:
            np.random.seed(idx)

        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        if validmask is not None:
            mask = mask.reshape((self.rand_size_h, self.rand_size_w))

            validmask_patch = self.pixel_unshuffle(torch.from_numpy(validmask).unsqueeze(0).unsqueeze(0).float())
            validmask_patch = torch.sum(validmask_patch, dim=[0, 1])
            validmask_patch = validmask_patch > (self.mask_patch_size ** 2 * 0.3)
            validmask_patch = validmask_patch.squeeze().numpy().astype(np.int64)

            remain_selection = (validmask_patch * mask).flatten()
            visible_valid_patch_num = np.sum((1 - mask) * validmask_patch)

            mask = mask.flatten()
            remain = 2 - visible_valid_patch_num # At least Two Patch Visible
            if remain > 0:
                rnd_idx = np.random.permutation(self.token_count)
                remain_idx = rnd_idx[remain_selection[rnd_idx] == 1]
                remain_idx = remain_idx[0:remain]
                mask[remain_idx] = 0

        mask = mask.reshape((self.rand_size_h, self.rand_size_w))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class SimMIMTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])

        if config.MODEL.TYPE == 'swin':
            model_patch_size = config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size = config.MODEL.VIT.PATCH_SIZE
        elif config.MODEL.TYPE == 'pvt_small' or config.MODEL.TYPE == 'pvt_medium':
            model_patch_size = 4
        else:
            raise NotImplementedError

        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO_SCANNET,
        )

    def __call__(self, img, idx=None, validmask=None):
        img = self.transform_img(img)
        mask = self.mask_generator(idx, validmask)

        return img, mask


def build_loader_scannet(config, logger):
    transform = SimMIMTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    scannet = ScanNetBuilder(data_root=config.DATA.DATA_PATH_SCANNET, progress_bar=False, minoverlap=config.DATA.MINOVERLAP_SCANNET, debug=False)
    scannet_train = scannet.build_scenes(split="train", transform=transform)
    scannet_train = ConcatDataset(scannet_train)

    logger.info(f'Build dataset: train images = {len(scannet_train)}')
    logger.info(f'MIN OVERLAP = {config.DATA.MINOVERLAP_SCANNET}')

    return scannet_train

def build_loader_imagenetaug(config, logger):
    from datasets.imagenet_aug import ImangeNetAug
    transform = SimMIMTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    imagenetaug = ImangeNetAug(data_root=config.DATA.DATA_PATH, auscale=1.0, ht=config.DATA.IMG_SIZE[0],
                               wd=config.DATA.IMG_SIZE[1],
                               homography_augmentation=True, transform=transform)
    return imagenetaug