import os, math, h5py, random
import numpy as np
from PIL import Image

import copy
import torch
from torch.utils.data import ConcatDataset
from loguru import logger
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import to_2tuple

from tools.tools import get_depth_tuple_transform_ops, get_tuple_transform_ops, warp_kpts

import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate


class MegadepthScene:
    def __init__(
        self,
        data_root,
        scene_info,
        min_overlap=0.0,
        transform=None
    ) -> None:
        self.data_root = data_root
        self.image_paths = scene_info["image_paths"]
        self.depth_paths = scene_info["depth_paths"]
        self.intrinsics = scene_info["intrinsics"]
        self.poses = scene_info["poses"]
        self.pairs = scene_info["pairs"]
        self.overlaps = scene_info["overlaps"]
        threshold = self.overlaps > min_overlap
        self.pairs = self.pairs[threshold]
        self.overlaps = self.overlaps[threshold]
        if len(self.pairs) > 100000:
            pairinds = np.random.choice(
                np.arange(0, len(self.pairs)), 100000, replace=False
            )
            self.pairs = self.pairs[pairinds]
            self.overlaps = self.overlaps[pairinds]

        self.transform = transform
    def load_im(self, im_ref, crop=None):
        im = Image.open(im_ref)
        return im

    def load_depth(self, depth_ref, crop=None):
        depth = np.array(h5py.File(depth_ref, "r")["depth"])
        return torch.from_numpy(depth)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, pair_idx):
        idx1, idx2 = self.pairs[pair_idx]

        im_src_ref, im_pos_ref = self.image_paths[idx1], self.image_paths[idx2]
        im_src_ref = os.path.join(self.data_root, im_src_ref)
        im_pos_ref = os.path.join(self.data_root, im_pos_ref)

        im_src = self.load_im(im_src_ref)
        im_pos = self.load_im(im_pos_ref)

        img1, mask1 = self.transform(im_src)
        img2, mask2 = self.transform(im_pos)

        return img1, mask1, img2, mask2

class MegadepthBuilder:
    def __init__(self, data_root="data/megadepth") -> None:
        self.data_root = data_root
        self.scene_info_root = os.path.join(data_root, "prep_scene_info")
        self.all_scenes = os.listdir(self.scene_info_root)
        self.test_scenes = ["0017.npy", "0004.npy", "0048.npy", "0013.npy"]
        self.test_scenes_loftr = ["0015.npy", "0022.npy"]

    def build_scenes(self, split="train", min_overlap=0.0, transform=None):
        if split == "train":
            scene_names = set(self.all_scenes) - set(self.test_scenes)
        elif split == "train_loftr":
            scene_names = set(self.all_scenes) - set(self.test_scenes_loftr)
        elif split == "test":
            scene_names = self.test_scenes
        elif split == "test_loftr":
            scene_names = self.test_scenes_loftr
        else:
            raise ValueError(f"Split {split} not available")
        scenes = []
        for scene_name in scene_names:
            scene_info = np.load(os.path.join(self.scene_info_root, scene_name), allow_pickle=True).item()
            scenes.append(MegadepthScene(self.data_root, scene_info, min_overlap=min_overlap, transform=transform))
        return scenes

    def weight_scenes(self, concat_dataset, alpha=0.5):
        ns = []
        for d in concat_dataset.datasets:
            ns.append(len(d))
        ws = torch.cat([torch.ones(n) / n**alpha for n in ns])
        return ws

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
        self.config = copy.deepcopy(config)

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
            mask_ratio=config.DATA.MASK_RATIO,
        )
        logger.info("Mask Patch Size %d, ratio %f" % (config.DATA.MASK_PATCH_SIZE, config.DATA.MASK_RATIO))

    def __call__(self, img, idx=None, validmask=None):
        transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(self.config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])

        img = transform_img(img)
        mask = self.mask_generator(idx, validmask)

        return img, mask

def build_loader_mega(config, logger=None):
    transform = SimMIMTransform(config)
    mega = MegadepthBuilder(data_root=config.DATA.DATA_PATH)
    megadepth_train = mega.build_scenes(split="train_loftr", min_overlap=0.35, transform=transform)
    dataset = ConcatDataset(megadepth_train)

    if logger is not None:
        logger.info(f'Pre-train data transform:\n{transform}')
        logger.info(f'Build dataset: train images = {len(dataset)}')

    return dataset