import os, io
import os.path as osp
import h5py
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import copy
import torchvision.transforms as T
from loguru import logger
from torch.utils.data import ConcatDataset
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import to_2tuple

class ScanNetScene:
    def __init__(self, data_root, scene_info, transform, minoverlap=0.0) -> None:
        self.scene_root = osp.join(data_root, "scans")
        self.data_names = scene_info['name']
        self.overlaps = scene_info['score']
        # Only sample 10s
        valid = (self.data_names[:, -2:] % 10).sum(axis=-1) == 0
        valid = valid * (self.overlaps > minoverlap)
        self.overlaps = self.overlaps[valid]
        self.data_names = self.data_names[valid]
        if len(self.data_names) > 10000:
            pairinds = np.random.choice(np.arange(0, len(self.data_names)), 10000, replace=False)
            self.data_names = self.data_names[pairinds]
            self.overlaps = self.overlaps[pairinds]
        self.transform = transform

    def load_im(self, im_ref, crop=None):
        im = Image.open(im_ref)
        return im

    def load_depth(self, depth_ref, crop=None):
        depth = Image.open(depth_ref)
        depth = np.array(depth).astype(np.uint16)
        depth = depth / 1000
        depth = torch.from_numpy(depth).float()  # (h, w)
        return depth

    def __len__(self):
        return len(self.data_names)

    def scale_intrinsic(self, K, wi, hi):
        sx, sy = self.wt / wi, self.ht / hi
        sK = torch.tensor([[sx, 0, 0],
                           [0, sy, 0],
                           [0, 0, 1]])
        return sK @ K

    def read_scannet_pose(self, path):
        """ Read ScanNet's Camera2World pose and transform it to World2Camera.

        Returns:
            pose_w2c (np.ndarray): (4, 4)
        """
        cam2world = np.loadtxt(path, delimiter=' ')
        world2cam = np.linalg.inv(cam2world)
        return world2cam

    def read_scannet_intrinsic(self, path):
        """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
        """
        intrinsic = np.loadtxt(path, delimiter=' ')
        return intrinsic[:-1, :-1]

    def __getitem__(self, pair_idx):
        # read intrinsics of original size

        if pair_idx >= len(self.data_names):
            print("Scannet PairIdx Larger Than max Length")
        assert pair_idx < len(self.data_names)

        # data_name = self.data_names[10]
        scene_name, scene_sub_name, stem_name_1, stem_name_2 = 653, 1, 700, 760
        # scene_name, scene_sub_name, stem_name_1, stem_name_2 = data_name
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'
        h5pypath = os.path.join(self.scene_root, scene_name, '{}.hdf5'.format(scene_name))

        ims = list()
        with h5py.File(h5pypath, 'r') as hf:
            for time in range(0, 500, 10):
                # Load positive pair data
                try:
                    im = io.BytesIO(np.array(hf['color'][f'{stem_name_1 + time}.jpg']))
                    im = self.load_im(im)
                    ims.append(im)
                except:
                    break

        data = dict()
        for idx, im in enumerate(ims):
            im, mask = self.transform(im)
            data[idx] = im

        data['mask'] = mask
        return data

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
            mask_ratio=config.DATA.MASK_RATIO_SCANNET,
        )
        logger.info("Mask Patch Size %d, ratio %f" % (config.DATA.MASK_PATCH_SIZE, config.DATA.MASK_RATIO_SCANNET))

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

def build_loader_scannet(config):
    transform = SimMIMTransform(config)
    all_scenes = os.listdir(os.path.join(config.DATA.DATA_PATH_SCANNET, 'scannet_indices'))
    scene_name = all_scenes[100]
    scene_info = np.load(os.path.join(os.path.join(config.DATA.DATA_PATH_SCANNET, 'scannet_indices'), scene_name), allow_pickle=True)
    scene_train = ScanNetScene(data_root=config.DATA.DATA_PATH_SCANNET, scene_info=scene_info, transform=transform, minoverlap=0.3)
    return scene_train