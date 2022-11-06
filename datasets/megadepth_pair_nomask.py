import os, copy
import h5py
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as tvf
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tools.tools import get_depth_tuple_transform_ops, get_tuple_transform_ops


class MegadepthScene:
    def __init__(
        self,
        data_root,
        scene_info,
        ht=384,
        wt=512,
        min_overlap=0.0,
        shake_t=0,
        rot_prob=0.0,
        normalize=True,
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
            # np.random.seed(2022)
            pairinds = np.random.choice(
                np.arange(0, len(self.pairs)), 100000, replace=False
            )
            # print(pairinds[0:100], len(self.pairs))
            self.pairs = self.pairs[pairinds]
            self.overlaps = self.overlaps[pairinds]
        # counts, bins = np.histogram(self.overlaps,20)
        # print(counts)
        self.im_transform_ops = get_tuple_transform_ops(
            resize=(ht, wt), normalize=None
        )
        self.depth_transform_ops = get_depth_tuple_transform_ops(
            resize=(ht, wt), normalize=False
        )
        self.wt, self.ht = wt, ht
        self.shake_t = shake_t
        self.rot_prob = rot_prob

        self.transform = transform

    def load_im(self, im_ref, crop=None):
        im = Image.open(im_ref)
        return im

    def load_depth(self, depth_ref, crop=None):
        depth = np.array(h5py.File(depth_ref, "r")["depth"])
        return torch.from_numpy(depth)

    def __len__(self):
        return len(self.pairs)

    def scale_intrinsic(self, K, wi, hi):
        sx, sy = self.wt / wi, self.ht / hi
        sK = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return sK @ K

    def rand_shake(self, *things):
        t = np.random.choice(range(-self.shake_t, self.shake_t + 1), size=2)
        return [
            tvf.affine(thing, angle=0.0, translate=list(t), scale=1.0, shear=[0.0, 0.0])
            for thing in things
        ], t

    def __getitem__(self, pair_idx):
        img1_path, img2_path = self.image_paths[self.pairs[pair_idx][0]], self.image_paths[self.pairs[pair_idx][1]]
        img1_path = os.path.join(self.data_root, img1_path)
        img2_path = os.path.join(self.data_root, img2_path)
        img1 = self.load_im(img1_path)
        img1, mask1 = self.transform(img1)
        # img2 = self.load_im(img2_path)
        # img2, mask2 = self.transform(img2)
        # mask2 = mask2 * 0
        img2 = copy.deepcopy(img1)
        mask2 = copy.deepcopy(mask1) * 0
        return img1, mask1, img2, mask2


class MegadepthBuilder:
    def __init__(self, data_root="data/megadepth") -> None:
        self.data_root = data_root
        self.scene_info_root = os.path.join(data_root, "prep_scene_info")
        self.all_scenes = os.listdir(self.scene_info_root)
        self.test_scenes = ["0017.npy", "0004.npy", "0048.npy", "0013.npy"]
        self.test_scenes_loftr = ["0015.npy", "0022.npy"]

    def build_scenes(self, split="train", min_overlap=0.0, **kwargs):
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
        # scene_names = list(scene_names)
        # scene_names.sort()
        for scene_name in scene_names:
            scene_info = np.load(
                os.path.join(self.scene_info_root, scene_name), allow_pickle=True
            ).item()
            scenes.append(
                MegadepthScene(
                    self.data_root, scene_info, min_overlap=min_overlap, **kwargs
                )
            )
        return scenes

    def weight_scenes(self, concat_dataset, alpha=0.5):
        ns = []
        for d in concat_dataset.datasets:
            ns.append(len(d))
        ws = torch.cat([torch.ones(n) / n**alpha for n in ns])
        return ws


if __name__ == "__main__":
    mega_test = ConcatDataset(MegadepthBuilder().build_scenes(split="train"))
    mega_test[0]
