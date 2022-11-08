import os, copy
import h5py
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as tvf
from torch.utils.data import ConcatDataset


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
        # read intrinsics of original size
        idx1, idx2 = self.pairs[pair_idx]
        K1 = torch.tensor(self.intrinsics[idx1].copy(), dtype=torch.float).reshape(3, 3)
        K2 = torch.tensor(self.intrinsics[idx2].copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T1 = self.poses[idx1]
        T2 = self.poses[idx2]
        T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1)), dtype=torch.float)[
            :4, :4
        ]  # (4, 4)

        # Load positive pair data
        im1, im2 = self.image_paths[idx1], self.image_paths[idx2]
        depth1, depth2 = self.depth_paths[idx1], self.depth_paths[idx2]
        im_src_ref = os.path.join(self.data_root, im1)
        im_pos_ref = os.path.join(self.data_root, im2)
        depth_src_ref = os.path.join(self.data_root, depth1)
        depth_pos_ref = os.path.join(self.data_root, depth2)
        # return torch.randn((1000,1000))
        im_src = self.load_im(im_src_ref)
        im_pos = self.load_im(im_pos_ref)
        depth_src = self.load_depth(depth_src_ref)
        depth_pos = self.load_depth(depth_pos_ref)

        # Recompute camera intrinsic matrix due to the resize
        im_src, depth_src, K1, masksrc = self.transform(im_src, depth_src, K1)
        im_pos, depth_pos, K2, maskpos = self.transform(im_pos, depth_pos, K2)

        p = np.random.random()
        if p < 0.1:
            im_pos = copy.deepcopy(im_src)
            depth_pos = copy.deepcopy(depth_src)
            K2 = copy.deepcopy(K1)
            maskpos = maskpos * 0
            T_1to2 = T_1to2 @ torch.linalg.inv(T_1to2)

        elif (p >= 0.1) and (p < 0.4):
            maskpos = maskpos * 0

        data_dict = {
            "image0_color": im_src,
            "mask0_": masksrc,
            "image1_color": im_pos,
            "mask1_": maskpos,
            "depth0": depth_src,
            "depth1": depth_pos,
            "K0": K1,
            "K1": K2,
            "T_0to1": T_1to2,
            "T_1to0": torch.linalg.inv(T_1to2),
        }

        return data_dict



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
