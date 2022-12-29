import os, io
from PIL import Image
import h5py
import numpy as np
import torch
import os.path as osp
from tqdm import tqdm

class ScanNetScene:
    def __init__(self, data_root, scene_info, transform, minoverlap=0.0) -> None:
        self.scene_root = osp.join(data_root, "scans")
        self.data_names = scene_info['name']
        self.overlaps = scene_info['score']
        # Only sample 10s
        valid = (self.data_names[:,-2:] % 10).sum(axis=-1) == 0
        valid = valid * (self.overlaps > minoverlap)
        self.overlaps = self.overlaps[valid]
        self.data_names = self.data_names[valid]
        if len(self.data_names) > 10000:
            pairinds = np.random.choice(np.arange(0,len(self.data_names)), 10000, replace=False)
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
        sx, sy = self.wt / wi, self.ht /  hi
        sK = torch.tensor([[sx, 0, 0],
                        [0, sy, 0],
                        [0, 0, 1]])
        return sK@K

    def read_scannet_pose(self,path):
        """ Read ScanNet's Camera2World pose and transform it to World2Camera.
        
        Returns:
            pose_w2c (np.ndarray): (4, 4)
        """
        cam2world = np.loadtxt(path, delimiter=' ')
        world2cam = np.linalg.inv(cam2world)
        return world2cam

    def read_scannet_intrinsic(self,path):
        """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
        """
        intrinsic = np.loadtxt(path, delimiter=' ')
        return intrinsic[:-1, :-1]

    def __getitem__(self, pair_idx):
        # read intrinsics of original size
        data_name = self.data_names[pair_idx]
        scene_name, scene_sub_name, stem_name_1, stem_name_2 = data_name
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'

        h5pypath = os.path.join(self.scene_root, scene_name, '{}.hdf5'.format(scene_name))
        # print(h5pypath)
        with h5py.File(h5pypath, 'r') as hf:
            # Load positive pair data
            im_src_ref = io.BytesIO(np.array(hf['color'][f'{stem_name_1}.jpg']))
            im_pos_ref = io.BytesIO(np.array(hf['color'][f'{stem_name_2}.jpg']))

            im_src = self.load_im(im_src_ref)
            im_pos = self.load_im(im_pos_ref)

        img1 = im_src
        img2 = im_pos

        img1, mask1 = self.transform(img1)
        img2, mask2 = self.transform(img2)
        return img1, mask1, img2, mask2


class ScanNetBuilder:
    def __init__(self, data_root='data/scannet', debug=False, progress_bar=False, minoverlap=0.0) -> None:
        self.data_root = data_root
        self.scene_info_root = os.path.join(data_root, 'scannet_indices')
        self.all_scenes = os.listdir(self.scene_info_root)
        self.debug = debug
        self.progress_bar = progress_bar
        self.minoverlap = minoverlap

    def build_scenes(self, split='train', transform=None):
        # Note: split doesn't matter here as we always use same scannet_train scenes
        scene_names = self.all_scenes

        if self.debug:
            scene_names = scene_names[0:100]

        scenes = []
        for scene_name in tqdm(scene_names, disable=not self.progress_bar):
            scene_info = np.load(os.path.join(self.scene_info_root, scene_name), allow_pickle=True)
            scenes.append(ScanNetScene(self.data_root, scene_info, transform=transform, minoverlap=self.minoverlap))
        return scenes
    
    def weight_scenes(self, concat_dataset, alpha=.5):
        ns = []
        for d in concat_dataset.datasets:
            ns.append(len(d))
        ws = torch.cat([torch.ones(n)/n**alpha for n in ns])
        return ws
