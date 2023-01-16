import os, io, glob
import h5py
import numpy as np
import torch
import os.path as osp
from PIL import Image
from tqdm import tqdm
import natsort

class ScanNetScene:
    def __init__(self, data_root, scene_info) -> None:
        self.scene_root = osp.join(data_root, "scans")
        self.data_names = scene_info['name']
        self.overlaps = scene_info['score']
        # Only sample 10s
        valid = (self.data_names[:, -2:] % 10).sum(axis=-1) == 0
        self.overlaps = self.overlaps[valid]
        self.data_names = self.data_names[valid]

        scene_name, scene_sub_name, _, _ = self.data_names[0]
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'
        h5pypath = os.path.join(self.scene_root, scene_name, '{}.hdf5'.format(scene_name))

        stem_names = list()
        for data_name in self.data_names:
            _, _, stem_name_1, stem_name_2 = data_name
            stem_names.append(stem_name_1)
            stem_names.append(stem_name_2)

        # Prepare for writing
        h5pyfolder_dst = os.path.join('/media/shengjie/scratch2/EMAwareFlowDatasets/ScanNetRGB/scans', scene_name)
        os.makedirs(h5pyfolder_dst, exist_ok=True)
        h5pypath_dst = os.path.join(h5pyfolder_dst, '{}.hdf5'.format(scene_name))
        if os.path.exists(h5pypath_dst):
            os.remove(h5pypath_dst)

        hfdst = h5py.File(h5pypath_dst, 'a')  # open the file in append mode
        gp = hfdst.create_group('color')

        # Unique
        stem_names = list(set(stem_names))
        stem_names = natsort.natsorted(stem_names)
        with h5py.File(h5pypath, 'r') as hf:
            for stem_name in stem_names:
                # Load positive pair data
                im_src = np.array(hf['color'][f'{stem_name}.jpg'])
                gp.create_dataset(f'{stem_name}.jpg', data=im_src)  # save it in the subgroup. each a-subgroup contains all the images.

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
    def __init__(self, data_root='data/scannet') -> None:
        self.data_root = data_root
        self.scene_info_root = os.path.join(data_root, 'scannet_indices')
        self.all_scenes = os.listdir(self.scene_info_root)

    def build_scenes(self):
        # Note: split doesn't matter here as we always use same scannet_train scenes
        scene_names = self.all_scenes
        scenes = []
        for scene_name in tqdm(scene_names):
            scene_info = np.load(os.path.join(self.scene_info_root, scene_name), allow_pickle=True)
            scenes.append(ScanNetScene(self.data_root, scene_info))
        return scenes


if __name__ == '__main__':
    source_root = '/media/shengjie/scratch2/EMAwareFlowDatasets/ScanNet'
    target_root = '/media/shengjie/scratch2/EMAwareFlowDatasets/ScanNetRGB'
    os.makedirs(target_root, exist_ok=True)

    scannet = ScanNetBuilder(data_root=source_root)
    scannet_train = scannet.build_scenes()