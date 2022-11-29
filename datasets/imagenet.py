import os, copy, glob, io, tqdm
import h5py
import numpy as np

from PIL import Image
from io import StringIO

import torch
import torchvision.transforms.functional as tvf
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tools.tools import get_depth_tuple_transform_ops, get_tuple_transform_ops

class ImageNetDataset:
    def __init__(
        self,
        data_root,
        split,
        transform
    ) -> None:
        self.data_root = data_root
        assert split in ['train', 'val']
        # Get the list of images from ImageNet
        self.transform = transform

        hd5files = glob.glob(os.path.join(data_root, split, '*.hdf5'))
        imgs = list()
        for h5path in tqdm.tqdm(hd5files):
            with h5py.File(h5path, 'r') as hf:
                if split == 'val':
                    for x in tqdm.tqdm(hf.keys()):
                        imgs.append(x)
                else:
                    imgs += list(hf.keys())

        self.imgs = imgs
        self.split = split
        if split == 'val':
            self.imgs = self.imgs[::50]

    def load_im(self, im_ref):
        im = Image.open(im_ref)
        return im

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.split == 'val':
            h5path = os.path.join(self.data_root, self.split, 'n00000000.hdf5')
        elif self.split == 'train':
            h5path = os.path.join(self.data_root, self.split, self.imgs[idx].split('_')[0])
        with h5py.File(h5path, 'r') as hf:
            img = self.load_im(io.BytesIO(np.array(hf[self.imgs[idx]])))

        if self.split == 'val':
            img, mask = self.transform(img, idx)
        else:
            img, mask = self.transform(img)

        return img, mask