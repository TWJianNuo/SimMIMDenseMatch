import os, glob, tqdm, sys, inspect, io, math, random, copy
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))

import cv2
import h5py
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, List
from collections.abc import Sequence
from loguru import logger


import torch, torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import resized_crop, InterpolationMode

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from tools.tools import get_tuple_transform_ops, warp_kpts, tensor2rgb, tensor2disp

class RandomAffine:
    """Apply random affine transformation."""
    def __init__(self, p_flip=0.0, max_rotation=0.0, max_shear=0.0, max_scale=0.0, max_ar_factor=0.0, maxtrans=0.0):
        """
        Args:
            p_flip:
            max_rotation:
            max_shear:
            max_scale:
            max_ar_factor:
            border_mode:
        """
        super().__init__()
        self.p_flip = p_flip
        self.max_rotation = max_rotation
        self.max_shear = max_shear
        self.max_scale = max_scale
        self.max_ar_factor = max_ar_factor
        self.maxtrans = maxtrans

    def roll(self, imgh, imgw):
        do_flip = random.random() < self.p_flip
        theta = random.uniform(-self.max_rotation, self.max_rotation)

        shear_x = random.uniform(-self.max_shear, self.max_shear)
        shear_y = random.uniform(-self.max_shear, self.max_shear)

        ar_factor = np.exp(random.uniform(-self.max_ar_factor, self.max_ar_factor))
        scale_factor = np.exp(random.uniform(-self.max_scale, self.max_scale))

        transx = random.uniform(-self.maxtrans, self.maxtrans)
        transy = random.uniform(-self.maxtrans, self.maxtrans)
        transx = transx * imgw
        transy = transy * imgh

        return do_flip, theta, (shear_x, shear_y), (scale_factor / np.sqrt(ar_factor), scale_factor * np.sqrt(ar_factor)), transx, transy

    def _construct_t_mat(self, im_h, im_w, do_flip, theta, shear_values, scale_factors, tx, ty):
        t_mat = np.identity(3)

        if do_flip:
            t_mat[0, 0] = -1.0
            t_mat[0, 2] = im_w

        t_rot = cv2.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), theta, 1.0)
        t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))

        t_shear = np.array([[1.0, shear_values[0], 0.0],
                            [shear_values[1], 1.0, 0.0],
                            [0.0, 0.0, 1.0]])

        t_scale = np.array([[scale_factors[0], 0.0, 0.0],
                            [0.0, scale_factors[1], 0.0],
                            [0.0, 0.0, 1.0]])

        t_translation = np.identity(3)
        t_translation[0, 2] = tx
        t_translation[1, 2] = ty

        rndH = t_scale @ t_rot @ t_shear @ t_mat @ t_translation

        return rndH

    def transform(self, imgh, imgw):
        do_flip, theta, shear_values, scale_factors, tx, ty = self.roll(imgh, imgw)
        rndH = self._construct_t_mat(imgh, imgw, do_flip, theta, shear_values, scale_factors, tx, ty)
        return rndH

class RandomResizedHomography():
    """Crop a random portion of image and resize it to a given size.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image[.Resampling].NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
    """

    def __init__(self,
                 size,
                 auscale=1.0,
                 scale=(0.08, 1.0),
                 ratio=(3.0 / 4.0, 4.0 / 3.0),
                 p_flip=0.0,
                 max_rotation=45.0,
                 max_shear=0.5,
                 max_scale=0.5,
                 max_ar_factor=0.5,
                 maxtrans=0.2,
                 padding='zeros',
                 disablefill=False,
                 homography_augmentation=False
                 ):
        self.size = size

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")

        self.scale = scale
        self.ratio = ratio
        self.homography_augmentation = homography_augmentation

        if self.homography_augmentation:
            # Reduce Augmentation if including the homography
            auscale = auscale / 1.5

        self.p_flip = p_flip * auscale
        self.max_rotation = max_rotation * auscale
        self.max_shear = max_shear * auscale
        self.max_scale = max_scale * auscale
        self.max_ar_factor = max_ar_factor * auscale
        self.maxtrans = maxtrans * auscale
        self.rndhomolim = 0.25 * auscale

        self.padding = padding
        self.disablefill = disablefill

    @staticmethod
    def get_params(img: Image, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = img.size
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def rndhomo(self, imgh, imgw, rndhomolim):
        while True:
            try:
                fx = float(imgw)
                fy = float(imgh)
                src_point = np.float32([[fx / 4, fy / 4],
                                        [3 * fx / 4, fy / 4],
                                        [fx / 4, 3 * fy / 4],
                                        [3 * fx / 4, 3 * fy / 4]
                                        ])
                random_shift = (np.random.rand(4, 2) - 0.5) * 2
                random_shift[:, 0] = random_shift[:, 0] * rndhomolim * fx
                random_shift[:, 1] = random_shift[:, 1] * rndhomolim * fy
                dst_point = src_point + random_shift.astype(np.float32)
                transform, status = cv2.findHomography(src_point, dst_point)
                break
            except:
                continue
        return transform

    def sample_img(self, H, img, imgh, imgw, imgorgh, imgorgw):
        # Img -> H -> Img Sampled
        xxf, yyf = np.meshgrid(
            *[
                np.linspace(
                    -1 + 1 / n, 1 - 1 / n, n
                )
                for n in (imgw, imgh)
            ]
        )
        xxf, yyf = (xxf + 1) / 2 * imgw, (yyf + 1) / 2 * imgh
        xxf = xxf.flatten()
        yyf = yyf.flatten()

        pts2d = np.stack([xxf, yyf, np.ones_like(xxf)], axis=0)
        prj2d = np.linalg.inv(H) @ pts2d
        prjxf, prjyf, d = np.split(prj2d, [1, 2], axis=0)
        prjxf = prjxf / (d + 1e-5)
        prjyf = prjyf / (d + 1e-5)

        prjxf = (prjxf / imgorgw - 0.5) * 2
        prjyf = (prjyf / imgorgh - 0.5) * 2

        prjx = np.reshape(prjxf, self.size)
        prjy = np.reshape(prjyf, self.size)

        pts2dsample = torch.stack([torch.from_numpy(prjx), torch.from_numpy(prjy)], dim=-1).unsqueeze(0).float()
        img_torch = torch.from_numpy(np.array(img)).permute([2, 0, 1]).unsqueeze(0).float()
        imgsampled = torch.nn.functional.grid_sample(img_torch, pts2dsample, mode='bilinear', align_corners=True, padding_mode=self.padding)
        imgsampled = torch.clamp(imgsampled, min=0.0, max=255.0)
        imgsampled = imgsampled.squeeze().permute([1, 2, 0]).numpy().astype(np.uint8)
        imgsampled = Image.fromarray(imgsampled)

        invisible_mask = (prjx > -1) * (prjx < 1) * (prjy > -1) * (prjy < 1)
        invisible_mask = invisible_mask == 0

        return imgsampled, pts2dsample, invisible_mask

    def downsample8(self, invisible_mask):
        height, width = self.size
        height8 = int(height / 8)
        width8 = int(width / 8)
        invisible_mask8 = torch.nn.functional.interpolate(torch.from_numpy(invisible_mask).float().view([1, 1, height, width]), (height8, width8), mode='nearest')
        invisible_mask8 = (invisible_mask8.squeeze().numpy() == 1)
        return invisible_mask8

    def fill_by_bck(self, img, imgbck, mask):
        img = np.array(img)
        imgbck = np.array(imgbck)

        mask = np.expand_dims(mask, axis=2).astype(np.float32)

        img = img.astype(np.float32) * (1 - mask) + imgbck.astype(np.float32) * mask
        img = Image.fromarray(img.astype(np.uint8))

        return img

    def forward(self, img, imgbck):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        worg, horg = img.size

        imgbck = imgbck.resize(self.size[::-1])

        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        shiftM = np.eye(3)
        shiftM[0, 2] = -j
        shiftM[1, 2] = -i

        scaleM = np.eye(3)
        scaleM[0, 0] = self.size[1] / w
        scaleM[1, 1] = self.size[0] / h

        srcH = scaleM @ shiftM

        if np.random.uniform(0, 1) < 0.3:
            if self.homography_augmentation:
                rndH = self.rndhomo(imgh=self.size[0], imgw=self.size[1], rndhomolim=self.rndhomolim)
            else:
                rndH = np.eye(3)

            # One Rectangle and One Transformed
            raffine = RandomAffine(p_flip=self.p_flip, max_rotation=self.max_rotation,
                                   max_shear=self.max_shear, max_scale=self.max_scale,
                                   max_ar_factor=self.max_ar_factor, maxtrans=self.maxtrans)
            transformM = raffine.transform(imgh=self.size[0], imgw=self.size[1]) @ rndH @ srcH

            if np.random.uniform(0, 1) < 0.5:
                tmp = transformM
                transformM = srcH
                srcH = tmp

            imgsrc, _, invis_src = self.sample_img(srcH, img, imgh=self.size[0], imgw=self.size[1], imgorgh=horg, imgorgw=worg)
            imgdst, _, invis_dst = self.sample_img(transformM, img, imgh=self.size[0], imgw=self.size[1], imgorgh=horg, imgorgw=worg)
            imgsrc_sampled, gtcorrespondence, invis_src_recon = self.sample_img(
                srcH @ np.linalg.inv(transformM), imgdst, imgh=self.size[0], imgw=self.size[1], imgorgh=self.size[0], imgorgw=self.size[1]
            )
            imgdst_sampled, _, invis_dst_recon = self.sample_img(
                transformM @ np.linalg.inv(srcH), imgsrc, imgh=self.size[0], imgw=self.size[1], imgorgh=self.size[0], imgorgw=self.size[1]
            )
            HM = np.linalg.inv(srcH @ np.linalg.inv(transformM))
        else:
            scale = 1.414

            if self.homography_augmentation:
                rndH1 = self.rndhomo(imgh=self.size[0], imgw=self.size[1], rndhomolim=self.rndhomolim / scale)
                rndH2 = self.rndhomo(imgh=self.size[0], imgw=self.size[1], rndhomolim=self.rndhomolim / scale)
            else:
                rndH1 = np.eye(3)
                rndH2 = np.eye(3)

            raffine = RandomAffine(p_flip=self.p_flip / scale, max_rotation=self.max_rotation / scale,
                                   max_shear=self.max_shear / scale, max_scale=self.max_scale / scale,
                                   max_ar_factor=self.max_ar_factor / scale, maxtrans=self.maxtrans / scale)
            transformM1 = raffine.transform(imgh=self.size[0], imgw=self.size[1]) @ rndH1 @ srcH
            transformM2 = raffine.transform(imgh=self.size[0], imgw=self.size[1]) @ rndH2 @ srcH
            imgsrc, _, invis_src = self.sample_img(transformM1, img, imgh=self.size[0], imgw=self.size[1], imgorgh=horg, imgorgw=worg)
            imgdst, _, invis_dst = self.sample_img(transformM2, img, imgh=self.size[0], imgw=self.size[1], imgorgh=horg, imgorgw=worg)
            imgsrc_sampled, gtcorrespondence, invis_src_recon = self.sample_img(
                transformM1 @ np.linalg.inv(transformM2), imgdst, imgh=self.size[0], imgw=self.size[1], imgorgh=self.size[0], imgorgw=self.size[1]
            )
            imgdst_sampled, _, invis_dst_recon = self.sample_img(
                transformM2 @ np.linalg.inv(transformM1), imgsrc, imgh=self.size[0], imgw=self.size[1], imgorgh=self.size[0], imgorgw=self.size[1]
            )
            HM = np.linalg.inv(transformM1 @ np.linalg.inv(transformM2))

        if not self.disablefill:
            imgsrc = self.fill_by_bck(imgsrc, imgbck, invis_src)
            imgdst = self.fill_by_bck(imgdst, imgbck.transpose(Image.FLIP_TOP_BOTTOM), invis_dst)

        # # Debug Purpuse
        # imgsrc.show()
        # imgsrc_sampled.show()
        # imgdst.show()
        # imgdst_sampled.show()
        # a = 1
        # xx, yy = np.meshgrid(np.arange(self.size[1]), np.arange(self.size[0]), indexing='xy')
        # xxf = xx.flatten()
        # yyf = yy.flatten()
        # pts2d = np.stack([xxf, yyf, np.ones_like(xxf)], axis=0)
        # pts2d_ = np.linalg.inv(transformM) @ pts2d
        # xxf_, yyf_, _ = np.split(pts2d_, [1, 2], axis=0)
        # xxf_ = (xxf_ / (worg - 1) - 0.5) * 2
        # yyf_ = (yyf_ / (horg - 1) - 0.5) * 2
        #
        # xxf_ = np.reshape(xxf_, self.size)
        # yyf_ = np.reshape(yyf_, self.size)
        #
        # pts2dsample = torch.stack([torch.from_numpy(xxf_), torch.from_numpy(yyf_)], dim=-1).unsqueeze(0).float()
        #
        # img_torch = torch.from_numpy(np.array(img)).permute([2, 0, 1]).unsqueeze(0).float() / 255.0
        # tensor2rgb(resized_crop(img_torch, i, j, h, w, self.size, InterpolationMode.BILINEAR), viewind=0).show()
        #
        # resampled = torch.nn.functional.grid_sample(img_torch, pts2dsample, mode='bilinear', align_corners=True)
        # tensor2rgb(resampled, viewind=0).show()

        # vis_src_recon = (invis_src_recon == 0) * (invis_src == 0)
        vis_src_recon = invis_src == 0
        # tensor2disp(torch.from_numpy(vis_src_recon).float().view([1, 1, 240, 320]), vmax=1, viewind=0).show()
        vis_src_recon8 = self.downsample8(vis_src_recon)
        # tensor2disp(torch.from_numpy(vis_src_recon).float().view([1, 1, 30, 40]), vmax=1, viewind=0).show()

        # vis_dst_recon = (invis_dst_recon == 0) * (invis_dst == 0)
        vis_dst_recon = invis_dst == 0
        # tensor2disp(torch.from_numpy(vis_dst_recon).float().view([1, 1, 240, 320]), vmax=1, viewind=0).show()
        vis_dst_recon8 = self.downsample8(vis_dst_recon)
        # tensor2disp(torch.from_numpy(vis_dst_recon).float().view([1, 1, 30, 40]), vmax=1, viewind=0).show()

        return imgsrc, imgdst, gtcorrespondence, vis_src_recon, vis_dst_recon, vis_src_recon8, vis_dst_recon8, HM.astype(np.float32), np.linalg.inv(HM).astype(np.float32)

class ImangeNetAug:
    def __init__(
        self,
        data_root,
        ht=384,
        wd=512,
        auscale=1.0,
        padding='zeros',
        disablefill=False,
        homography_augmentation=False,
        transform=None
    ) -> None:
        self.data_root = data_root
        self.wd, self.ht = wd, ht
        filename = os.path.join(project_root, 'splits', 'imagenet.txt')
        with open(filename) as file:
            lines = [line.rstrip() for line in file]
        self.image_paths = lines

        if homography_augmentation:
            logger.info("Enable Homography Augmentation with Scale %f" % auscale)
        else:
            logger.info("Disable Homography Augmentation with Scale %f" % auscale)

        self.rndhomography = RandomResizedHomography((self.ht, self.wd),
                                                     auscale=auscale,
                                                     scale=(0.5, 0.8),
                                                     ratio=(3. / 4., 4. / 3.),
                                                     padding=padding,
                                                     disablefill=disablefill,
                                                     homography_augmentation=homography_augmentation
                                                     )
        self.augcolor = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.5 / 3.14)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def load_im_byidx(self, idx):
        filename, hdf5path = self.image_paths[idx].split(' ')
        hdf5path = os.path.join(self.data_root, hdf5path)

        with h5py.File(hdf5path, 'r') as hf:
            # Load positive pair data
            im_src_ref = io.BytesIO(np.array(hf[filename]))
            im = Image.open(im_src_ref)

        if np.array(im).ndim == 2:
            im = np.array(im)
            im = np.stack([im, im, im], axis=2)
            im = Image.fromarray(im)

        im = np.array(im)[:, :, 0:3]
        im = Image.fromarray(im)

        return im

    def __getitem__(self, idx):
        im_src = self.load_im_byidx(idx)
        im_bck = self.load_im_byidx(np.random.randint(0, len(self.image_paths), 1).item())

        imgsrc, imgdst, gtcorrespondence, vis_src_recon, vis_dst_recon, vis_src_recon8, vis_dst_recon8, _, _ = self.rndhomography.forward(im_src, im_bck)

        if np.random.uniform(0, 1) < 0.5:
            imgsrc = self.augcolor(imgsrc)
        if np.random.uniform(0, 1) < 0.5:
            imgdst = self.augcolor(imgdst)

        imgsrc, mask1 = self.transform(imgsrc, validmask=vis_src_recon)
        imgdst, mask2 = self.transform(imgdst, validmask=vis_dst_recon)

        vis_src_recon = vis_src_recon.astype(np.int64)
        vis_dst_recon = vis_dst_recon.astype(np.int64)

        return {'imgsrc': imgsrc, 'mask1': mask1, 'imgdst': imgdst, 'mask2': mask2, 'vis_src_recon': vis_src_recon, 'vis_dst_recon': vis_dst_recon}

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

    def __call__(self, img, idx=None, validmask=None):
        transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])

        img = transform_img(img)
        mask = self.mask_generator(idx, validmask)

        return img, mask

def build_loader_imagenetaug(config):
    transform = SimMIMTransform(config)
    imagenetaug = ImangeNetAug(data_root=config.DATA.DATA_PATH, auscale=1.0, ht=config.DATA.IMG_SIZE[0],
                               wd=config.DATA.IMG_SIZE[1], homography_augmentation=True, transform=transform)
    return imagenetaug