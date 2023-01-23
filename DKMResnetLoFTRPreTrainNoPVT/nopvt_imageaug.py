# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------
from __future__ import print_function, division
import os, sys, inspect, time, tqdm, copy
project_root = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_root)

import logging
import argparse
import datetime
import numpy as np
import PIL.Image as Image

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data._utils.collate import default_collate

from config import get_config
from logger import create_logger
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from tools.tools import tensor2rgb, tensor2disp
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper

from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# from DKMResnetLoFTRPreTrainNoPVT.models.build_model import DKMv2
from DKMResnetLoFTRPreTrainNoPVT.models.build_modelwconf import DKMv2wconf
from DKMResnetLoFTRPreTrainNoPVT.models.build_modelwconf_sinview import DKMv2wconfSinView
from DKMResnetLoFTRPreTrainNoPVT.datasets.imagenetaug import build_loader_imagenetaug

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--minoverlap-scannet', type=float, default=0.0)
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--adjustscaler', type=float, default=1.0)
    parser.add_argument('--mask-patch-size', type=int, default=32)

    parser.add_argument('--dist_url', type=str, help='url used to set up distributed training', default='tcp://127.0.0.1:1235')
    parser.add_argument('--dist_backend', type=str, help='distributed backend', default='nccl')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args)

    return args, config

def main(gpu, config, args):
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=gpu)

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")
    if gpu != 0:
        logger.setLevel(logging.ERROR)

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    seed = config.SEED + dist.get_rank() + gpu
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # print config
    logger.info(config.dump())

    imagenetaug = build_loader_imagenetaug(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = DKMv2wconf()
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    n_iter_per_epoch = 5000
    lr_scheduler = build_scheduler(config, optimizer, n_iter_per_epoch)

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    if gpu == 0:
        logger.info(f'Create Summary Writer')
        writer = SummaryWriter(config.OUTPUT, flush_secs=30)
    else:
        writer = None

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        print("Start to Initialize New dataloader")
        imagenetaug_sampler = torch.utils.data.distributed.DistributedSampler(
            copy.deepcopy(imagenetaug), seed=epoch, shuffle=True, drop_last=True)
        imagenetaug_sampler.set_epoch(int(epoch))
        data_loader_train_imagenetaug = torch.utils.data.DataLoader(
            copy.deepcopy(imagenetaug), batch_size=config.DATA.BATCH_SIZE,
            sampler=imagenetaug_sampler, num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True)
        data_loader_train_imagenetaug = iter(data_loader_train_imagenetaug)

        train_one_epoch(config, model, data_loader_train_imagenetaug, optimizer, epoch, lr_scheduler, writer, logger, gpu)
        if gpu == 0 and dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, logger)
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    torch.cuda.synchronize()

def on_after_backward(model, optimizer, logger):
    valid_gradients = True
    max_tensor = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
            if not valid_gradients:
                break
        if max_tensor < param.data.abs().max():
            max_tensor = param.data.abs().max()

    if not valid_gradients:
        logger.setLevel(logging.WARNING)
        logger.warning(f'detected inf or nan values in gradients. not updating model parameters')
        logger.setLevel(logging.ERROR)
        optimizer.zero_grad()

    return valid_gradients, max_tensor

def train_one_epoch(config, model, data_loader_imagenetaug, optimizer, epoch, lr_scheduler, writer, logger, gpu):
    model.train()
    optimizer.zero_grad()

    num_steps = 5000
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx in range(num_steps):
        imagenetaug_batch = next(data_loader_imagenetaug)
        img1_imagenetaug = imagenetaug_batch['imgsrc']
        mask_imagenetaug = imagenetaug_batch['mask1']
        img2_imagenetaug = imagenetaug_batch['imgdst']
        mask_imagenetaug_sup = imagenetaug_batch['vis_src_recon']

        img1_imagenetaug = img1_imagenetaug.cuda(non_blocking=True)
        img2_imagenetaug = img2_imagenetaug.cuda(non_blocking=True)
        mask_imagenetaug = mask_imagenetaug.cuda(non_blocking=True)
        mask_imagenetaug_sup = mask_imagenetaug_sup.cuda(non_blocking=True)

        loss, x_rec = model(img1_imagenetaug, mask_imagenetaug, img2_imagenetaug, mask_imagenetaug_sup)

        assert torch.sum(torch.isnan(img1_imagenetaug)) == 0
        assert torch.sum(torch.isnan(img2_imagenetaug)) == 0
        assert torch.sum(torch.isnan(mask_imagenetaug)) == 0

        img = img1_imagenetaug
        img2 = img2_imagenetaug
        mask = mask_imagenetaug
        mask_sup = mask_imagenetaug_sup

        optimizer.zero_grad()
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())

        valid_gradients, max_tensor = on_after_backward(model, optimizer, logger)

        if not valid_gradients:
            print("=============== NAN Detected, Saving ckpt for Debugging... ===============")
            print("Max Img Value: %f, %f, %f" % (img.max().item(), img2.max().item(), torch.sum(mask).item()))

            for p in model.parameters():
                assert torch.sum(torch.isnan(p.data)) == 0
                assert torch.sum(torch.isinf(p.data)) == 0

            print("No nan Value in Model paramters found")

            save_checkpoint(config, 99999999, model.module, 0., optimizer, lr_scheduler, logger)
            import pickle
            with open(os.path.join(config.OUTPUT, 'debug_input_gpu{}.pkl'.format(gpu)), 'wb') as f:
                pickle.dump({
                    'img1_imagenetaug': img1_imagenetaug.cpu(),
                    'img2_imagenetaug': img2_imagenetaug.cpu(),
                    'mask_imagenetaug': mask_imagenetaug.cpu(),
                    'mask_sup': mask_imagenetaug_sup.cpu()}, f)
            raise NotImplementedError()

            # import pickle
            # debug_folder = '/home/shengjie/Documents/MultiFlow/SimMIMDenseMatch/checkpoints/simmim_pretrain/AblateCoaseCorr/'
            # with open(os.path.join(debug_folder, 'nopvt_imageaug_psz32_bz64', 'debug_input_gpu0.pkl'), 'rb') as f:
            #     debug_in = pickle.load(f)
            #     img1_imagenetaug = debug_in['img1_imagenetaug']
            #     img2_imagenetaug = debug_in['img2_imagenetaug']
            #     mask_imagenetaug = debug_in['mask_imagenetaug']
            #     mask_imagenetaug_sup = debug_in['mask_sup']
            #
            # state_dict = torch.load(os.path.join(debug_folder, 'nopvt_imageaug_psz32_bz64', 'ckpt_epoch_99999999.pth'))
            # incompactible = model.module.load_state_dict(state_dict['model'], strict=True)
            #
            # img1_imagenetaug = img1_imagenetaug.cuda(non_blocking=True)
            # img2_imagenetaug = img2_imagenetaug.cuda(non_blocking=True)
            # mask_imagenetaug = mask_imagenetaug.cuda(non_blocking=True)
            # mask_imagenetaug_sup = mask_imagenetaug_sup.cuda(non_blocking=True)
            #
            # with torch.no_grad():
            #     loss, x_rec = model(img1_imagenetaug, mask_imagenetaug, img2_imagenetaug, mask_imagenetaug_sup)
            # a = 1

        if writer is not None:
            writer.add_scalar('loss', loss, num_steps * epoch + idx)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], num_steps * epoch + idx)
            writer.add_scalar('maxtensor', max_tensor, num_steps * epoch + idx)

        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        loss_meter.update(loss.item(), img.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB\t'
                f'maxparam {max_tensor.item():.2f}')

        if (idx % config.PRINT_FREQ == 0) and (writer is not None):
            img_vls = img * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view(
                [1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view(
                [1, 3, 1, 1]).cuda().float()
            rec_vls = x_rec * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view(
                [1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view(
                [1, 3, 1, 1]).cuda().float()
            img2_vls = img2 * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view(
                [1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view(
                [1, 3, 1, 1]).cuda().float()

            b, _, h, w = img.shape

            vls1 = tensor2rgb(img_vls)
            vls2 = tensor2rgb(img2_vls)
            vls3 = tensor2rgb(rec_vls)
            vls4 = tensor2disp(F.interpolate(mask.unsqueeze(1).float(), [h, w]), vmax=1, viewind=0)
            vls5 = tensor2disp(F.interpolate(mask_sup.unsqueeze(1).float(), [h, w]), vmax=1, viewind=0)
            vls = np.concatenate([vls1, vls2, vls3, vls4, vls5], axis=0)

            writer.add_image('visualization', (torch.from_numpy(vls).float() / 255).permute([2, 0, 1]), num_steps * epoch + idx)

        if idx == num_steps:
            break

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    args, config = parse_option()

    config.defrost()
    config.DATA.IMG_SIZE = (160, 192)  # 192
    config['DATA']['MASK_RATIO_SCANNET'] = 0.9
    config['MODEL']['SWIN']['PATCH_SIZE'] = 2
    config['MODEL']['VIT']['PATCH_SIZE'] = 2
    config['DATA']['MASK_PATCH_SIZE'] = args.mask_patch_size
    config.freeze()

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * 2 * args.adjustscaler # 4e-4
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * 2 * args.adjustscaler # 2e-6
    linear_scaled_min_lr = config.TRAIN.MIN_LR * 2 * args.adjustscaler # 2e-5

    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    args.world_size = torch.cuda.device_count()
    args.dist_url = args.dist_url.rstrip('1235') + str(np.random.randint(2000, 3000, 1).item())
    mp.spawn(main, nprocs=args.world_size, args=(config, args, ))