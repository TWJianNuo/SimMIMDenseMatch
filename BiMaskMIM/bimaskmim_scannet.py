# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------
from __future__ import print_function, division
import os, sys, inspect, time, tqdm
project_root = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_root)

import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader, DistributedSampler

from pprint import pprint
from config import get_config
from logger import create_logger
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from tools.tools import tensor2rgb, tensor2disp
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper

from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from BiMaskMIM.models.build_model import DKMv2
from BiMaskMIM.datasets.scannet import build_loader_scannet

try:
    # noinspection PyUnresolvedReferences
    import apex
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
    parser.add_argument('--uselinearattention', action='store_true')

    parser.add_argument('--mask-patch-size', type=int, default=32)
    parser.add_argument('--mask-ratio-view1', type=float, default=0.75)
    parser.add_argument('--mask-ratio-view2', type=float, default=0.25)
    parser.add_argument('--minoverlap', type=float, default=0.5)

    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args)

    return args, config

def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret

def main(config):
    scannet = build_loader_scannet(config)
    logger.info(f"The Length of ScanNet is %d {scannet.__len__()}")

    model = DKMv2(uselinearattention=args.uselinearattention)
    model.cuda()

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
        model = apex.parallel.convert_syncbn_model(model)
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

    if config.LOCAL_RANK == 0:
        logger.info(f'Create Summary Writer')
        writer = SummaryWriter(config.OUTPUT, flush_secs=30)
    else:
        writer = None

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        scannet_sampler = DistributedSampler(scannet, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        scannet_sampler.set_epoch(int(epoch))
        data_loader_train_scannet = DataLoader(scannet, config.DATA.BATCH_SIZE, sampler=scannet_sampler, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=True, collate_fn=collate_fn)
        data_loader_train_scannet = iter(data_loader_train_scannet)

        train_one_epoch(config, model, data_loader_train_scannet, optimizer, epoch, lr_scheduler, writer)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, model, data_loader_scannet, optimizer, epoch, lr_scheduler, writer):
    model.train()
    optimizer.zero_grad()

    num_steps = 5000
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx in range(num_steps):
        scannet_batch = next(data_loader_scannet)

        img1_scannet, mask1_scannet, img2_scannet, mask2_scannet = scannet_batch

        img1_scannet = img1_scannet.cuda(non_blocking=True)
        img2_scannet = img2_scannet.cuda(non_blocking=True)
        mask1_scannet = mask1_scannet.cuda(non_blocking=True)
        mask2_scannet = mask2_scannet.cuda(non_blocking=True)

        loss, x_rec = model(img1_scannet, mask1_scannet, img2_scannet, mask2_scannet)

        if writer is not None:
            writer.add_scalar('loss', loss, num_steps * epoch + idx)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], num_steps * epoch + idx)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), img1_scannet.size(0))
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
                f'mem {memory_used:.0f}MB')

        if (idx % int(config.PRINT_FREQ * 5) == 0) and (writer is not None):
            img1_vls = img1_scannet * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view(
                [1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view(
                [1, 3, 1, 1]).cuda().float()
            img2_vls = img2_scannet * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view(
                [1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view(
                [1, 3, 1, 1]).cuda().float()

            rec_vls = x_rec * torch.from_numpy(np.array(IMAGENET_DEFAULT_STD)).view(
                [1, 3, 1, 1]).cuda().float() + torch.from_numpy(np.array(IMAGENET_DEFAULT_MEAN)).view(
                [1, 3, 1, 1]).cuda().float()

            b, _, h, w = img1_scannet.shape

            vls1 = tensor2rgb(img1_vls)
            vls2 = tensor2rgb(img2_vls)
            vls3 = tensor2rgb(rec_vls)
            vls4 = tensor2disp(F.interpolate(mask1_scannet.unsqueeze(1).float(), [h, w]), vmax=1, viewind=0)
            vls5 = tensor2disp(F.interpolate(mask2_scannet.unsqueeze(1).float(), [h, w]), vmax=1, viewind=0)
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

    config['MODEL']['VIT']['PATCH_SIZE'] = 2
    config['DATA']['DATA_PATH'] = args.data_path
    config['DATA']['MASK_RATIO_VIEW1'] = args.mask_ratio_view1
    config['DATA']['MASK_RATIO_VIEW2'] = args.mask_ratio_view2
    config['DATA']['MASK_PATCH_SIZE'] = args.mask_patch_size
    config['DATA']['MINOVERLAP'] = args.minoverlap
    config.freeze()

    config_dict = {
        'PATCH_SIZE': 2,
        'MASK_RATIO_VIEW1': args.mask_ratio_view1,
        'MASK_RATIO_VIEW2': args.mask_ratio_view2,
        'MASK_PATCH_SIZE': args.mask_patch_size,
        'MINOVERLAP': args.minoverlap
    }

    pprint(config_dict)

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    linear_scaled_lr = config.TRAIN.BASE_LR * 2
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * 2
    linear_scaled_min_lr = config.TRAIN.MIN_LR * 2
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
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)