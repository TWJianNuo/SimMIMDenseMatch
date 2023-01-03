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

from config import get_config
from logger import create_logger
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from tools.tools import tensor2rgb, tensor2disp
from data.data_simmim_mega import build_loader_imagenet
from data.data_simmim_scannet_twoview import build_loader_imagenetaug
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper

from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from DKMResnetLoFTRPreTrainStandard.models.build_model import DKMv2

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
    parser.add_argument('--data-path-scannet', type=str, help='path to dataset scannet')
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
    # logger = None
    # data_loader_train = build_loader_imagenet(config, logger, split="train")
    # data_loader_train_scannet = build_loader_scannet(config, logger)

    imagenet = build_loader_imagenet(config, logger, split="train")
    imagenetaug = build_loader_imagenetaug(config, logger)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = DKMv2(pvt_depth=4, resolution='extrasmall')
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

    if config.LOCAL_RANK == 0:
        logger.info(f'Create Summary Writer')
        writer = SummaryWriter(config.OUTPUT, flush_secs=30)
    else:
        writer = None

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        imagenet_sampler = DistributedSampler(imagenet, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        imagenet_sampler.set_epoch(int(epoch))
        data_loader_train = DataLoader(imagenet, config.DATA.BATCH_SIZE, sampler=imagenet_sampler, num_workers=config.DATA.NUM_WORKERS,
                                       pin_memory=True, drop_last=True, collate_fn=collate_fn)
        data_loader_train = iter(data_loader_train)

        imagenetaug_sampler = DistributedSampler(imagenetaug, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        imagenetaug_sampler.set_epoch(int(epoch))
        data_loader_train_imagenetaug = DataLoader(imagenetaug, config.DATA.BATCH_SIZE, sampler=imagenetaug_sampler, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=True, collate_fn=collate_fn)
        data_loader_train_imagenetaug = iter(data_loader_train_imagenetaug)
        # print("=================")
        # print(len(scannet_sampler), len(imagenet_sampler))
        # data_loader_train.sampler.set_epoch(epoch)
        # data_loader_train_scannet.sampler.set_epoch(epoch)
        train_one_epoch(config, model, data_loader_train, data_loader_train_imagenetaug, optimizer, epoch, lr_scheduler, writer)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, logger)
        # eval(model, n_iter_per_epoch * (epoch + 1), config, writer)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

@torch.no_grad()
def eval(model, eval_step, config, writer):
    model.eval()
    imagenet_val = build_loader_imagenet(config, logger, split="val", drop_last=True)

    imagenet_sampler = DistributedSampler(imagenet_val, num_replicas=dist.get_world_size(), rank=dist.get_rank(),
                                          shuffle=True)
    data_loader = DataLoader(imagenet_val, config.DATA.BATCH_SIZE, sampler=imagenet_sampler,
                             num_workers=config.DATA.NUM_WORKERS,
                             pin_memory=True, drop_last=True, collate_fn=collate_fn)

    logger.info('Dataset Length {}, BatchSize {}'.format(len(data_loader), config.DATA.BATCH_SIZE))

    loss_l1 = 0
    tot_pxl = 0
    for _, (img, mask) in enumerate(tqdm.tqdm(data_loader)):
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        loss, rgb_recon = model(img, mask)

        patch_sizeh = int(img.shape[2] / mask.shape[1])
        patch_sizew = int(img.shape[3] / mask.shape[2])
        assert patch_sizeh == patch_sizew
        mask = mask.repeat_interleave(patch_sizeh, 1).repeat_interleave(patch_sizew, 2).unsqueeze(1).contiguous()

        in_chans = 3
        loss_recon = F.l1_loss(img, rgb_recon, reduction='none')

        loss_l1 += (loss_recon * mask).sum() / in_chans
        tot_pxl += mask.sum()

    loss_l1 = loss_l1 / tot_pxl
    if config.LOCAL_RANK == 0:
        logger.info('Eval L1 {} at step {}'.format(loss_l1.item(), eval_step))
        writer.add_scalar('Eval/L1', loss_l1, eval_step)

def train_one_epoch(config, model, data_loader, data_loader_imagenetaug, optimizer, epoch, lr_scheduler, writer):
    model.train()
    optimizer.zero_grad()

    num_steps = 5000
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx in range(num_steps):
        imagenet_batch = next(data_loader)
        imagenetaug_batch = next(data_loader_imagenetaug)

        img_imagenet, mask_imagenet = imagenet_batch
        img1_imagenetaug, mask_imagenetaug, img2_imagenetaug, _, mask_scannet_sup, _ = imagenetaug_batch

        img_imagenet = img_imagenet.cuda(non_blocking=True)
        mask_imagenet = mask_imagenet.cuda(non_blocking=True)

        img1_imagenetaug = img1_imagenetaug.cuda(non_blocking=True)
        img2_imagenetaug = img2_imagenetaug.cuda(non_blocking=True)
        mask_imagenetaug = mask_imagenetaug.cuda(non_blocking=True)
        mask_scannet_sup = mask_scannet_sup.cuda(non_blocking=True)

        loss1, _ = model(img_imagenet, mask_imagenet)
        loss2, x_rec = model(img1_imagenetaug, mask_imagenetaug, img2_imagenetaug, mask_scannet_sup)

        loss = (loss1 + loss2) / 2

        img = img1_imagenetaug
        img2 = img2_imagenetaug
        mask = mask_imagenetaug
        masksup = mask_scannet_sup

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
                f'mem {memory_used:.0f}MB')

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
            vls5 = tensor2disp(F.interpolate(masksup.unsqueeze(1).float(), [h, w]), vmax=1, viewind=0)
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
    config['DATA']['MASK_RATIO'] = 0.75
    config['DATA']['MASK_RATIO_SCANNET'] = 0.9
    config['MODEL']['SWIN']['PATCH_SIZE'] = 2
    config['MODEL']['VIT']['PATCH_SIZE'] = 2
    config.freeze()

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

    # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = config.TRAIN.BASE_LR * dist.get_world_size() * args.adjustscaler
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * dist.get_world_size() * args.adjustscaler
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * dist.get_world_size() * args.adjustscaler
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