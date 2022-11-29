import sys, os, time
sys.path.append('core')
import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import ConcatDataset

from loguru import logger

from tools.tools import read_split, coords_gridN, DistributedSamplerNoEvenlyDivisible
from tools.evaluator import compute_flow_metrics_mega
from PVTLoFTRMIM.DKMResnetLoFTRPreTrain.utils.utils import postprocess_mega, to_cuda
from PVTLoFTRMIM.DKMResnetLoFTRPreTrain.datasets.megadepth import MegadepthBuilder

@torch.no_grad()
def validate_megadepth(model, args, steps, writer=None, group=None):
    """ In Validation, random sample N Images to mimic a real test set situation  """
    model.eval()

    entries = read_split("mega_test")
    mega = MegadepthBuilder(data_root=args.data_root)
    megadepth_test = mega.build_scenes(
        split="test_loftr", min_overlap=0.35, ht=480, wt=640, shake_t=0
    )
    megadepth_test = ConcatDataset(megadepth_test)
    megadepth_test = torch.utils.data.Subset(megadepth_test, entries)

    measurements_scale = dict()
    for scale in args.eval_scale:
        measurements_scale[scale] = torch.zeros(5).cuda(device=args.gpu)

    sampler = DistributedSamplerNoEvenlyDivisible(megadepth_test, shuffle=False)
    val_loader = DataLoader(
        megadepth_test, batch_size=dist.get_world_size(), sampler=sampler,
        pin_memory=False, shuffle=False, num_workers=2, drop_last=False
    )
    val_loader = iter(val_loader)

    for val_id, data_blob in enumerate(tqdm.tqdm(val_loader, disable=True)):
        data_blob = to_cuda(data_blob)

        if args.eval_resolution:
            for key in ['query', 'support']:
                    images = data_blob[key]
                    h, w = args.eval_resolution
                    _, _, fullscaleh, fullscalew = images.shape

                    images = F.interpolate(images, (h, w), mode='bilinear', align_corners=True)
                    data_blob[key] = images

            for key in ['query_mask', 'support_mask']:
                masks = data_blob[key]
                h, w = args.eval_resolution
                b, maskh, maskw = images.shape[0], int(h / 8), int(w / 8)

                masks = F.interpolate(masks.unsqueeze(1), (maskh, maskw), mode='nearest').view(b, maskh, maskw)
                data_blob[key] = masks

        flow_predictions = model(data_blob)

        flow_predictions_postprocessed = postprocess_mega(data_blob, flow_predictions)
        for scale in args.eval_scale:
            flow_pr = flow_predictions_postprocessed[scale]['dense_flow']

            if args.eval_resolution:
                coords0 = coords_gridN(flow_pr.shape[0], h, w, device=flow_pr.device)
                flow_pr = flow_pr - coords0
                flow_pr = F.interpolate(flow_pr, (int(fullscaleh), int(fullscalew)), mode='bilinear', align_corners=True)
                flow_pr = flow_pr + coords_gridN(flow_pr.shape[0], fullscaleh, fullscalew, device=flow_pr.device)

            sub_measurements = compute_flow_metrics_mega(
                data_blob["query_depth"],
                data_blob["support_depth"],
                data_blob["T_1to2"],
                data_blob["K1"],
                data_blob["K2"],
                flow_pr,
            )

            measurements_scale[scale] += sub_measurements

    for k in measurements_scale.keys():
        dist.all_reduce(tensor=measurements_scale[k], op=dist.ReduceOp.SUM, group=group)

    px1_scale = dict()
    for scale in args.eval_scale:
        evaluated_pixel_num = 0
        px1 = 0

        measurements = measurements_scale[scale]
        evaluated_pixel_num += measurements[4]
        px1 += measurements[1]
        measurements[0:4] = measurements[0:4] / measurements[4]
        measurements = {'epe': measurements[0].item(), 'px1': measurements[1].item(), 'px5': measurements[2].item(), 'px8': measurements[3].item()}

        px1 = px1 / evaluated_pixel_num
        writer.add_scalar('Eval_MegaDepth_scale_{}'.format(scale), measurements['px1'], steps)
        px1_scale[scale] = px1

        measurements_scale[scale] = measurements
    return measurements_scale, px1_scale[1]

class Validation():
    def __init__(self, project_root):
        self.project_root = project_root
        self.max_px_scannet = 0
        self.max_px_mega = 0

    def apply_eval(self, model, writer, steps, args, save=True, group=None):
        results_scales, px1 = validate_megadepth(model, args, writer=writer, steps=steps, group=group)
        torch.cuda.synchronize()

        if args.gpu == 0:
            for scale in args.eval_scale:
                result = results_scales[scale]
                logger.info("MegaDepth Scale %d, Metric Epe: %.3f, px1: %.3f, px5: %.3f, px8: %.3f" % (scale, result['epe'], result['px1'], result['px5'], result['px8']))
                logger.info("=============================================")

                if (px1 > self.max_px_mega) and save and (scale == 1):
                    self.max_px_mega = px1
                    PATH = os.path.join(self.project_root, 'checkpoints/MIMAug', args.experiment_name, 'maxa1_mega.pth')
                    if isinstance(model, (DataParallel, DistributedDataParallel)):
                        model = model.module
                    torch.save(model.state_dict(), PATH)
                    logger.info("saving checkpoints to %s" % PATH)

        model.train()
