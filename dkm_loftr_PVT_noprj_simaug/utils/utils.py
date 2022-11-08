import torch
import torch.nn.functional as F
import numpy as np
from tools.tools import InputPadder, coords_gridN

def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch

def to_cuda_scannet(data_blob):
    for k in data_blob['rgbs'].keys():
        data_blob['rgbs'][k] = data_blob['rgbs'][k].cuda()

    for k in data_blob['flowgts'].keys():
        flow, val = data_blob['flowgts'][k]
        flow = flow.cuda()
        val = val.cuda()
        data_blob['flowgts'][k] = [flow, val]

    return data_blob

def preprocess(data_blob, relfrmin, orgfrm):
    padder = InputPadder(data_blob['rgbs'][orgfrm].shape, mode='leftend', ds=32)

    # To Cuda
    data_blob = to_cuda(data_blob)

    # Preprocess
    device = data_blob['rgbs'][orgfrm].device
    mean = torch.Tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1]).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1]).to(device)

    query = list()
    support = list()

    batch_size = data_blob['rgbs'][orgfrm].shape[0]
    for b in range(batch_size):
        for frm in relfrmin:
            rgb = data_blob['rgbs'][frm][b:b+1]
            rgb = (rgb / 255.0 - mean) / std
            rgb, = padder.pad(rgb)

            if frm == orgfrm:
                query = query + [rgb] * (len(relfrmin) - 1)
            else:
                support.append(rgb)

    query = torch.cat(query, dim=0)
    support = torch.cat(support, dim=0)

    batch = {'query': query, 'support': support}
    data_blob['batch'] = batch
    return data_blob



def postprocess(data_blob, flow_predictions, relfrmin, orgfrm):
    padder = InputPadder(data_blob['rgbs'][orgfrm].shape, mode='leftend', ds=32)
    _flow_predictions = dict()

    N, _, H, W = data_blob['batch']['query'].shape
    device = data_blob['batch']['query'].device

    for key in [8, 4, 2, 1]:
        _flow_predictions[key] = dict()
        for x in ['dense_flow', 'dense_certainty']:
            pr = flow_predictions[key][x]
            if x == 'dense_flow':
                _, _, h, w = pr.shape
                coords0 = coords_gridN(N, h, w, device=device)
                pr = pr - coords0

                pr_x, pr_y = torch.split(pr, 1, dim=1)
                pr_x = pr_x / 2 * w
                pr_y = pr_y / 2 * h
                pr = torch.cat([pr_x, pr_y], dim=1)

                pr = torch.nn.functional.interpolate(pr, (H, W), mode='bilinear', align_corners=True) * (H / h)
            else:
                pr = torch.nn.functional.interpolate(pr, (H, W), mode='bilinear', align_corners=True)

            _flow_predictions[key][x] = padder.unpad(pr)
    return _flow_predictions

def postprocess_mega(data_blob, flow_predictions):
    padder = InputPadder(data_blob['query'].shape, mode='leftend', ds=16)
    _flow_predictions = dict()

    N, _, H, W = data_blob['query'].shape
    device = data_blob['query'].device

    for key in [8, 4, 2, 1]:
        _flow_predictions[key] = dict()
        for x in ['dense_flow', 'dense_certainty']:
            pr = flow_predictions[key][x]
            if x == 'dense_flow':
                _, _, h, w = pr.shape
                coords0 = coords_gridN(N, h, w, device=device)
                pr = pr - coords0
                pr = torch.nn.functional.interpolate(pr, (H, W), mode='bilinear', align_corners=True)

                coords0_scale1 = coords_gridN(N, H, W, device=device)
                pr = coords0_scale1 + pr
            else:
                pr = torch.nn.functional.interpolate(pr, (H, W), mode='bilinear', align_corners=True)

            _flow_predictions[key][x] = padder.unpad(pr)
    return _flow_predictions