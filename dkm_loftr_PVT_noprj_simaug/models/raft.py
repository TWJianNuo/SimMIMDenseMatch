import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dkm_loftr_PVT_noprj_simaug.models.update import BasicUpdateBlock
from dkm_loftr_PVT_noprj_simaug.models.extractor import BasicEncoder
from tools.tools import coords_grid, coords_gridN, unnorm_coords

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, iters):
        super(RAFT, self).__init__()
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=False)
        self.update_block = BasicUpdateBlock()

        self.iters = iters

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask, confidence, scale):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        scale = int(scale)
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)

        up_confidence = F.unfold(confidence, [3, 3], padding=1)
        up_confidence = up_confidence.view(N, 1, 9, 1, 1, H, W)

        mask = torch.softmax(mask * up_confidence, dim=2)

        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_flow = up_flow.reshape(N, 2, scale*H, scale*W)
        return up_flow

    def forward(self, image1, flow_init, corr_fn):
        """ Estimate optical flow between pair of frames """
        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)


        # Flow Init
        N, _, h, w = flow_init.shape
        _, _, H, W = image1.shape
        device = flow_init.device
        assert h == H // 8
        assert w == W // 8

        coords0 = coords_gridN(N, H//8, W//8, device=device)
        coords1 = flow_init.detach()

        # Convert to RAFT Coordinate System
        # [-1 + 1/h, 1 - 1/h] -> [-1, 1]
        coords1_x, coords1_y = torch.split(coords1, 1, 1)
        coords1_x = coords1_x * (w / (w - 1))
        coords1_y = coords1_y * (h / (h - 1))
        coords1 = torch.cat([coords1_x, coords1_y], dim=1)

        coords0_x, coords0_y = torch.split(coords0, 1, 1)
        coords0_x = coords0_x * (w / (w - 1))
        coords0_y = coords0_y * (h / (h - 1))
        coords0 = torch.cat([coords0_x, coords0_y], dim=1)
        # x in [-1, 1]; y in [0, h-1]; y = (x + 1) / 2 * (h - 1)

        # Convert to DKM Coordinate System
        coords0_scale1 = coords_gridN(N, H, W, device=device)

        flow_predictions = []
        for itr in range(self.iters):
            coords1 = coords1.detach()
            corr = corr_fn(unnorm_coords(coords1, h, w))  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow, confidence = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            flow_up = self.upsample_flow(coords1 - coords0, up_mask, confidence, scale=8)

            flow_predictions.append(coords0_scale1 + flow_up)

        return flow_predictions