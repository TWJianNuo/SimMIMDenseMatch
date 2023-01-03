import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from tools.tools import get_tuple_transform_ops, RGB2Gray


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb=None,
        displacement_emb_dim=None,
        scale=None
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)

        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2,displacement_emb_dim,1,1,0)
        else:
            self.has_displacement_emb = False
        self.scale = int(scale)

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
        )
        norm = nn.BatchNorm2d(out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, x, y, flow):
        """Computes the relative refining displacement in pixels for a given image x,y and a coarse flow-field between them

        Args:
            x ([type]): [description]
            y ([type]): [description]
            flow ([type]): [description]

        Returns:
            [type]: [description]
        """
        b,c,hs,ws = x.shape
        with torch.no_grad():
            x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1), align_corners=False)
        if self.has_displacement_emb:
            query_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device="cuda"),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device="cuda"),
            )
            )
            query_coords = torch.stack((query_coords[1], query_coords[0]))
            query_coords = query_coords[None].expand(b, 2, hs, ws)
            in_displacement = flow-query_coords
            emb_in_displacement = self.disp_emb(in_displacement)
            d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
        else:  
            d = torch.cat((x, x_hat), dim=1)
        d = self.block1(d)
        d = self.hidden_blocks(d)

        certainty_displacement = self.out_conv(d)
        certainty, displacement = certainty_displacement[:, :-2], certainty_displacement[:, -2:]

        return certainty, displacement


class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x  # high, low (old, new)
        x = torch.cat([x1, x2], dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res


class RRB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)


class DFN(nn.Module):
    def __init__(
        self,
        internal_dim,
        feat_input_modules,
        pred_input_modules,
        rrb_d_dict,
        cab_dict,
        rrb_u_dict,
        use_global_context=False,
        global_dim=None,
        terminal_module=None,
        upsample_mode="bilinear",
        align_corners=False,
    ):
        super().__init__()
        if use_global_context:
            assert (
                global_dim is not None
            ), "Global dim must be provided when using global context"
        self.align_corners = align_corners
        self.internal_dim = internal_dim
        self.feat_input_modules = feat_input_modules
        self.pred_input_modules = pred_input_modules
        self.rrb_d = rrb_d_dict
        self.cab = cab_dict
        self.rrb_u = rrb_u_dict
        self.use_global_context = use_global_context
        if use_global_context:
            self.global_to_internal = nn.Conv2d(global_dim, self.internal_dim, 1, 1, 0)
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.terminal_module = (
            terminal_module if terminal_module is not None else nn.Identity()
        )
        self.upsample_mode = upsample_mode
        self._scales = [int(key) for key in self.terminal_module.keys()]

    def scales(self):
        return self._scales.copy()

    def forward(self, new_stuff, feats, old_stuff, key):
        feats = self.feat_input_modules[str(key)](feats)
        new_stuff = torch.cat([feats, new_stuff], dim=1)
        new_stuff = self.rrb_d[str(key)](new_stuff)
        new_old_stuff = self.cab[str(key)]([old_stuff, new_stuff])
        new_old_stuff = self.rrb_u[str(key)](new_old_stuff)
        preds = self.terminal_module[str(key)](new_old_stuff)
        pred_coord = preds[:, -2:]
        pred_certainty = preds[:, :-2]
        return pred_coord, pred_certainty, new_old_stuff


class GP(nn.Module):
    def __init__(
        self,
        gp_dim=64,
        basis="fourier",
    ):
        super().__init__()
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.basis = basis

    def get_local_cov(self, cov):
        K = self.covar_size
        b, h, w, h, w = cov.shape
        hw = h * w
        cov = F.pad(cov, 4 * (K // 2,))  # pad v_q
        delta = torch.stack(
            torch.meshgrid(
                torch.arange(-(K // 2), K // 2 + 1), torch.arange(-(K // 2), K // 2 + 1)
            ),
            dim=-1,
        )
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(K // 2, h + K // 2), torch.arange(K // 2, w + K // 2)
            ),
            dim=-1,
        )
        neighbours = positions[:, :, None, None, :] + delta[None, :, :]
        points = torch.arange(hw)[:, None].expand(hw, K**2)
        local_cov = cov.reshape(b, hw, h + K - 1, w + K - 1)[
            :,
            points.flatten(),
            neighbours[..., 0].flatten(),
            neighbours[..., 1].flatten(),
        ].reshape(b, h, w, K**2)
        return local_cov

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def project_to_basis(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_conv(x))
        elif self.basis == "linear":
            return self.pos_conv(x)
        else:
            raise ValueError(
                "No other bases other than fourier and linear currently supported in public release"
            )

    def get_pos_enc(self, b, c, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            )
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.project_to_basis(coarse_coords)
        return coarse_embedded_coords

    def forward(self, sim_matrix, b, h, w):
        f = self.get_pos_enc(b, None, h, w, device=sim_matrix.device)
        f = f.view([b, -1, h * w]).permute([0, 2, 1])

        conf_matrix = torch.softmax(sim_matrix, dim=2)
        gp_feats = conf_matrix @ f
        gp_feats = rearrange(gp_feats, "b (h w) d -> b d h w", h=h, w=w)

        return gp_feats


class Encoder(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet

    def forward(self, x):
        x0 = x
        b, c, h, w = x.shape
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x1 = self.resnet.relu(x)

        x = self.resnet.maxpool(x1)
        x2 = self.resnet.layer1(x)

        x3 = self.resnet.layer2(x2)

        x4 = self.resnet.layer3(x3)

        x5 = self.resnet.layer4(x4)
        feats = {32: x5, 16: x4, 8: x3, 4: x2, 2: x1, 1: x0}
        return feats
    
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            pass

class Decoder(nn.Module):
    def __init__(
        self, embedding_decoder, gp, conv_refiner, detach=False, scales="all"
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.gp = gp
        self.conv_refiner = conv_refiner
        self.detach = detach
        if scales == "all":
            self.scales = ["32", "16", "8", "4", "2", "1"]
        else:
            self.scales = scales

    def upsample_preds(self, flow, certainty, query, support):
        b, hs, ws, d = flow.shape
        b, c, h, w = query.shape

        flow = flow.permute(0, 3, 1, 2)
        certainty = F.interpolate(
            certainty, size=query.shape[-2:], align_corners=False, mode="bilinear"
        )
        flow = F.interpolate(
            flow, size=query.shape[-2:], align_corners=False, mode="bilinear"
        )
        delta_certainty, delta_flow = self.conv_refiner["1"](query, support, flow)
        flow = torch.stack(
                (
                    flow[:, 0] + delta_flow[:, 0] / (4 * w),
                    flow[:, 1] + delta_flow[:, 1] / (4 * h),
                ),
                dim=1,
            )
        return flow, certainty

    def get_placeholder_flow(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            )
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords

    def forward(self, f1, f2, sim_matrix):
        coarse_scales = self.embedding_decoder.scales()
        all_scales = self.scales
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        b = f1[1].shape[0]
        old_stuff = torch.zeros(
            b, self.embedding_decoder.internal_dim, *sizes[8], device=f1[8].device
        )
        dense_corresps = {}
        dense_certainty = 0.0

        for new_scale in all_scales:
            ins = int(new_scale)
            f1_s, f2_s = f1[ins], f2[ins]

            if ins in coarse_scales:
                bh = f1[ins].shape[2]
                bw = f1[ins].shape[3]
                new_stuff = self.gp(sim_matrix, b=b, h=bh, w=bw)
                dense_flow, dense_certainty, old_stuff = self.embedding_decoder(
                    new_stuff, f1_s, old_stuff, new_scale
                )

            if new_scale in self.conv_refiner:
                hs, ws = h // ins, w // ins
                delta_certainty, displacement = self.conv_refiner[new_scale](
                    f1_s, f2_s, dense_flow
                )
                dense_flow = torch.stack(
                    (
                        dense_flow[:, 0] + ins * displacement[:, 0] / (4 * w),
                        dense_flow[:, 1] + ins * displacement[:, 1] / (4 * h),
                    ),
                    dim=1,
                )  # multiply with scale
                dense_certainty = (
                    dense_certainty + delta_certainty
                )  # predict both certainty and displacement

            dense_corresps[ins] = {
                "dense_flow": dense_flow,
                "dense_certainty": dense_certainty
            }

            if new_scale != "1":
                dense_flow = F.interpolate(
                    dense_flow,
                    size=sizes[ins // 2],
                    align_corners=False,
                    mode="bilinear",
                )

                dense_certainty = F.interpolate(
                    dense_certainty,
                    size=sizes[ins // 2],
                    align_corners=False,
                    mode="bilinear",
                )
                if self.detach:
                    dense_flow = dense_flow.detach()
                    dense_certainty = dense_certainty.detach()
        return dense_corresps


class RegressionMatcher(nn.Module):
    def __init__(
        self,
        decoder,
        h=384,
        w=512,
    ):
        super().__init__()
        self.decoder = decoder
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)

    def train(self, mode=True):
        self.decoder.train(mode)
        self.loftr.train(mode)

    def extract_backbone_features(self, batch):
        x_q = batch["query"]
        x_s = batch["support"]
        X = torch.cat((x_q, x_s))
        feature_pyramid = self.encoder(X)
        return feature_pyramid

    def sample(
        self,
        dense_matches,
        dense_certainty,
        num=20000,
        relative_confidence_threshold=0.0,
        conf_thresh = 0.1,
    ):
        matches, certainty = (
            dense_matches.reshape(-1, 4).cpu().numpy(),
            dense_certainty.reshape(-1).cpu().numpy(),
        )
        certainty[certainty > conf_thresh] = 1
        relative_confidence = certainty / certainty.max()
        matches, certainty = (
            matches[relative_confidence > relative_confidence_threshold],
            certainty[relative_confidence > relative_confidence_threshold],
        )
        good_samples = np.random.choice(
            np.arange(len(matches)),
            size=min(num, len(certainty)),
            replace=False,
            p=certainty / np.sum(certainty),
        )
        return matches[good_samples], certainty[good_samples]

    def forward(self, batch):
        # Get Correlation Volume
        data = dict()
        data['image0'], data['image0_color'] = copy.deepcopy(batch['query']), copy.deepcopy(batch['query'])
        data['image1'], data['image1_color'] = copy.deepcopy(batch['support']), copy.deepcopy(batch['support'])

        if 'query_identifier' in batch:
            data['pair_names'] = batch['query_identifier']
        else:
            data['pair_names'] = ['None']

        if 'query_mask' in batch and 'support_mask' in batch:
            data['mask0'] = batch['query_mask']
            data['mask1'] = batch['support_mask']

        if self.training:
            data['K0'] = batch['K1']
            data['K1'] = batch['K2']
            data['T_0to1'] = batch['T_1to2']
            data['T_1to0'] = torch.linalg.inv(batch['T_1to2'])
            data['depth0'] = batch['query_depth']
            data['depth1'] = batch['support_depth']

        f_q_pyramid, f_s_pyramid, sim_matrix = self.loftr(data)

        dense_corresps = self.decoder(f_q_pyramid, f_s_pyramid, sim_matrix)

        dense_corresps.update({'data': data, 'sim_matrix': sim_matrix})
        return dense_corresps

    def forward_symmetric(self, batch):
        feature_pyramid = self.extract_backbone_features(batch)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]))
            for scale, f_scale in feature_pyramid.items()
        }
        dense_corresps = self.decoder(f_q_pyramid, f_s_pyramid)
        return dense_corresps

    def cycle_constraint(self, query_coords, query_to_support, support_to_query):
        dist = query_coords - F.grid_sample(
            support_to_query, query_to_support, mode="bilinear"
        )
        return dist

    def stable_neighbours(self, query_coords, query_to_support, support_to_query):
        qts = query_to_support
        for t in range(4):
            _qts = qts
            q = F.grid_sample(support_to_query, qts, mode="bilinear")
            qts = F.grid_sample(
                query_to_support.permute(0, 3, 1, 2),
                q.permute(0, 2, 3, 1),
                mode="bilinear",
            ).permute(0, 2, 3, 1)
        d = (qts - _qts).norm(dim=-1)
        qd = (q - query_coords).norm(dim=1)
        stabneigh = torch.logical_and(d < 1e-3, qd < 5e-3)  # Hello boy
        return q, qts, stabneigh

    def nms(self, confidence, kernel_size=5):
        return confidence * (
            confidence
            == F.max_pool2d(confidence, kernel_size, stride=1, padding=kernel_size // 2)
        )

    def match(
        self,
        im1,
        im2,
        batched=False,
        check_cycle_consistency=False,
        do_pred_in_og_res=False,
    ):
        self.train(False)
        with torch.no_grad():
            if not batched:
                b = 1
                w, h = im1.size
                w2, h2 = im2.size
                # Get images in good format
                ws = self.w_resized
                hs = self.h_resized
                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                query, support = test_transform((im1, im2))
                batch = {"query": query[None].cuda(), "support": support[None].cuda()}
            else:
                b, c, h, w = im1.shape
                b, c, h2, w2 = im2.shape
                assert w == w2 and h == h2, "For batched images we assume same size"
                batch = {"query": im1.cuda(), "support": im2.cuda()}
                hs, ws = self.h_resized, self.w_resized
            finest_scale = 1  # i will assume that we go to the finest scale (otherwise min(list(dense_corresps.keys())) also works)
            # Run matcher
            dense_corresps = self.forward(batch)
            query_to_support = dense_corresps[finest_scale]["dense_flow"].permute(
                0, 2, 3, 1
            )
            # Get certainty interpolation
            dense_certainty = dense_corresps[finest_scale]["dense_certainty"]

            if do_pred_in_og_res:  # Will assume that there is no batching going on.
                og_query, og_support = self.og_transforms((im1, im2))
                query_to_support, dense_certainty = self.decoder.upsample_preds(
                    query_to_support,
                    dense_certainty,
                    og_query.cuda()[None],
                    og_support.cuda()[None],
                )
                hs, ws = h, w
                query_to_support = query_to_support.permute(0, 2, 3, 1)
            # Create im1 meshgrid
            query_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device="cuda"),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device="cuda"),
                )
            )
            query_coords = torch.stack((query_coords[1], query_coords[0]))
            query_coords = query_coords[None].expand(b, 2, hs, ws)
            dense_certainty = dense_certainty.sigmoid()  # logits -> probs
            query_coords = query_coords.permute(0, 2, 3, 1)

            query_to_support = torch.clamp(query_to_support, -1, 1)
            if batched:
                return (
                    torch.cat((query_coords, query_to_support), dim=-1),
                    dense_certainty[:, 0],
                )
            else:
                return (
                    torch.cat((query_coords, query_to_support), dim=-1)[0],
                    dense_certainty[0, 0],
                )