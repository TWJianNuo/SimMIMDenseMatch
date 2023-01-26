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
        rrb_d_dict,
        terminal_module=None,
    ):
        super().__init__()
        self.internal_dim = internal_dim
        self.feat_input_modules = feat_input_modules
        self.rrb_d = rrb_d_dict
        self.terminal_module = (
            terminal_module if terminal_module is not None else nn.Identity()
        )
        self._scales = [int(key) for key in self.terminal_module.keys()]

    def scales(self):
        return self._scales.copy()

    def forward(self, embeddings, feats, key):
        feats = self.feat_input_modules[str(key)](feats)
        embeddings = torch.cat([feats, embeddings], dim=1)
        embeddings = self.rrb_d[str(key)](embeddings)
        pred_coord = self.terminal_module[str(key)](embeddings)
        return pred_coord


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