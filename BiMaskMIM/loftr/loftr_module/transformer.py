import copy
import torch
import torch.nn as nn
from .linear_attention import FullAttention, LinearAttention
from loguru import logger


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        logger.info("Linear Attention") if attention == 'linear' else logger.info("Full Attention")

    def forward(self, x, source, x_mask=None, source_mask=None, detach_left=False, detach_right=False):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        if detach_left:
            with torch.no_grad():
                query = self.q_proj(query).view(bs, -1, self.nhead, self.dim).float()  # [N, L, (H, D)]
        else:
            query = self.q_proj(query).view(bs, -1, self.nhead, self.dim).float()  # [N, L, (H, D)]

        if detach_right:
            with torch.no_grad():
                key = self.k_proj(key).view(bs, -1, self.nhead, self.dim).float()  # [N, S, (H, D)]
                value = self.v_proj(value).view(bs, -1, self.nhead, self.dim).float()
        else:
            key = self.k_proj(key).view(bs, -1, self.nhead, self.dim).float()  # [N, S, (H, D)]
            value = self.v_proj(value).view(bs, -1, self.nhead, self.dim).float()

        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask, detach_left=detach_left, detach_right=detach_right)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        if detach_left:
            x = x.detach()

        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)
        y = x + message

        return y


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, detach_left=False, detach_right=False):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                if detach_left:
                    with torch.no_grad():
                        feat0 = layer(feat0, feat0, mask0, mask0)
                else:
                    feat0 = layer(feat0, feat0, mask0, mask0)
                if detach_right:
                    with torch.no_grad():
                        feat1 = layer(feat1, feat1, mask1, mask1)
                else:
                    feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1, detach_left, detach_right)
                feat1 = layer(feat1, feat0, mask1, mask0, detach_left, detach_right)
            else:
                raise KeyError

        return feat0, feat1

    def forward_onelayer(self, feat0, feat1, mask0=None, mask1=None, detach_left=False, detach_right=False, stlayer=0):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        st_indx = stlayer * 2
        ed_indx = (stlayer + 1) * 2
        for layer, name in zip(self.layers[st_indx: ed_indx], self.layer_names[st_indx: ed_indx]):
            if name == 'self':
                if detach_left:
                    with torch.no_grad():
                        feat0 = layer(feat0, feat0, mask0, mask0)
                else:
                    feat0 = layer(feat0, feat0, mask0, mask0)
                if detach_right:
                    with torch.no_grad():
                        feat1 = layer(feat1, feat1, mask1, mask1)
                else:
                    feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1, detach_left, detach_right)
                feat1 = layer(feat1, feat0, mask1, mask0, detach_left, detach_right)
            else:
                raise KeyError

        return feat0, feat1