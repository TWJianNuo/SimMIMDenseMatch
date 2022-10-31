import os
from copy import deepcopy

import torch
import torch.nn as nn
from .dkm import *
from .local_corr import LocalCorr
from .corr_channels import NormedCorr
from torchvision.models import resnet as tv_resnet

from dkm_loftr_multiscale_encoderonly.loftr import LoFTR, default_cfg
# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.

def DKMv2(config_simmim, version="outdoor", outputfeature='concatenated', **kwargs):
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                320,
                320,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                encoder_stride=8
            ),
            "4": ConvRefiner(
                128,
                128,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                encoder_stride=4
            )
        }
    )
    decoder = Decoder(conv_refiner, scales=["8", "4"])

    # Init LoFTR
    _default_cfg = deepcopy(default_cfg)
    if version == 'indoor':
        _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    loftr = LoFTR(config_simmim=config_simmim, config=_default_cfg, outputfeature=outputfeature)

    matcher = RegressionMatcher(loftr=loftr, decoder=decoder)
    return matcher
