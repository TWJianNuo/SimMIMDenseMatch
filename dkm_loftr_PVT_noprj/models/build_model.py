import os
from copy import deepcopy
from .dkm import *

from dkm_loftr_PVT_noprj.loftr import LoFTR, default_cfg
# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.

def DKMv2(config_simmim, h, w, version="outdoor", outputfeature='concatenated', pvt_depth=4, **kwargs):
    if outputfeature == 'concatenated':
        concatenated_dim = 640
    elif outputfeature == 'fpn':
        concatenated_dim = 320
    elif outputfeature == 'transformer':
        concatenated_dim = 320
    else:
        raise NotImplementedError()

    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    conv_refiner = nn.ModuleDict(
        {
            "8": ConvRefiner(
                concatenated_dim,
                concatenated_dim,
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
            ),
        }
    )
    decoder = Decoder(conv_refiner, scales=["8", "4"])
    # Init LoFTR
    _default_cfg = deepcopy(default_cfg)
    if version == 'indoor':
        _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    _default_cfg['resnetfpn']['IMG_SIZE'] = (h, w)
    _default_cfg['resnetfpn']['pvt_depth'] = pvt_depth
    _default_cfg['coarse']['d_model'] = 320
    loftr = LoFTR(config=_default_cfg, outputfeature=outputfeature)

    print("LoFTR Checkpoint UnLoaded")
    print("PVT Depth %d" % pvt_depth)
    matcher = RegressionMatcher(loftr=loftr, decoder=decoder, **kwargs)
    return matcher

