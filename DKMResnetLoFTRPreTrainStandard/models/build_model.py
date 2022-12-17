import torch
from copy import deepcopy
from DKMResnetLoFTRPreTrainStandard.loftr import LoFTR, default_cfg
# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.

class Wrapper(torch.nn.Module):
    def __init__(self, loftr):
        super().__init__()
        # Misc
        self.loftr = loftr


def DKMv2(resolution="extrasmall", pvt_depth=4):
    if resolution == "low":
        h, w = 384, 512
    elif resolution == "high":
        h, w = 480, 640
    elif resolution == "extrasmall":
        h, w = 240, 320

    h, w = 384, 512
    # Init LoFTR
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['resnetfpn']['IMG_SIZE'] = (h, w)
    _default_cfg['resnetfpn']['pvt_depth'] = pvt_depth
    _default_cfg['coarse']['d_model'] = 320
    loftr = LoFTR(config=_default_cfg)
    return Wrapper(loftr)
