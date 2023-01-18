import torch
from copy import deepcopy
from DKMResnetLoFTRPreTrainNoPVT.loftr import LoFTR, default_cfg, LoFTRBS
# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.

class Wrapper(torch.nn.Module):
    def __init__(self, loftr):
        super().__init__()
        # Misc
        self.loftr = loftr

    def forward(self, x, mask, x2=None, masksup=None):
        return self.loftr(x, mask, x2, masksup)

def DKMv2(uselinearattention=False):
    h, w = 384, 512
    # Init LoFTR
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['resnetfpn']['IMG_SIZE'] = (h, w)
    _default_cfg['resnetfpn']['two_bn'] = True
    _default_cfg['coarse']['d_model'] = 320

    if uselinearattention:
        _default_cfg['coarse']['attention'] = 'linear'
    else:
        _default_cfg['coarse']['attention'] = 'full'

    loftr = LoFTR(config=_default_cfg)
    return Wrapper(loftr)

def DKMv2Bs():
    h, w = 384, 512
    # Init LoFTR
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['resnetfpn']['IMG_SIZE'] = (h, w)
    _default_cfg['coarse']['d_model'] = 320
    _default_cfg['coarse']['attention'] = 'full'

    loftr = LoFTRBS(config=_default_cfg)
    return Wrapper(loftr)

