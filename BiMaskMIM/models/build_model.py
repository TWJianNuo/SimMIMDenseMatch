import torch
from copy import deepcopy
from BiMaskMIM.loftr import LoFTR, default_cfg

class Wrapper(torch.nn.Module):
    def __init__(self, loftr):
        super().__init__()
        # Misc
        self.loftr = loftr

    def forward(self, x1, mask1, x2, mask2):
        return self.loftr(x1, mask1, x2, mask2)

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
