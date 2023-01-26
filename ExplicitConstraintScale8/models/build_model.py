import torch
import torch.nn as nn
from copy import deepcopy
from ExplicitConstraintScale8.loftr import LoFTR, default_cfg
from ExplicitConstraintScale8.models.dkm import DFN, RRB, GP

class Wrapper(torch.nn.Module):
    def __init__(self, loftr):
        super().__init__()
        # Misc
        self.loftr = loftr

    def forward(self, x, mask, x2=None, masksup=None):
        return self.loftr(x, mask, x2, masksup)

    def extrac_feature(self, x, x2):
        return self.loftr.extrac_feature(x, x2)

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

    dfn_dim = 384
    feat_dim = 256

    coordinate_decoder = DFN(
        internal_dim=dfn_dim,
        feat_input_modules=nn.ModuleDict(
            {
                "8": nn.Conv2d(320, feat_dim, 1, 1),
            }
        ),
        rrb_d_dict=nn.ModuleDict(
            {
                "8": RRB(576, dfn_dim),
            }
        ),
        terminal_module=nn.ModuleDict(
            {
                "8": nn.Conv2d(dfn_dim, 2, 1, 1, 0),
            }
        ),
    )

    gp = GP(gp_dim=320)

    loftr = LoFTR(config=_default_cfg, gp=gp, embedding_decoder=coordinate_decoder)
    return Wrapper(loftr)
