from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4
from .resnet_swin_fpn import ResNetSwinFPN

def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        return ResNetSwinFPN(config['resnetfpn'])
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
