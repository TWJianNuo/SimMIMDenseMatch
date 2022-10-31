from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4


def build_backbone(config, config_simmim):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 2):
            return ResNetFPN_8_2(config['resnetfpn'], config_simmim)
        elif config['resolution'] == (16, 4):
            return ResNetFPN_16_4(config['resnetfpn'], config_simmim)
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
