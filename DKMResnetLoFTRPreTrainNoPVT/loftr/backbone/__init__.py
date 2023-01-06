from .resnet_swin_fpn import ResNetSwinFPN

def build_backbone(config):
    return ResNetSwinFPN(config['resnetfpn'])
