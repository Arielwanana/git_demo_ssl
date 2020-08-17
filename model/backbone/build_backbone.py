from model.backbone import resnet

def build_backbone(backbone):
    if backbone == 'resnet18':
        return resnet.resnet18()

    else:
        raise NotImplementedError