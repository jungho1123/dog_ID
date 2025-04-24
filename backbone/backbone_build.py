# backbone/backbone_build.py

def get_backbone(name: str, pretrained: bool = True):#IBN 삽입된 resnext50_32x4d.a1_in1k
    if name == "resnext50_ibn_custom": 
        from backbone.timm.models.resnet_ibn_custom import resnext50_ibn_custom
        return resnext50_ibn_custom(pretrained=pretrained)

    elif name == "resnext50_plain":#	IBN 없이 순수 resnext50_32x4d.a1_in1k
        from timm import create_model 
        return create_model("resnext50_32x4d.a1_in1k", pretrained=pretrained, num_classes=0)

    elif name == "seresnext50_ibn_custom": #IBN 삽입된 seresnext50_32x4d.racm_in1k
        from backbone.timm.models.seresnet_ibn_custom import seresnext50_ibn_custom
        return seresnext50_ibn_custom(pretrained=pretrained)

    elif name == "seresnext50_plain": #IBN 없이 순수 seresnext50_32x4d.racm_in1k
        from timm import create_model
        return create_model("seresnext50_32x4d.racm_in1k", pretrained=pretrained, num_classes=0)

    else:
        raise ValueError(f"❌ Unknown backbone: {name}")
