# backbone/timm/models/resnet_ibn_custom.py

import torch
import torch.nn as nn
import math
from backbone.ibnnet.modules import IBN  # ✅ IBN 모듈 경로 정확히 지정


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=4, cardinality=32, ibn=False):
        super(Bottleneck_IBN, self).__init__()
        D = int(math.floor(planes * (base_width / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(D * C)  # ✅ IBN 삽입
            self.use_relu1 = False  # ❌ ReLU 생략
        else:
            self.bn1 = nn.BatchNorm2d(D * C)
            self.use_relu1 = True

        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)

        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.use_relu1:
            out = self.relu(out)  # ✅ IBN이면 생략됨

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_IBN_Custom(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3], base_width=4, cardinality=32, num_classes=1000):
        super().__init__()
        block = Bottleneck_IBN

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, ibn=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, ibn=False)  # ❌ Layer4에는 IBN X

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 512 * block.expansion
        self.fc = nn.Identity()  # fc 생략 (Siamese에서 projector 사용)

    def _make_layer(self, block, planes, blocks, stride=1, ibn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, ibn=ibn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=4, cardinality=32, ibn=ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


# 외부 호출용 함수
def resnext50_ibn_custom(pretrained=False):
    model = ResNet_IBN_Custom()
    if pretrained:
        from timm import create_model
        print(" Loading pretrained weights from timm's resnext50_32x4d.a1_in1k")
        ref = create_model("resnext50_32x4d.a1_in1k", pretrained=True)
        ref_dict = ref.state_dict()
        model_dict = model.state_dict()
        matched = {k: v for k, v in ref_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)
    return model
