# updated SiameseNetwork with extract(normalize=True) for flexibility
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.backbone_build import get_backbone

class SiameseNetwork(nn.Module):
    def __init__(self, backbone_name: str, in_features: int, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone = get_backbone(backbone_name, pretrained=pretrained)

        self.projector = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, feature_dim)
        )

    def extract(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.backbone(x)
        x = self.projector(x)
        return F.normalize(x, dim=1) if normalize else x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z1 = self.extract(x1, normalize=True)
        z2 = self.extract(x2, normalize=True)
        return z1, z2
