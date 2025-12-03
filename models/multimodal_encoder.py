
"""Multi-modal CNN encoder and fusion module."""

from typing import Dict, Tuple

import torch
from torch import nn
from torchvision import models

from config import CONFIG


class ConvBackbone(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", out_dim: int = 256):
        super().__init__()
        if backbone_name == "resnet18":
            net = models.resnet18(weights=None)
            feat_dim = net.fc.in_features
            modules = list(net.children())[:-2]
            self.backbone = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        self.proj = nn.Conv2d(feat_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.proj(feat)
        return feat


class MultiModalEncoder(nn.Module):
    """Encodes RGB, thermal, and depth frames and fuses them channel-wise."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.rgb_backbone = ConvBackbone(CONFIG.model.cnn_backbone, out_dim=d_model)
        self.th_backbone = ConvBackbone(CONFIG.model.cnn_backbone, out_dim=d_model)
        self.d_backbone = ConvBackbone(CONFIG.model.cnn_backbone, out_dim=d_model)

        self.fuse = nn.Sequential(
            nn.Conv2d(3 * d_model, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        f_rgb = self.rgb_backbone(rgb)
        f_th = self.th_backbone(thermal)
        f_d = self.d_backbone(depth)
        fused = torch.cat([f_rgb, f_th, f_d], dim=1)
        fused = self.fuse(fused)
        return fused  # (B, d_model, H, W)
