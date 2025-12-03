
"""Spatio-temporal transformer for skeleton-based behavioral modeling."""

from typing import Dict

import torch
from torch import nn

from config import CONFIG


class TemporalBehaviorEncoder(nn.Module):
    def __init__(self, num_joints: int, pose_dim: int, d_model: int):
        super().__init__()
        self.input_proj = nn.Linear(num_joints * pose_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # windows: (B, W, T, J, 2)
        B, W, T, J, D = windows.shape
        x = windows.view(B * W, T, J * D)
        x = self.input_proj(x)
        x = self.encoder(x)  # (B*W, T, d_model)
        x = x.transpose(1, 2)  # (B*W, d_model, T)
        x = self.pool(x).squeeze(-1)  # (B*W, d_model)
        x = x.view(B, W, -1)  # (B, W, d_model)
        return x
