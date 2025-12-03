
"""Contrastive head for InfoNCE-style loss and prototype storage."""

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from config import CONFIG


class ContrastiveHead(nn.Module):
    def __init__(self, d_model: int, temperature: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.temperature = temperature
        self.register_buffer("prototypes", torch.empty(0, d_model))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, W, d_model)
        z = self.proj(z)
        z = F.normalize(z, dim=-1)
        return z

    def info_nce_loss(self, z_anchor: torch.Tensor, z_pos: torch.Tensor,
                      z_neg: torch.Tensor) -> torch.Tensor:
        # flatten over windows
        a = z_anchor.reshape(-1, z_anchor.size(-1))
        p = z_pos.reshape(-1, z_pos.size(-1))
        n = z_neg.reshape(-1, z_neg.size(-1))

        logits_pos = torch.sum(a * p, dim=-1, keepdim=True) / self.temperature
        logits_neg = a @ n.t() / self.temperature
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        labels = torch.zeros(a.size(0), dtype=torch.long, device=a.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    @torch.no_grad()
    def update_prototypes(self, z_normal: torch.Tensor, momentum: float = 0.9) -> None:
        z = F.normalize(z_normal.reshape(-1, z_normal.size(-1)), dim=-1)
        if self.prototypes.numel() == 0:
            self.prototypes = z.mean(0, keepdim=True)
        else:
            self.prototypes = momentum * self.prototypes + (1 - momentum) * z.mean(0, keepdim=True)
