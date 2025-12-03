
"""High-level wrapper combining all modules into a single model."""

from typing import Dict, Any

import torch
from torch import nn

from config import CONFIG
from .multimodal_encoder import MultiModalEncoder
from .detector import TransformerDetector
from .temporal_encoder import TemporalBehaviorEncoder
from .contrastive_head import ContrastiveHead
from .vlm_module import VisionLanguageModule


class MultiTaskSurveillanceModel(nn.Module):
    def __init__(self, num_det_classes: int, num_vl_classes: int):
        super().__init__()
        d_model = CONFIG.model.d_model
        self.encoder = MultiModalEncoder(d_model)
        self.detector = TransformerDetector(
            d_model=d_model,
            nhead=CONFIG.model.nhead,
            num_encoder_layers=CONFIG.model.num_encoder_layers,
            num_decoder_layers=CONFIG.model.num_decoder_layers,
            num_queries=CONFIG.model.num_queries,
            num_classes=num_det_classes,
        )
        self.temporal_encoder = TemporalBehaviorEncoder(
            num_joints=CONFIG.model.num_joints,
            pose_dim=CONFIG.model.pose_dim,
            d_model=d_model,
        )
        self.contrastive = ContrastiveHead(
            d_model=d_model,
            temperature=CONFIG.model.contrastive_temperature,
        )
        self.vlm = VisionLanguageModule(d_model=d_model, num_classes=num_vl_classes)

    def forward_xmas(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat = self.encoder(batch["rgb"], batch["thermal"], batch["depth"])
        det_out = self.detector(feat)
        return det_out

    def forward_chad(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        windows = batch["windows"]  # (B, W, T, J, 2)
        z = self.temporal_encoder(windows)
        z_norm = self.contrastive(z)
        return {"z": z, "z_norm": z_norm}

    def forward_tudat(self, visual_tokens: torch.Tensor, text_hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.vlm(visual_tokens, text_hidden)
