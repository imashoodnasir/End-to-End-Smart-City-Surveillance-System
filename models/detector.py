
"""Transformer-based object detector (DETR-style)."""

from typing import Dict, Tuple

import torch
from torch import nn

from config import CONFIG


class TransformerDetector(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 num_queries: int = 100, num_classes: int = 10):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.bbox_head = nn.Linear(d_model, 4)
        self.class_head = nn.Linear(d_model, num_classes)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        # feat: (B, C, H, W) -> (B, HW, C)
        B, C, H, W = feat.shape
        x = feat.flatten(2).permute(0, 2, 1)
        x = self.encoder(x)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        hs = self.decoder(queries, x)

        logits = self.class_head(hs)          # (B, Q, num_classes)
        boxes = self.bbox_head(hs).sigmoid()  # normalized (0,1)

        return {"pred_logits": logits, "pred_boxes": boxes}
