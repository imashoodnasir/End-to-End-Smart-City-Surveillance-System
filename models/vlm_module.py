
"""Vision--language module using a text encoder and cross-attention."""

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

from config import CONFIG


class VisionLanguageModule(nn.Module):
    def __init__(self, d_model: int = 256, num_classes: int = 10):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.model.vl_text_encoder)
        self.text_encoder = AutoModel.from_pretrained(CONFIG.model.vl_text_encoder)
        self.proj_visual = nn.Linear(d_model, self.text_encoder.config.hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.text_encoder.config.hidden_size, num_heads=8, batch_first=True
        )
        self.cls_head = nn.Linear(self.text_encoder.config.hidden_size, num_classes)

    def encode_text(self, texts: list, device: str) -> Dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=CONFIG.model.vl_max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = self.text_encoder(**tokens)
        return out.last_hidden_state  # (B, L, H)

    def forward(self, visual_tokens: torch.Tensor, text_hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        # visual_tokens: (B, N, d_model)
        v = self.proj_visual(visual_tokens)
        attn_out, _ = self.cross_attn(query=text_hidden, key=v, value=v)
        pooled = attn_out.mean(dim=1)
        logits = self.cls_head(pooled)
        return {"vl_logits": logits}
