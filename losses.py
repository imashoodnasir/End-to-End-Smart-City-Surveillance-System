
"""Loss utilities for detection, contrastive learning, and vision--language tasks."""

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


def detection_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Placeholder: in practice, Hungarian matching and focal + GIoU are used.
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    # Fake labels for skeleton: assume targets contain "labels" and "boxes"
    labels = targets.get("labels")
    gt_boxes = targets.get("boxes")
    if labels is None or gt_boxes is None:
        return torch.tensor(0.0, device=logits.device)

    cls_loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    l1_loss = F.l1_loss(boxes, gt_boxes)
    return cls_loss + l1_loss


def vl_alignment_loss(outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    logits = outputs["vl_logits"]
    return F.cross_entropy(logits, labels)
