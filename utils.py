
"""Utility functions: collate, metrics stubs, etc."""

from typing import List, Tuple, Dict, Any

import torch


def detection_collate(batch):
    samples, targets = zip(*batch)
    return list(samples), list(targets)


def chad_collate(batch):
    windows = [b["windows"] for b in batch]
    labels = [b["label"] for b in batch]
    windows = torch.stack(windows, dim=0)
    labels = torch.stack(labels, dim=0)
    return {"windows": windows, "label": labels}
