
"""Dataset definitions for X-MAS, CHAD, and TU-DAT.

These classes provide a unified PyTorch Dataset interface for the three
tasks:
- X-MAS: multi-modal object detection
- CHAD: skeleton-based anomaly detection
- TU-DAT: video + text for vision--language reasoning
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from config import CONFIG


def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


@dataclass
class XMasSample:
    rgb: Image.Image
    thermal: Image.Image
    depth: Image.Image
    target: Dict[str, torch.Tensor]


class XMasDataset(Dataset):
    """Simplified X-MAS dataset loader.

    Expects a directory layout such as:
        root/
            images_rgb/
            images_thermal/
            images_depth/
            annotations.pt  # dict: image_id -> target
    """

    def __init__(self, root: str, split: str = "train", transforms=None) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.transforms = transforms

        ann_path = os.path.join(root, f"annotations_{split}.pt")
        if os.path.exists(ann_path):
            data = torch.load(ann_path)
            self.ids: List[str] = data["ids"]
            self.targets: Dict[str, Dict[str, torch.Tensor]] = data["targets"]
        else:
            # Placeholder if annotations are not yet prepared.
            self.ids = []
            self.targets = {}
        self.rgb_dir = os.path.join(root, "images_rgb")
        self.thermal_dir = os.path.join(root, "images_thermal")
        self.depth_dir = os.path.join(root, "images_depth")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        img_id = self.ids[idx]
        rgb = load_image(os.path.join(self.rgb_dir, f"{img_id}.png"))
        thermal = load_image(os.path.join(self.thermal_dir, f"{img_id}.png"))
        depth = load_image(os.path.join(self.depth_dir, f"{img_id}.png"))

        target = self.targets.get(img_id, {})
        sample = {
            "rgb": rgb,
            "thermal": thermal,
            "depth": depth,
            "image_id": img_id,
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample, target


class CHADDataset(Dataset):
    """Skeleton-based anomaly detection dataset.

    Expects pre-extracted pose sequences stored as .pt files containing:
        {
            "poses": Tensor[T, J, 2],
            "label": int  # 0 = normal, 1 = anomalous
        }
    """

    def __init__(self, root: str, split: str = "train", window_size: int = 16, overlap: float = 0.5):
        super().__init__()
        self.root = os.path.join(root, split)
        self.files = sorted([f for f in os.listdir(self.root) if f.endswith(".pt")])
        self.window_size = window_size
        self.overlap = overlap

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = os.path.join(self.root, self.files[idx])
        data = torch.load(path)
        poses: torch.Tensor = data["poses"]  # (T, J, 2)
        label: int = int(data["label"])

        T = poses.shape[0]
        step = max(1, int(self.window_size * (1 - self.overlap)))
        windows = []
        for start in range(0, max(1, T - self.window_size + 1), step):
            end = start + self.window_size
            if end <= T:
                windows.append(poses[start:end])

        if not windows:
            windows.append(poses[: self.window_size])

        windows = torch.stack(windows, dim=0)  # (W, T, J, 2)

        return {
            "windows": windows,
            "label": torch.tensor(label, dtype=torch.long),
        }


class TUDATDataset(Dataset):
    """TU-DAT video-language dataset.

    This implementation assumes that visual features have been pre-extracted
    as clip-level tensors and paired with tokenized text prompts.
    """

    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        self.root = os.path.join(root, split)
        self.files = sorted([f for f in os.listdir(self.root) if f.endswith(".pt")])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = os.path.join(self.root, self.files[idx])
        data = torch.load(path)
        # expected keys: "video_features", "input_ids", "attention_mask", "label"
        return data
