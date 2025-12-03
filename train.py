
"""High-level training loop for multi-task unified surveillance model.

This script demonstrates how to alternate between X-MAS, CHAD, and TU-DAT
batches and optimize a single model with a composite multi-task loss.
"""

from typing import Dict, Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from config import CONFIG
from datasets import XMasDataset, CHADDataset, TUDATDataset
from models import MultiModalEncoder, TransformerDetector, TemporalBehaviorEncoder, ContrastiveHead, VisionLanguageModule, MultiTaskSurveillanceModel
from losses import detection_loss, vl_alignment_loss
from utils import detection_collate, chad_collate


def build_dataloaders():
    ds_cfg = CONFIG.dataset
    xmas_train = XMasDataset(ds_cfg.xmas_root, split="train")
    chad_train = CHADDataset(ds_cfg.chad_root, split="train",
                             window_size=ds_cfg.window_size,
                             overlap=ds_cfg.window_overlap)
    tudat_train = TUDATDataset(ds_cfg.tudat_root, split="train")

    xmas_loader = DataLoader(
        xmas_train,
        batch_size=CONFIG.training.batch_size,
        shuffle=True,
        num_workers=CONFIG.training.num_workers,
        collate_fn=detection_collate,
    )
    chad_loader = DataLoader(
        chad_train,
        batch_size=CONFIG.training.batch_size,
        shuffle=True,
        num_workers=CONFIG.training.num_workers,
        collate_fn=chad_collate,
    )
    tudat_loader = DataLoader(
        tudat_train,
        batch_size=CONFIG.training.batch_size,
        shuffle=True,
        num_workers=CONFIG.training.num_workers,
    )

    return xmas_loader, chad_loader, tudat_loader


def train():
    device = CONFIG.training.device
    xmas_loader, chad_loader, tudat_loader = build_dataloaders()

    # Placeholder: number of classes should come from dataset metadata
    num_det_classes = 10
    num_vl_classes = 10

    model = MultiTaskSurveillanceModel(num_det_classes, num_vl_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG.training.lr,
                            weight_decay=CONFIG.training.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG.training.amp)

    for epoch in range(CONFIG.training.max_epochs):
        model.train()
        for (xmas_batch, chad_batch, tudat_batch) in zip(xmas_loader, chad_loader, tudat_loader):
            optimizer.zero_grad(set_to_none=True)

            # X-MAS branch
            x_samples, x_targets = xmas_batch
            if len(x_samples) > 0:
                rgb = torch.stack([s["rgb"] for s in x_samples]).to(device)
                thermal = torch.stack([s["thermal"] for s in x_samples]).to(device)
                depth = torch.stack([s["depth"] for s in x_samples]).to(device)
                x_batch = {"rgb": rgb, "thermal": thermal, "depth": depth}
            else:
                x_batch = None

            # CHAD branch
            chad_batch = {k: v.to(device) for k, v in chad_batch.items()}

            # TU-DAT branch (assumes features + labels present)
            tudat_batch = {k: v.to(device) for k, v in tudat_batch.items()}

            with torch.cuda.amp.autocast(enabled=CONFIG.training.amp):
                total_loss = torch.tensor(0.0, device=device)

                if x_batch is not None:
                    det_out = model.forward_xmas(x_batch)
                    # For simplicity we use the first target dict
                    if len(x_targets) > 0:
                        det_loss = detection_loss(det_out, x_targets[0])
                        total_loss = total_loss + CONFIG.training.lambda_det * det_loss

                chad_out = model.forward_chad(chad_batch)
                # here we create a simple positive/negative split for contrastive loss
                z_norm = chad_out["z_norm"]
                # naive split into anchor/pos/neg
                B, W, D = z_norm.shape
                if W >= 3:
                    anchor = z_norm[:, 0:1]
                    pos = z_norm[:, 1:2]
                    neg = z_norm[:, 2:]
                    con_loss = model.contrastive.info_nce_loss(anchor, pos, neg)
                    total_loss = total_loss + CONFIG.training.lambda_contrastive * con_loss

                # TU-DAT: vision-language alignment
                visual_tokens = tudat_batch["video_features"]
                text_hidden = tudat_batch["text_hidden"]
                vl_out = model.forward_tudat(visual_tokens, text_hidden)
                vl_loss = vl_alignment_loss(vl_out, tudat_batch["label"])
                total_loss = total_loss + CONFIG.training.lambda_vl * vl_loss

            scaler.scale(total_loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch {epoch+1}: loss={total_loss.item():.4f}")


if __name__ == "__main__":
    train()
