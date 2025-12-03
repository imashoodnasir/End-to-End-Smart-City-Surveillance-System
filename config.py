
"""Global configuration for unified multimodal surveillance experiments.

This module centralizes dataset paths, training hyperparameters, and model
settings so that all scripts (training, validation, inference) share a
single source of truth.
"""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class DatasetConfig:
    xmas_root: str = "data/xmas"
    chad_root: str = "data/chad"
    tudat_root: str = "data/tudat"
    img_size: Tuple[int, int] = (640, 640)
    fps: int = 10
    window_size: int = 16
    window_overlap: float = 0.5


@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_workers: int = 4
    max_epochs: int = 50
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cuda"
    amp: bool = True

    # multi-task loss weights
    lambda_det: float = 1.0
    lambda_contrastive: float = 0.5
    lambda_vl: float = 1.0


@dataclass
class ModelConfig:
    cnn_backbone: str = "resnet18"
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_queries: int = 100
    pose_dim: int = 2
    num_joints: int = 17
    pose_embed_dim: int = 128
    contrastive_temperature: float = 0.1
    vl_text_encoder: str = "distilbert-base-uncased"
    vl_max_length: int = 32
    vocab_size: int = 30522  # placeholder


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


CONFIG = ExperimentConfig()
