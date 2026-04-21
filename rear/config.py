"""
Sec. 4.1.4 training defaults aligned with the paper (tunable on the validation set).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Optimizer
    lr: float = 3e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    epochs: int = 30
    early_stop_patience: int = 5
    tau_lace: float = 1.0
    # Video / features (frozen encoder output dimension from the paper)
    feature_dim: int = 768
    clip_frames: int = 8
    spatial_size: int = 224
    # Model
    d_model: int = 256
    max_k: int = 20
    num_verbs: int = 97
    num_nouns: int = 300
    # Class space (EPIC-style sizes; change for other datasets)
    seed: int = 42
