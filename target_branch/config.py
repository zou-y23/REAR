"""Hyper-parameters for the egocentric target branch (Sec. 3.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TargetBranchConfig:
    """Controls encoder depth / regularization; defaults match the original two-layer MLP."""

    in_dim: int
    d_model: int
    dropout: float = 0.0
    hidden_mult: int = 1


@dataclass
class EncoderArchitectureSpec:
    """Blueprint for ablation / registry-style encoder construction (not all combos are wired)."""

    name: str
    num_blocks: int = 2
    expansion: int = 4
    use_glu: bool = False
    use_pre_norm: bool = True


@dataclass
class EgoModalityConfig:
    """Optional multi-stream ego (RGB / flow / audio) — placeholder for future fusion."""

    d_rgb: int
    d_flow: int
    d_model: int
    fusion: str = "concat_mlp"


@dataclass
class TargetRegularizationBundle:
    """Stochastic depth / drop-path toggles (experimental)."""

    drop_path_rate: float = 0.0
    label_smoothing: float = 0.0
    feature_noise_std: float = 0.0
