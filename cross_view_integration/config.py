"""Cross-view integration hyper-parameters (Sec. 3.3)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrossViewIntegrationConfig:
    d_model: int
    max_k: int
    attn_dropout: float = 0.0


@dataclass
class IntegrationAblationFlags:
    """Toggle sub-stages for paper ablations (not all wired into CrossViewIntegrationModule)."""

    use_sim_agg: bool = True
    use_cross_attn: bool = True
    use_final_mlp: bool = True
    use_residual_around_attn: bool = False


@dataclass
class MultiHeadCrossViewSpec:
    num_heads: int = 4
    head_dim: int = 64
    rope_theta: float = 10000.0


@dataclass
class GatingFusionSpec:
    """Highway / gated residual between ego and fused exo."""

    init_bias_toward_ego: float = 0.0
