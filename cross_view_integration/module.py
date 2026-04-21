"""
Cross-view integration module (Fig. 2, right): wires submodules from Sec. 3.3.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from cross_view_integration.attention import CrossViewAttentionFusion, CrossViewMultiHeadFusion
from cross_view_integration.config import CrossViewIntegrationConfig
from cross_view_integration.final_fusion import FinalIntegration, GatedFinalIntegration
from cross_view_integration.similarity_agg import SimilarityGuidedAggregation


class CrossViewIntegrationModule(nn.Module):
    """Fuses z_ego and {z_exo}: (i) similarity aggregation (ii) attention (iii) MLP."""

    def __init__(self, d_model: int, max_k: int, *, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.cfg = CrossViewIntegrationConfig(
            d_model=d_model, max_k=max_k, attn_dropout=attn_dropout
        )
        self.sim_agg = SimilarityGuidedAggregation(d_model, max_k)
        self.cross_attn = CrossViewAttentionFusion(d_model, dropout=attn_dropout)
        self.final_int = FinalIntegration(d_model)

    def forward(self, z_ego: Tensor, z_exo: Tensor, k_active: int | None = None) -> Tensor:
        _zp, _f_sum, alpha = self.sim_agg(z_ego, z_exo, k_active=k_active)
        b_exo = alpha.unsqueeze(-1) * z_exo
        z_exo_ego = self.cross_attn(z_ego, b_exo)
        return self.final_int(z_ego, z_exo_ego)


class CrossViewIntegrationModuleMHA(nn.Module):
    """Same pipeline as CrossViewIntegrationModule but swaps in multi-head cross attention."""

    def __init__(self, d_model: int, max_k: int, *, num_heads: int = 4, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.cfg = CrossViewIntegrationConfig(
            d_model=d_model, max_k=max_k, attn_dropout=attn_dropout
        )
        self.sim_agg = SimilarityGuidedAggregation(d_model, max_k)
        self.cross_attn = CrossViewMultiHeadFusion(d_model, num_heads=num_heads, dropout=attn_dropout)
        self.final_int = FinalIntegration(d_model)

    def forward(self, z_ego: Tensor, z_exo: Tensor, k_active: int | None = None) -> Tensor:
        _zp, _f, alpha = self.sim_agg(z_ego, z_exo, k_active=k_active)
        b_exo = alpha.unsqueeze(-1) * z_exo
        z_exo_ego = self.cross_attn(z_ego, b_exo)
        return self.final_int(z_ego, z_exo_ego)


class CrossViewIntegrationModuleGated(nn.Module):
    """Gated final fusion variant."""

    def __init__(self, d_model: int, max_k: int, *, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.cfg = CrossViewIntegrationConfig(
            d_model=d_model, max_k=max_k, attn_dropout=attn_dropout
        )
        self.sim_agg = SimilarityGuidedAggregation(d_model, max_k)
        self.cross_attn = CrossViewAttentionFusion(d_model, dropout=attn_dropout)
        self.final_int = GatedFinalIntegration(d_model)

    def forward(self, z_ego: Tensor, z_exo: Tensor, k_active: int | None = None) -> Tensor:
        _zp, _f, alpha = self.sim_agg(z_ego, z_exo, k_active=k_active)
        b_exo = alpha.unsqueeze(-1) * z_exo
        z_exo_ego = self.cross_attn(z_ego, b_exo)
        return self.final_int(z_ego, z_exo_ego)


class CrossViewIntegrationRegistry:
    """String -> constructor for integration variants (experiments)."""

    _kinds: dict[str, type[nn.Module]] = {
        "default": CrossViewIntegrationModule,
        "mha": CrossViewIntegrationModuleMHA,
        "gated": CrossViewIntegrationModuleGated,
    }

    @classmethod
    def build(cls, kind: str, d_model: int, max_k: int, **kwargs: object) -> nn.Module:
        ctor = cls._kinds.get(kind, CrossViewIntegrationModule)
        return ctor(d_model, max_k, **kwargs)  # type: ignore[arg-type,misc]
