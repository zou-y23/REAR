"""Similarity-guided aggregation (Eqs. 7–11)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cross_view_integration.ops import cosine_sim


class SimilarityGuidedAggregation(nn.Module):
    """Eqs. (7)-(11)."""

    def __init__(self, d_model: int, max_k: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_k = max_k
        h = d_model * 2
        self.mlp1 = nn.Sequential(
            nn.Linear((max_k + 1) * d_model, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, d_model),
        )
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, z_ego: Tensor, z_exo: Tensor, k_active: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
        b, k, d = z_exo.shape
        assert d == self.d_model and k <= self.max_k
        if k < self.max_k:
            z_exo = F.pad(z_exo, (0, 0, 0, self.max_k - k))
        k = z_exo.size(1)

        s = cosine_sim(z_ego, z_exo)
        stack = torch.cat([z_ego.unsqueeze(1), z_exo], dim=1)
        flat = stack.reshape(b, (k + 1) * d)
        z_prime = self.mlp1(flat)
        s_prime = cosine_sim(z_prime, z_exo)
        logits = self.w1 * s + self.w2 * s_prime
        if k_active is not None and k_active < k:
            logits[:, k_active:] = float("-inf")
        alpha = F.softmax(logits, dim=-1)
        f_exo_stack = (alpha.unsqueeze(-1) * z_exo).sum(dim=1)
        return z_prime, f_exo_stack, alpha


class DepthwiseExoMixer(nn.Module):
    """Per-exo-slot linear + LayerNorm before similarity logits (extra capacity)."""

    def __init__(self, d_model: int, max_k: int) -> None:
        super().__init__()
        self.slot_norm = nn.LayerNorm(d_model)
        self.slot_proj = nn.Linear(d_model, d_model)
        self.max_k = max_k

    def forward(self, z_exo: Tensor) -> Tensor:
        b, k, d = z_exo.shape
        if k < self.max_k:
            z_exo = F.pad(z_exo, (0, 0, 0, self.max_k - k))
        h = self.slot_proj(self.slot_norm(z_exo))
        return h + z_exo


class SimilarityGuidedAggregationLite(nn.Module):
    """Cheaper variant: single cosine softmax without z_prime MLP (ablation)."""

    def __init__(self, d_model: int, max_k: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_k = max_k
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, z_ego: Tensor, z_exo: Tensor, k_active: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
        if z_exo.size(1) < self.max_k:
            z_exo = F.pad(z_exo, (0, 0, 0, self.max_k - z_exo.size(1)))
        s = cosine_sim(z_ego, z_exo) / self.temperature.abs().clamp(min=1e-3)
        if k_active is not None and k_active < s.size(1):
            s[:, k_active:] = float("-inf")
        alpha = F.softmax(s, dim=-1)
        z_prime = z_ego
        f_exo = (alpha.unsqueeze(-1) * z_exo).sum(dim=1)
        return z_prime, f_exo, alpha


class DualTowerSimilarityAgg(nn.Module):
    """Runs two SimilarityGuidedAggregation stacks and fuses (ensemble)."""

    def __init__(self, d_model: int, max_k: int) -> None:
        super().__init__()
        self.a = SimilarityGuidedAggregation(d_model, max_k)
        self.b = SimilarityGuidedAggregation(d_model, max_k)
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, z_ego: Tensor, z_exo: Tensor, k_active: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
        zp1, f1, a1 = self.a(z_ego, z_exo, k_active=k_active)
        zp2, f2, a2 = self.b(z_ego, z_exo, k_active=k_active)
        w = torch.sigmoid(self.beta)
        return w * zp1 + (1 - w) * zp2, w * f1 + (1 - w) * f2, a1
