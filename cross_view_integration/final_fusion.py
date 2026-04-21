"""Final MLP integration (Eq. 14)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FinalIntegration(nn.Module):
    """Eq. (14)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp2 = nn.Sequential(
            nn.Linear(2 * d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, z_ego: Tensor, z_exo_attended: Tensor) -> Tensor:
        return self.mlp2(torch.cat([z_ego, z_exo_attended], dim=-1))


class GatedFinalIntegration(nn.Module):
    """Highway gate between ego-only and fused representation."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, z_ego: Tensor, z_exo_attended: Tensor) -> Tensor:
        cat = torch.cat([z_ego, z_exo_attended], dim=-1)
        g = torch.sigmoid(self.gate(cat))
        y = self.mlp(cat)
        return g * y + (1 - g) * z_ego


class ResidualFusionBlock(nn.Module):
    """Residual around FinalIntegration (deeper stack)."""

    def __init__(self, d_model: int, depth: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([FinalIntegration(d_model) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])

    def forward(self, z_ego: Tensor, z_exo_attended: Tensor) -> Tensor:
        h = self.blocks[0](z_ego, z_exo_attended)
        h = self.norms[0](h)
        for i in range(1, len(self.blocks)):
            h = h + self.blocks[i](z_ego, z_exo_attended)
            h = self.norms[i](h)
        return h


class BilinearFinalIntegration(nn.Module):
    """Bilinear mixing of ego and exo streams before small MLP."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.w = nn.Bilinear(d_model, d_model, d_model)
        self.post = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

    def forward(self, z_ego: Tensor, z_exo_attended: Tensor) -> Tensor:
        return self.post(self.w(z_ego, z_exo_attended))
