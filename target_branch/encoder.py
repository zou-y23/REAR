"""
Shared video encoder E: maps pooled egocentric clip features to d_model (Eq. 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from target_branch.config import TargetBranchConfig


class SharedVideoEncoder(nn.Module):
    """LayerNorm + two Linear layers to d_model (UniFormerV2-style pooled feature adapter)."""

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        *,
        dropout: float = 0.0,
        hidden_mult: int = 1,
    ) -> None:
        super().__init__()
        cfg = TargetBranchConfig(
            in_dim=in_dim,
            d_model=d_model,
            dropout=dropout,
            hidden_mult=hidden_mult,
        )
        self._cfg = cfg
        layers: list[nn.Module] = [
            nn.LayerNorm(cfg.in_dim),
            nn.Linear(cfg.in_dim, cfg.d_model),
            nn.ReLU(inplace=True),
        ]
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        h = cfg.d_model * cfg.hidden_mult
        if cfg.hidden_mult > 1:
            layers.extend(
                [
                    nn.Linear(cfg.d_model, h),
                    nn.ReLU(inplace=True),
                ]
            )
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            layers.append(nn.Linear(h, cfg.d_model))
        else:
            layers.append(nn.Linear(cfg.d_model, cfg.d_model))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GatedLinearBlock(nn.Module):
    """SwiGLU-style gated FFN fragment (alternative to plain ReLU MLP)."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_in, d_hidden)
        self.w_up = nn.Linear(d_in, d_hidden)
        self.w_down = nn.Linear(d_hidden, d_out)

    def forward(self, x: Tensor) -> Tensor:
        g = F.silu(self.w_gate(x))
        u = self.w_up(x)
        return self.w_down(g * u)


class GatedSharedVideoEncoder(nn.Module):
    """Encoder variant using stacked gated blocks (heavier than SharedVideoEncoder)."""

    def __init__(self, in_dim: int, d_model: int, *, num_blocks: int = 2, expand: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model))
        hid = d_model * expand
        self.blocks = nn.ModuleList(
            [GatedLinearBlock(d_model, hid, d_model) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.norm(h)


class BottleneckVideoEncoder(nn.Module):
    """Bottleneck mid-layer (in -> wide -> narrow -> d_model)."""

    def __init__(self, in_dim: int, d_model: int, *, bottleneck_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.ln(x)
        h = self.fc2(F.gelu(self.fc1(h)))
        return x + self.drop(h)


class ResidualStackEncoder(nn.Module):
    """Repeated residual MLP blocks after a linear stem."""

    def __init__(self, in_dim: int, d_model: int, *, depth: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model))
        self.blocks = nn.ModuleList([ResidualMLPBlock(d_model, dropout) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return h


class TwinTowerVideoEncoder(nn.Module):
    """Two parallel towers averaged — redundant capacity for robustness experiments."""

    def __init__(self, in_dim: int, d_model: int) -> None:
        super().__init__()
        self.tower_a = SharedVideoEncoder(in_dim, d_model)
        self.tower_b = SharedVideoEncoder(in_dim, d_model)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: Tensor) -> Tensor:
        a = self.tower_a(x)
        b = self.tower_b(x)
        w = torch.sigmoid(self.alpha)
        return w * a + (1.0 - w) * b


class EncoderFactory:
    """Builds named encoder variants (registry pattern)."""

    @staticmethod
    def build(kind: str, in_dim: int, d_model: int, **kwargs: object) -> nn.Module:
        if kind == "mlp":
            return SharedVideoEncoder(in_dim, d_model, **kwargs)  # type: ignore[arg-type]
        if kind == "gated":
            return GatedSharedVideoEncoder(in_dim, d_model, **kwargs)  # type: ignore[arg-type]
        if kind == "bottleneck":
            bd = int(kwargs.get("bottleneck_dim", (in_dim + d_model) // 2))
            return BottleneckVideoEncoder(in_dim, d_model, bottleneck_dim=bd)
        if kind == "residual_stack":
            return ResidualStackEncoder(in_dim, d_model, **kwargs)  # type: ignore[arg-type]
        if kind == "twin":
            return TwinTowerVideoEncoder(in_dim, d_model)
        raise KeyError(f"unknown encoder kind: {kind}")
