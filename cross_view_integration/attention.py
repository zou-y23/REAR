"""Cross-view attention fusion (Eqs. 12–13)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossViewAttentionFusion(nn.Module):
    """Eqs. (12)-(13)."""

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.scale = d_model**0.5
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, z_ego: Tensor, b_exo: Tensor) -> Tensor:
        q = self.Wq(z_ego).unsqueeze(1)
        k = self.Wk(b_exo)
        v = self.Wv(b_exo)
        att = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        w = F.softmax(att, dim=-1)
        if self.dropout is not None:
            w = self.dropout(w)
        return torch.matmul(w, v).squeeze(1)


class CrossViewMultiHeadFusion(nn.Module):
    """Multi-head attention over exo tokens (alternative to single-head)."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, z_ego: Tensor, b_exo: Tensor) -> Tensor:
        q = z_ego.unsqueeze(1)
        out, _ = self.mha(q, b_exo, b_exo, need_weights=False)
        return out.squeeze(1)


class LinearAttentionFusion(nn.Module):
    """Lightweight surrogate: mean-pool exo then fuse with ego (placeholder for linear-attn papers)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, z_ego: Tensor, b_exo: Tensor) -> Tensor:
        pooled = b_exo.mean(dim=1)
        u = self.Wv(pooled)
        g = torch.sigmoid(self.gate(torch.cat([self.Wq(z_ego), u], dim=-1)))
        return g * u + (1 - g) * z_ego
