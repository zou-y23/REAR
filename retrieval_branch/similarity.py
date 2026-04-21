"""
Batch similarity for cross-view retrieval (Eq. 5): video-video and video-text terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def pairwise_cosine_vv(z_ego: Tensor, exo_bank: Tensor, eps: float = 1e-8) -> Tensor:
    """z_ego [B,D], exo_bank [N,D] -> sim_vv [B,N]."""
    return F.cosine_similarity(
        z_ego.unsqueeze(1),
        exo_bank.unsqueeze(0),
        dim=-1,
        eps=eps,
    )


def broadcast_sim_vt(sim_vt: Tensor | None, sim_vv: Tensor) -> Tensor:
    """Expand sim_vt [N] or [B,N] to [B,N] matching sim_vv."""
    if sim_vt is None:
        return sim_vv
    if sim_vt.dim() == 1:
        return sim_vt.unsqueeze(0).expand_as(sim_vv)
    return sim_vt


def combined_score(
    sim_vv: Tensor,
    sim_vt: Tensor,
    *,
    w_vv: float = 0.5,
    w_vt: float = 0.5,
) -> Tensor:
    """Mean of the two similarity channels (Eq. 5)."""
    return w_vv * sim_vv + w_vt * sim_vt


def pairwise_l2_neg_distance(z_ego: Tensor, exo_bank: Tensor) -> Tensor:
    """Negative L2 distance as similarity surrogate (unbounded; use with care)."""
    d = z_ego.unsqueeze(1) - exo_bank.unsqueeze(0)
    return -(d * d).sum(dim=-1)


def bilinear_score(z_ego: Tensor, exo_bank: Tensor, weight: Tensor) -> Tensor:
    """Bilinear match z^T W e with W [D,D] — learnable cross-view map (experimental)."""
    q = z_ego @ weight
    return (q.unsqueeze(1) * exo_bank.unsqueeze(0)).sum(dim=-1)


class LearnableSimilarityHead(nn.Module):
    """Two-layer MLP on concatenated [z_ego; e_j] -> scalar score per pair (expensive O(BN))."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

    def forward(self, z_ego: Tensor, exo_bank: Tensor) -> Tensor:
        b, d = z_ego.shape
        n = exo_bank.size(0)
        zr = z_ego.unsqueeze(1).expand(b, n, d)
        er = exo_bank.unsqueeze(0).expand(b, n, d)
        x = torch.cat([zr, er], dim=-1)
        return self.mlp(x).squeeze(-1)


class MultiScaleCosineFusion(nn.Module):
    """Fuse cosine at multiple normalized temperatures (ensemble of softmax logits)."""

    def __init__(self, temperatures: tuple[float, ...] = (0.07, 0.1, 0.2)) -> None:
        super().__init__()
        self.temps = temperatures
        self.weights = nn.Parameter(torch.ones(len(temperatures)) / len(temperatures))

    def forward(self, z_ego: Tensor, exo_bank: Tensor) -> Tensor:
        out = []
        for t in self.temps:
            s = pairwise_cosine_vv(z_ego, exo_bank) / t
            out.append(s)
        w = torch.softmax(self.weights, dim=0)
        return sum(w[i] * out[i] for i in range(len(out)))
