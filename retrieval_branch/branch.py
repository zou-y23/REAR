"""
Retrieval branch (Fig. 2, bottom): cross-view retriever + class-adaptive selector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from retrieval_branch.class_adaptive import class_adaptive_k, frequency_bins_from_counts
from retrieval_branch.config import RetrievalBranchConfig
from retrieval_branch.similarity import (
    broadcast_sim_vt,
    combined_score,
    pairwise_cosine_vv,
)
from retrieval_branch.topk import retrieve_topk_exocentric


class RetrievalBranch(nn.Module):
    """
    Cross-view retriever (Eq. 5): mean(video-video, video-text) similarity, top-k.
    Class-adaptive k (Eq. 6) via static helpers when class frequency bins are known.
    """

    def __init__(self, d_model: int, max_k: int = 20) -> None:
        super().__init__()
        self.cfg = RetrievalBranchConfig(d_model=d_model, max_k=max_k)
        self.d_model = d_model
        self.max_k = max_k

    def forward(
        self,
        z_ego: Tensor,
        exo_bank: Tensor,
        k: int,
        sim_vt: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        z_ego: [B, D] egocentric embedding (same space as bank).
        exo_bank: [N, D] candidate exocentric clip embeddings (already encoded by E).
        k: number of clips to retrieve.
        sim_vt: optional [N] or [B, N] video-text similarity; if None, uses sim_vv for both terms.

        Returns z_exo [B, k_eff, D], topk_idx [B, k_eff].
        """
        k_eff = min(k, self.max_k, exo_bank.size(0))
        sim_vv = pairwise_cosine_vv(z_ego, exo_bank)
        sim_vt_b = broadcast_sim_vt(sim_vt, sim_vv)
        score = combined_score(
            sim_vv,
            sim_vt_b,
            w_vv=self.cfg.w_vv,
            w_vt=self.cfg.w_vt,
        )
        topk_idx = torch.topk(score, k_eff, dim=1).indices
        z_exo = exo_bank[topk_idx]
        return z_exo, topk_idx

    @staticmethod
    def k_from_class_frequency(
        class_id: int,
        head: set[int],
        mid: set[int],
        tail: set[int],
    ) -> int:
        return class_adaptive_k(class_id, head, mid, tail)

    @staticmethod
    def frequency_bins(class_counts: dict[int, int]) -> tuple[set[int], set[int], set[int]]:
        return frequency_bins_from_counts(class_counts)

    @staticmethod
    def retrieve_from_scalar_sims(sim_vv: Tensor, sim_vt: Tensor, k: int) -> Tensor:
        return retrieve_topk_exocentric(sim_vv, sim_vt, k)


class LearnableRetrievalBranch(nn.Module):
    """Retrieval with bilinear map W: z_ego -> query space before cosine to bank."""

    def __init__(self, d_model: int, max_k: int = 20) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_k = max_k
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        z_ego: Tensor,
        exo_bank: Tensor,
        k: int,
        sim_vt: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        q = self.proj(z_ego)
        k_eff = min(k, self.max_k, exo_bank.size(0))
        sim_vv = pairwise_cosine_vv(q, exo_bank)
        sim_vt_b = broadcast_sim_vt(sim_vt, sim_vv)
        score = combined_score(sim_vv, sim_vt_b, w_vv=0.5, w_vt=0.5)
        topk_idx = torch.topk(score, k_eff, dim=1).indices
        return exo_bank[topk_idx], topk_idx


class StochasticRetrievalBranch(nn.Module):
    """Adds Gumbel noise to scores before top-k (differentiable relaxation experiments)."""

    def __init__(self, d_model: int, max_k: int = 20, *, noise_scale: float = 0.01) -> None:
        super().__init__()
        self.inner = RetrievalBranch(d_model, max_k)
        self.noise_scale = noise_scale

    def forward(
        self,
        z_ego: Tensor,
        exo_bank: Tensor,
        k: int,
        sim_vt: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        k_eff = min(k, self.inner.max_k, exo_bank.size(0))
        sim_vv = pairwise_cosine_vv(z_ego, exo_bank)
        sim_vt_b = broadcast_sim_vt(sim_vt, sim_vv)
        score = combined_score(sim_vv, sim_vt_b, w_vv=0.5, w_vt=0.5)
        if self.training and self.noise_scale > 0:
            score = score + self.noise_scale * torch.randn_like(score)
        topk_idx = torch.topk(score, k_eff, dim=1).indices
        return exo_bank[topk_idx], topk_idx


class RetrievalBranchWithCache(nn.Module):
    """Caches last exo_bank reference for debugging / visualization hooks."""

    def __init__(self, d_model: int, max_k: int = 20) -> None:
        super().__init__()
        self.core = RetrievalBranch(d_model, max_k)
        self._last_bank: Tensor | None = None

    def forward(
        self,
        z_ego: Tensor,
        exo_bank: Tensor,
        k: int,
        sim_vt: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self._last_bank = exo_bank
        return self.core(z_ego, exo_bank, k, sim_vt=sim_vt)
