"""Scalar-pool top-k (used when similarities are precomputed per candidate)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def retrieve_topk_exocentric(
    sim_vv: Tensor,
    sim_vt: Tensor,
    k: int,
) -> Tensor:
    """
    sim_vv, sim_vt: each [N] similarities per exocentric candidate in the pool.
    Returns top-k indices [k] (Eq. 5: argmax over mean similarity).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    score = 0.5 * (sim_vv + sim_vt)
    k_eff = min(k, score.numel())
    _, idx = torch.topk(score, k_eff, dim=0)
    return idx


def retrieve_topk_random_tiebreak(score: Tensor, k: int, *, generator: torch.Generator | None = None) -> Tensor:
    """Top-k with small noise to break ties (reproducible if generator is fixed)."""
    if generator is None:
        noise = torch.randn_like(score) * 1e-6
    else:
        noise = torch.randn(score.shape, device=score.device, dtype=score.dtype, generator=generator) * 1e-6
    return torch.topk(score + noise, min(k, score.numel()), dim=0).indices


def retrieve_topk_with_mask(score: Tensor, k: int, mask: Tensor) -> Tensor:
    """mask [N] bool: valid candidates; invalid get -inf before topk."""
    s = score.clone()
    s[~mask] = float("-inf")
    k_eff = min(k, int(mask.sum().item()))
    if k_eff <= 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    return torch.topk(s, k_eff, dim=0).indices


class TopKSelector(nn.Module):
    """Wrapper module that delegates to retrieve_topk_exocentric (scalar path)."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(self, sim_vv: Tensor, sim_vt: Tensor) -> Tensor:
        return retrieve_topk_exocentric(sim_vv, sim_vt, self.k)


class DiverseTopKSelector:
    """Greedy max-sum diversity (non-differentiable sketch; not used in forward)."""

    def __init__(self, k: int, lambda_div: float = 0.2) -> None:
        self.k = k
        self.lambda_div = lambda_div

    def select(self, embeddings: Tensor, scores: Tensor) -> list[int]:
        """embeddings [N,D], scores [N] — return list of indices."""
        n = embeddings.size(0)
        chosen: list[int] = []
        remaining = set(range(n))
        while len(chosen) < min(self.k, n) and remaining:
            best = max(
                remaining,
                key=lambda i: scores[i].item()
                - self.lambda_div * max((1 - (embeddings[i] * embeddings[j]).sum().item() for j in chosen), default=0.0),
            )
            chosen.append(best)
            remaining.remove(best)
        return chosen
