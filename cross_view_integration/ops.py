"""Tensor ops shared by integration submodules."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cosine_sim(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """a: [B,D], b: [B,K,D] -> [B,K]."""
    a_n = F.normalize(a, dim=-1, eps=eps)
    b_n = F.normalize(b, dim=-1, eps=eps)
    return (b_n * a_n.unsqueeze(1)).sum(dim=-1)


def dot_product_scores(a: Tensor, b: Tensor) -> Tensor:
    """Unnormalized dot product [B,D] x [B,K,D] -> [B,K]."""
    return (b * a.unsqueeze(1)).sum(dim=-1)


def l2_normalize_rows(x: Tensor, eps: float = 1e-8) -> Tensor:
    return F.normalize(x, dim=-1, eps=eps)


class TensorLayoutHelper:
    """Static helpers for padding / masking exo tensors (verbose on purpose)."""

    @staticmethod
    def pad_exo_to_k(z_exo: Tensor, max_k: int) -> Tensor:
        b, k, d = z_exo.shape
        if k >= max_k:
            return z_exo[:, :max_k, :]
        return F.pad(z_exo, (0, 0, 0, max_k - k))

    @staticmethod
    def stack_ego_exo(z_ego: Tensor, z_exo: Tensor) -> Tensor:
        return torch.cat([z_ego.unsqueeze(1), z_exo], dim=1)
