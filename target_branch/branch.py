"""
Target branch (Fig. 2, left): ego encoder E producing z_ego.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from target_branch.encoder import SharedVideoEncoder


class TargetBranch(nn.Module):
    """Ego / target branch: z_ego = E(v_ego)."""

    def __init__(self, in_dim: int, d_model: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder = SharedVideoEncoder(in_dim, d_model, dropout=dropout)

    def forward(self, ego_input: Tensor) -> Tensor:
        """
        ego_input: [B, in_dim] pooled egocentric embedding.
        Returns z_ego: [B, d_model].
        """
        return self.encoder(ego_input)


class TargetBranchWithAuxHeads(nn.Module):
    """Main ego embedding + optional auxiliary verb/noun probes (multi-task ablation)."""

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        *,
        num_aux_verbs: int | None = None,
        num_aux_nouns: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder = SharedVideoEncoder(in_dim, d_model)
        self.aux_v = nn.Linear(d_model, num_aux_verbs) if num_aux_verbs else None
        self.aux_n = nn.Linear(d_model, num_aux_nouns) if num_aux_nouns else None

    def forward(self, ego_input: Tensor) -> Tensor | tuple[Tensor, Tensor | None, Tensor | None]:
        z = self.encoder(ego_input)
        if self.aux_v is None and self.aux_n is None:
            return z
        logits_v = self.aux_v(z) if self.aux_v is not None else None
        logits_n = self.aux_n(z) if self.aux_n is not None else None
        return z, logits_v, logits_n


class WrappedTargetBranch(nn.Module):
    """Decorator-style wrapper: pre-norm input noise + post-norm L2."""

    def __init__(self, inner: TargetBranch, *, noise_std: float = 0.0) -> None:
        super().__init__()
        self.inner = inner
        self.noise_std = noise_std
        self.post_ln = nn.LayerNorm(inner.d_model)

    def forward(self, ego_input: Tensor) -> Tensor:
        if self.training and self.noise_std > 0:
            ego_input = ego_input + self.noise_std * ego_input.new_empty(ego_input.shape).normal_()
        z = self.inner(ego_input)
        return self.post_ln(z)


class StochasticDepthTargetBranch(nn.Module):
    """Wraps TargetBranch and randomly drops residual identity (training regularization)."""

    def __init__(self, inner: TargetBranch, drop_prob: float = 0.1) -> None:
        super().__init__()
        self.inner = inner
        self.drop_prob = drop_prob
        self._identity = nn.Identity()

    def forward(self, ego_input: Tensor) -> Tensor:
        z = self.inner(ego_input)
        if not self.training or self.drop_prob <= 0:
            return z
        if torch.rand(1, device=z.device).item() < self.drop_prob:
            return z * 0.0
        return z


class TargetBranchEnsemble(nn.Module):
    """Shallow ensemble of multiple TargetBranch instances (mean fusion)."""

    def __init__(self, branches: list[TargetBranch]) -> None:
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.d_model = branches[0].d_model

    def forward(self, ego_input: Tensor) -> Tensor:
        outs = [b(ego_input) for b in self.branches]
        return torch.stack(outs, dim=0).mean(dim=0)
