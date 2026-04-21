"""
REAR full model (Fig. 2): wires the three top-level packages + classification heads.

  target_branch/          — ego encoder E
  retrieval_branch/       — cross-view retrieval + class-adaptive k
  cross_view_integration/ — fuse z_ego and {z_exo^i}

Verb / noun heads and LACE are applied on z (Sec. 3.4).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from cross_view_integration import CrossViewIntegrationModule
from retrieval_branch import RetrievalBranch
from target_branch import TargetBranch


class REAR(nn.Module):
    """
    Composes TargetBranch, RetrievalBranch, CrossViewIntegrationModule, then heads.

    Call either:
      - forward(ego_input, z_exo=..., k_active=...) when retrieval is offline, or
      - forward(ego_input, exo_bank=..., k=..., sim_vt=..., k_active=...) for on-line retrieval.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        num_verbs: int,
        num_nouns: int,
        max_k: int = 20,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_k = max_k
        self.target_branch = TargetBranch(in_dim, d_model)
        self.retrieval_branch = RetrievalBranch(d_model, max_k=max_k)
        self.cross_view_integration = CrossViewIntegrationModule(d_model, max_k)
        self.classifier_v = nn.Linear(d_model, num_verbs)
        self.classifier_n = nn.Linear(d_model, num_nouns)

    def forward(
        self,
        ego_input: Tensor,
        z_exo: Tensor | None = None,
        *,
        exo_bank: Tensor | None = None,
        k: int | None = None,
        sim_vt: Tensor | None = None,
        k_active: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        ego_input: [B, in_dim] pooled egocentric clip.

        If z_exo is given [B, K, D]: skip on-line retrieval.
        Else pass exo_bank [N, D] and k to run the retrieval branch.

        Returns z, verb logits, noun logits.
        """
        z_ego = self.target_branch(ego_input)
        if z_exo is None:
            if exo_bank is None or k is None:
                raise ValueError("Provide z_exo, or (exo_bank and k) for the retrieval branch.")
            z_exo, _ = self.retrieval_branch(z_ego, exo_bank, k, sim_vt=sim_vt)
        z = self.cross_view_integration(z_ego, z_exo, k_active=k_active)
        return z, self.classifier_v(z), self.classifier_n(z)
