"""
REAR (baseline) from Table 1: target branch only (no retrieval, no integration).
"""

from __future__ import annotations

import torch.nn as nn

from target_branch import TargetBranch


class REARBaseline(nn.Module):
    """z = E(ego), then verb/noun heads."""

    def __init__(self, in_dim: int, d_model: int, num_verbs: int, num_nouns: int) -> None:
        super().__init__()
        self.target_branch = TargetBranch(in_dim, d_model)
        self.head_v = nn.Linear(d_model, num_verbs)
        self.head_n = nn.Linear(d_model, num_nouns)

    def forward(self, ego_input):
        z = self.target_branch(ego_input)
        return z, self.head_v(z), self.head_n(z)
