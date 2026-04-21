"""Synthetic pre-extracted features for training smoke tests without real data."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticREARDataset(Dataset):
    """
    Each item:
      ego_feat: [in_dim] pooled ego embedding (e.g. 768-D)
      z_exo: [K, d_model] optional offline exo stack (kept for debugging / offline path)
      exo_bank: [N_bank, d_model] shared candidate pool for on-line retrieval in training
      y_v, y_n: integer labels
      k_active: valid retrieval count (<= max_k)
    """

    def __init__(
        self,
        n: int,
        in_dim: int,
        d_model: int,
        max_k: int,
        num_verbs: int,
        num_nouns: int,
        seed: int = 0,
        fix_k: int | None = None,
        num_exo_bank: int = 64,
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.ego = torch.randn(n, in_dim, generator=g)
        self.exo = torch.randn(n, max_k, d_model, generator=g)
        self.exo_bank = torch.randn(num_exo_bank, d_model, generator=g)
        self.y_v = torch.randint(0, num_verbs, (n,), generator=g)
        self.y_n = torch.randint(0, num_nouns, (n,), generator=g)
        self.k_active = fix_k if fix_k is not None else max_k

    def __len__(self) -> int:
        return self.ego.size(0)

    def __getitem__(self, i: int) -> dict[str, Tensor | int]:
        return {
            "ego_feat": self.ego[i],
            "z_exo": self.exo[i],
            "exo_bank": self.exo_bank,
            "y_v": int(self.y_v[i].item()),
            "y_n": int(self.y_n[i].item()),
            "k_active": self.k_active,
        }
