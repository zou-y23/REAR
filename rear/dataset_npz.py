"""
Load pre-extracted features from .npz (offline UniFormer / retrieval pipeline).

Required keys (float32 / int64):
  ego_feat: [N, in_dim]
  z_exo:    [N, K, d_model]
  y_v:      [N]
  y_n:      [N]
Optional:
  k_active: scalar or length-N vector; default K from z_exo.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzFeatureDataset(Dataset):
    """Loads numpy zip archive produced by offline feature extraction."""

    def __init__(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)
        self.ego = torch.from_numpy(data["ego_feat"]).float()
        self.exo = torch.from_numpy(data["z_exo"]).float()
        self.y_v = torch.from_numpy(data["y_v"]).long()
        self.y_n = torch.from_numpy(data["y_n"]).long()
        if "k_active" in data:
            ka = data["k_active"]
            self.k_active = int(ka) if ka.ndim == 0 else torch.from_numpy(ka).long()
        else:
            self.k_active = None

    def __len__(self) -> int:
        return self.ego.size(0)

    def __getitem__(self, i: int) -> dict:
        ka = self.k_active
        if isinstance(ka, torch.Tensor):
            ka = int(ka[i].item())
        elif ka is None:
            ka = self.exo.size(1)
        return {
            "ego_feat": self.ego[i],
            "z_exo": self.exo[i],
            "y_v": int(self.y_v[i].item()),
            "y_n": int(self.y_n[i].item()),
            "k_active": ka,
        }
