"""Retrieval-branch hyper-parameters (Sec. 3.2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalBranchConfig:
    d_model: int
    max_k: int = 20
    w_vv: float = 0.5
    w_vt: float = 0.5


@dataclass
class RetrievalPoolSpec:
    """Describes a candidate bank layout (shard id, augmentation tag, etc.)."""

    pool_id: str
    num_candidates: int
    deduplicate: bool = True


@dataclass
class HardNegativeMiningConfig:
    """Parameters for offline hard-negative index refresh (not wired to forward)."""

    num_hard: int = 32
    margin: float = 0.2
    refresh_every_epochs: int = 5


@dataclass
class ScoreFusionSchedule:
    """Anneal w_vv / w_vt across training (cosine schedule placeholder)."""

    w_vv_start: float = 0.7
    w_vv_end: float = 0.5
    total_steps: int = 100000
