"""Class-adaptive k (Eq. 6) and frequency binning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


def class_adaptive_k(
    class_id: int,
    head_classes: set[int],
    mid_classes: set[int],
    tail_classes: set[int],
) -> int:
    """k(c) in {20, 10, 5} for tail / mid / head bins."""
    if class_id in tail_classes:
        return 20
    if class_id in mid_classes:
        return 10
    if class_id in head_classes:
        return 5
    return 10


def frequency_bins_from_counts(class_counts: dict[int, int]) -> tuple[set[int], set[int], set[int]]:
    """Split classes by training count into top 20% / middle 60% / bottom 20%."""
    if not class_counts:
        return set(), set(), set()
    items = sorted(class_counts.items(), key=lambda x: -x[1])
    n = len(items)
    n_head = max(1, n * 20 // 100)
    n_tail = max(1, n * 20 // 100)
    head_ids = {c for c, _ in items[:n_head]}
    tail_ids = {c for c, _ in items[-n_tail:]}
    mid_ids = {c for c, _ in items[n_head:-n_tail]} if n_head + n_tail < n else set()
    return head_ids, mid_ids, tail_ids


class FrequencyTier(Enum):
    HEAD = "head"
    MID = "mid"
    TAIL = "tail"
    UNKNOWN = "unknown"


@dataclass
class ClassFrequencyRecord:
    class_id: int
    count: int
    tier: FrequencyTier = FrequencyTier.UNKNOWN


def tier_for_class(
    class_id: int,
    head_classes: set[int],
    mid_classes: set[int],
    tail_classes: set[int],
) -> FrequencyTier:
    if class_id in tail_classes:
        return FrequencyTier.TAIL
    if class_id in mid_classes:
        return FrequencyTier.MID
    if class_id in head_classes:
        return FrequencyTier.HEAD
    return FrequencyTier.UNKNOWN


def class_adaptive_k_piecewise(
    class_id: int,
    head_classes: set[int],
    mid_classes: set[int],
    tail_classes: set[int],
    *,
    k_tail: int = 20,
    k_mid: int = 10,
    k_head: int = 5,
    k_default: int = 10,
) -> int:
    """Configurable bucket sizes (same structure as Eq. 6)."""
    t = tier_for_class(class_id, head_classes, mid_classes, tail_classes)
    if t is FrequencyTier.TAIL:
        return k_tail
    if t is FrequencyTier.MID:
        return k_mid
    if t is FrequencyTier.HEAD:
        return k_head
    return k_default


def build_class_records(class_counts: dict[int, int]) -> list[ClassFrequencyRecord]:
    head, mid, tail = frequency_bins_from_counts(class_counts)
    out: list[ClassFrequencyRecord] = []
    for cid, cnt in class_counts.items():
        tier = tier_for_class(cid, head, mid, tail)
        out.append(ClassFrequencyRecord(class_id=cid, count=cnt, tier=tier))
    return sorted(out, key=lambda r: -r.count)
