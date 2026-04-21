"""Empirical class priors for LACE."""

from __future__ import annotations

import torch


def empirical_priors(labels: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """
    labels: [N] int in [0, num_classes).
    Returns normalized frequency vector [num_classes] with Laplace smoothing.
    """
    labels = labels.long().view(-1)
    cnt = torch.bincount(labels, minlength=num_classes).float()
    cnt = cnt + eps
    return cnt / cnt.sum()
