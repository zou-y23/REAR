"""
Sec. 3.4 Loss: separate verb/noun heads + logit-adjusted cross-entropy (LACE), Eqs. (16)-(19).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def lace_adjustment(num_classes: int, class_priors: Tensor, tau: float = 1.0) -> Tensor:
    """
    Eq. (16): Delta_c = tau * log(pi_c), pi_c empirical class prior on the training set.
    class_priors: [C], normalized.
    """
    return tau * torch.log(class_priors.clamp_min(1e-12))


def lace_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    delta: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """
    logits: [B, C], delta: [C] added per class before softmax CE (Eqs. 17, 18).
    """
    adj = logits + delta.unsqueeze(0)
    return F.cross_entropy(adj, targets, ignore_index=ignore_index)


def rear_total_loss(
    logits_v: Tensor,
    logits_n: Tensor,
    y_v: Tensor,
    y_n: Tensor,
    priors_v: Tensor,
    priors_n: Tensor,
    tau: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Total L = L_v + L_n, Eq. (19)."""
    dv = lace_adjustment(logits_v.size(-1), priors_v, tau)
    dn = lace_adjustment(logits_n.size(-1), priors_n, tau)
    lv = lace_cross_entropy(logits_v, y_v, dv)
    ln = lace_cross_entropy(logits_n, y_n, dn)
    return lv, ln, lv + ln
