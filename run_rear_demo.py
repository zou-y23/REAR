#!/usr/bin/env python3
"""Smoke test: REAR with the three modules (target / retrieval / cross-view integration)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from rear.loss_lace import rear_total_loss
from rear.model import REAR


def main() -> None:
    torch.manual_seed(0)
    b, in_dim, d_model, max_k = 2, 512, 128, 8
    n_verb, n_noun = 97, 300
    num_exo_candidates = 32

    model = REAR(
        in_dim=in_dim,
        d_model=d_model,
        num_verbs=n_verb,
        num_nouns=n_noun,
        max_k=max_k,
    )

    ego_feat = torch.randn(b, in_dim)
    exo_bank = torch.randn(num_exo_candidates, d_model)

    # Path A: on-line retrieval (target_branch -> retrieval_branch -> cross_view_integration)
    z, lv, ln = model(ego_feat, exo_bank=exo_bank, k=max_k, k_active=max_k)
    print("path A (exo_bank)", "z", tuple(z.shape), "logits_v", tuple(lv.shape), "logits_n", tuple(ln.shape))

    # Path B: precomputed z_exo (retrieval done offline)
    z_exo_offline = exo_bank[:max_k].unsqueeze(0).expand(b, -1, -1)
    z2, lv2, ln2 = model(ego_feat, z_exo=z_exo_offline, k_active=max_k)
    print("path B (z_exo)", "z", tuple(z2.shape), "logits_v", tuple(lv2.shape), "logits_n", tuple(ln2.shape))

    y_v = torch.randint(0, n_verb, (b,))
    y_n = torch.randint(0, n_noun, (b,))
    pri_v = torch.ones(n_verb) / n_verb
    pri_n = torch.ones(n_noun) / n_noun
    loss_v, loss_n, total = rear_total_loss(lv, ln, y_v, y_n, pri_v, pri_n)
    print("L_v", float(loss_v), "L_n", float(loss_n), "L", float(total))


if __name__ == "__main__":
    main()
