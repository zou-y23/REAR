#!/usr/bin/env python3
"""
REAR training entry point: Sec. 4.1 defaults (AdamW, batch size, epochs, LACE priors).

  # Synthetic smoke test (no dataset files)
  python train_rear.py --synthetic --epochs 3

  # Pre-extracted .npz (keys in rear/dataset_npz.py)
  python train_rear.py --npz path/to/features.npz

  # Ego-only baseline (paper REAR baseline)
  python train_rear.py --synthetic --mode ego_only --epochs 3

REAR forward in this script (mode rear):
  --synthetic: target_branch -> retrieval_branch (exo_bank from dataset) ->
    cross_view_integration -> heads (all three blocks run).
  --npz: no exo_bank; uses offline z_exo from file -> target_branch -> (retrieval skipped) ->
    cross_view_integration -> heads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rear.baseline import REARBaseline
from rear.config import TrainConfig
from rear.dataset_npz import NpzFeatureDataset
from rear.dataset_synthetic import SyntheticREARDataset
from rear.loss_lace import rear_total_loss
from rear.model import REAR
from rear.priors import empirical_priors


def _train_label_tensors(
    train_subset: Subset,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = train_subset.dataset
    idx = torch.tensor(train_subset.indices, dtype=torch.long)
    if hasattr(base, "y_v"):
        return base.y_v[idx], base.y_n[idx]
    raise TypeError("Dataset must expose y_v and y_n tensors (SyntheticREARDataset or NpzFeatureDataset)")


def collate_batch(items: list[dict]) -> dict:
    """Stack batch; k_active is None if it varies within the batch."""
    ego = torch.stack([x["ego_feat"] for x in items], dim=0)
    zexo = torch.stack([x["z_exo"] for x in items], dim=0)
    yv = torch.tensor([x["y_v"] for x in items], dtype=torch.long)
    yn = torch.tensor([x["y_n"] for x in items], dtype=torch.long)
    k0 = items[0]["k_active"]
    k_active = k0 if all(x["k_active"] == k0 for x in items) else None
    out: dict = {"ego_feat": ego, "z_exo": zexo, "y_v": yv, "y_n": yn, "k_active": k_active}
    if "exo_bank" in items[0]:
        out["exo_bank"] = items[0]["exo_bank"]
    return out


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Single-label top-1 accuracy."""
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    priors_v: torch.Tensor,
    priors_n: torch.Tensor,
    opt: AdamW | None,
    cfg: TrainConfig,
    device: torch.device,
    mode_rear: bool,
    train: bool,
) -> tuple[float, float, float]:
    """One train or val pass; returns mean loss, verb acc, noun acc."""
    if train:
        model.train()
    else:
        model.eval()

    tot_loss = 0.0
    acc_v = 0.0
    acc_n = 0.0
    n_batch = 0
    pv = priors_v.to(device)
    pn = priors_n.to(device)

    for batch in loader:
        ego = batch["ego_feat"].to(device)
        yv = batch["y_v"].to(device)
        yn = batch["y_n"].to(device)
        ka = batch.get("k_active")

        if mode_rear:
            assert isinstance(model, REAR)
            z_ego = model.target_branch(ego)
            if batch.get("exo_bank") is not None:
                exo_b = batch["exo_bank"].to(device)
                k_use = int(ka) if ka is not None else cfg.max_k
                k_use = min(k_use, cfg.max_k, exo_b.size(0))
                z_exo, _ = model.retrieval_branch(z_ego, exo_b, k_use, sim_vt=None)
            else:
                z_exo = batch["z_exo"].to(device)
            z = model.cross_view_integration(z_ego, z_exo, k_active=ka)
            lv = model.classifier_v(z)
            ln = model.classifier_n(z)
        else:
            _z, lv, ln = model(ego)

        _, _, loss = rear_total_loss(lv, ln, yv, yn, pv, pn, tau=cfg.tau_lace)

        if train and opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()

        tot_loss += float(loss.item())
        acc_v += accuracy(lv, yv)
        acc_n += accuracy(ln, yn)
        n_batch += 1

    return tot_loss / max(n_batch, 1), acc_v / max(n_batch, 1), acc_n / max(n_batch, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true", help="Use random synthetic features")
    p.add_argument("--npz", type=str, default="", help="Path to pre-extracted .npz")
    p.add_argument(
        "--mode",
        choices=("rear", "ego_only"),
        default="rear",
        help="rear=full REAR; ego_only=egocentric baseline (no retrieval fusion)",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    device = torch.device(args.device)

    if args.synthetic:
        full = SyntheticREARDataset(
            n=512,
            in_dim=cfg.feature_dim,
            d_model=cfg.d_model,
            max_k=cfg.max_k,
            num_verbs=cfg.num_verbs,
            num_nouns=cfg.num_nouns,
            seed=cfg.seed,
        )
    elif args.npz:
        full = NpzFeatureDataset(args.npz)
    else:
        p.error("Specify --synthetic or --npz")

    n_val = max(1, len(full) // 5)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    yv, yn = _train_label_tensors(train_ds)

    priors_v = empirical_priors(yv, cfg.num_verbs)
    priors_n = empirical_priors(yn, cfg.num_nouns)

    if args.mode == "rear":
        model = REAR(
            in_dim=cfg.feature_dim,
            d_model=cfg.d_model,
            num_verbs=cfg.num_verbs,
            num_nouns=cfg.num_nouns,
            max_k=cfg.max_k,
        ).to(device)
    else:
        model = REARBaseline(
            in_dim=cfg.feature_dim,
            d_model=cfg.d_model,
            num_verbs=cfg.num_verbs,
            num_nouns=cfg.num_nouns,
        ).to(device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_val = float("inf")
    bad = 0
    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr_av, tr_an = run_epoch(
            model,
            train_loader,
            priors_v,
            priors_n,
            opt,
            cfg,
            device,
            args.mode == "rear",
            train=True,
        )
        va_loss, va_av, va_an = run_epoch(
            model,
            val_loader,
            priors_v,
            priors_n,
            None,
            cfg,
            device,
            args.mode == "rear",
            train=False,
        )
        sched.step()
        print(
            f"epoch {ep:03d}  train loss={tr_loss:.4f} v_acc={tr_av:.3f} n_acc={tr_an:.3f}  "
            f"val loss={va_loss:.4f} v_acc={va_av:.3f} n_acc={va_an:.3f}"
        )

        if va_loss < best_val - 1e-5:
            best_val = va_loss
            bad = 0
        else:
            bad += 1
            if bad >= cfg.early_stop_patience:
                print(f"early stop at epoch {ep}")
                break


if __name__ == "__main__":
    main()
