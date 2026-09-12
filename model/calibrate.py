"""Temperature scaling + operating-point selection on the VALIDATION split (never on test).

Writes weights/calibration.json:
  temperature       T minimising NLL of sigmoid(logit / T) on val
  threshold         calibrated-probability threshold giving TARGET_FPR on val real images (the stated operating point)
  threshold_alt     0.5 for reference
  ece_before/after  expected calibration error (15 bins)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import config as C
from model.utils import autocast_ctx
from model.data.datasets import ManifestDataset, eval_transform, load_manifest
from model.nets import load_checkpoint


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi)
        if m.any():
            e += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(e)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    best_T, best_nll = 1.0, 1e9
    for T in grid:
        p = 1 / (1 + np.exp(-logits / T))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        nll = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        if nll < best_nll:
            best_T, best_nll = float(T), float(nll)
    return best_T


@torch.no_grad()
def collect_logits(model, rows, device, img_size, batch=128, workers=4) -> tuple[np.ndarray, np.ndarray]:
    ds = ManifestDataset(rows, eval_transform(img_size))
    dl = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
    logits, labels = [], []
    for x, y, _, _ in dl:
        with autocast_ctx(device):
            out = model(x.to(device))
        logits.append(out["logit"].float().cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=C.DEFAULT_CHECKPOINT)
    ap.add_argument("--target-fpr", type=float, default=C.TARGET_FPR)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(a.checkpoint, device)
    img_size = ckpt.get("config", {}).get("img_size", C.IMG_SIZE)
    rows = load_manifest(split="val")
    assert not ({r["generator"] for r in rows} & set(C.HELD_OUT_GENERATORS)), "held-out generator in val!"
    print(f"val rows: {len(rows)}")
    logits, labels = collect_logits(model, rows, device, img_size, workers=a.workers)
    T = fit_temperature(logits, labels)
    p_before = 1 / (1 + np.exp(-logits))
    p_after = 1 / (1 + np.exp(-logits / T))
    # operating point: FPR = target on val reals
    reals = p_after[labels == 0]
    thr = float(np.quantile(reals, 1 - a.target_fpr))
    thr = float(np.clip(thr, 0.05, 0.95))
    fpr_at = float((reals >= thr).mean())
    tpr_at = float((p_after[labels == 1] >= thr).mean())
    out = {
        "temperature": T, "threshold": round(thr, 4), "threshold_alt": 0.5, "target_fpr": a.target_fpr,
        "val_fpr_at_threshold": fpr_at, "val_tpr_at_threshold": tpr_at,
        "val_fpr_at_0.5": float((reals >= 0.5).mean()), "val_tpr_at_0.5": float((p_after[labels == 1] >= 0.5).mean()),
        "ece_before": ece(p_before, labels), "ece_after": ece(p_after, labels),
        "n_val": int(len(labels)), "n_val_real": int((labels == 0).sum()), "fitted": True,
        "checkpoint_epoch": ckpt.get("epoch"), "sources": sorted({r["source"] for r in rows}),
    }
    C.CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    C.CALIBRATION_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"-> {C.CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
