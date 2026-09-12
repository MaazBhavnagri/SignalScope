"""Train SignalScopeNet on the manifest (train split), select on val ROC-AUC, save best checkpoint.

Usage:
  python -m model.train --epochs 8 --batch 64
  python -m model.train --limit 2000 --epochs 1        # smoke test
Held-out generators are excluded by construction (they only exist in split=test).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from model import config as C
from model.utils import autocast_ctx
from model.data.datasets import ManifestDataset, balanced_sample_weights, eval_transform, load_manifest, train_transform
from model.nets import build_model


def seed_all(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_eval(model, loader, device) -> dict:
    model.eval()
    logits, labels, attr_pred, attr_true = [], [], [], []
    for x, y, a, _ in loader:
        x = x.to(device, non_blocking=True)
        with autocast_ctx(device):
            out = model(x)
        logits.append(out["logit"].float().cpu()); labels.append(y)
        attr_pred.append(out["attr_logits"].argmax(1).cpu()); attr_true.append(a)
    logits, labels = torch.cat(logits).numpy(), torch.cat(labels).numpy()
    attr_pred, attr_true = torch.cat(attr_pred).numpy(), torch.cat(attr_true).numpy()
    probs = 1 / (1 + np.exp(-logits))
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    acc = float(((probs > 0.5) == (labels > 0.5)).mean())
    m = attr_true >= 0
    attr_acc = float((attr_pred[m] == attr_true[m]).mean()) if m.any() else float("nan")
    return {"auc": float(auc), "acc": acc, "attr_acc": attr_acc, "loss_proxy": float(np.mean(np.abs(probs - labels)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=C.DEFAULT_BATCH)
    ap.add_argument("--lr", type=float, default=C.DEFAULT_LR)
    ap.add_argument("--img", type=int, default=C.IMG_SIZE)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sources", nargs="+", default=None, help="restrict training sources (default: all in manifest)")
    ap.add_argument("--limit", type=int, default=0, help="use only N train rows (smoke test)")
    ap.add_argument("--samples-per-epoch", type=int, default=32000, help="balanced draws per epoch (sampler uses replacement); 0 = len(train)")
    ap.add_argument("--out", type=Path, default=C.WEIGHTS_DIR)
    ap.add_argument("--name", default="signalscope_best.pt")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()

    seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    a.out.mkdir(parents=True, exist_ok=True)

    train_rows = load_manifest(split="train", sources=a.sources)
    val_rows = load_manifest(split="val", sources=a.sources)
    assert train_rows and val_rows, "manifest has no train/val rows - run model.data.prepare"
    # hard guarantee: no held-out generator in train/val
    leaked = {r["generator"] for r in train_rows + val_rows} & set(C.HELD_OUT_GENERATORS)
    assert not leaked, f"held-out generators leaked into train/val: {leaked}"
    if a.limit:
        rng = np.random.default_rng(a.seed)
        train_rows = [train_rows[i] for i in rng.permutation(len(train_rows))[: a.limit]]
        val_rows = [val_rows[i] for i in rng.permutation(len(val_rows))[: max(200, a.limit // 5)]]
    print(f"train={len(train_rows)}  val={len(val_rows)}  device={device}")

    ds_tr = ManifestDataset(train_rows, train_transform(a.img))
    ds_va = ManifestDataset(val_rows, eval_transform(a.img))
    n_draw = a.samples_per_epoch if a.samples_per_epoch and not a.limit else len(train_rows)
    sampler = WeightedRandomSampler(balanced_sample_weights(train_rows), num_samples=n_draw, replacement=True)
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, sampler=sampler, num_workers=a.workers, pin_memory=True,
                       drop_last=True, persistent_workers=a.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=a.batch * 2, shuffle=False, num_workers=a.workers, pin_memory=True,
                       persistent_workers=a.workers > 0)

    model = build_model(pretrained=True).to(device)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location=device)["state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=C.WEIGHT_DECAY)
    steps = a.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15, anneal_strategy="cos",
                                                div_factor=20, final_div_factor=100)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss(ignore_index=C.ATTR_IGNORE)

    history, best_auc = [], -1.0
    ckpt_path = a.out / a.name
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for step, (x, y, attr, _) in enumerate(dl_tr):
            x, y, attr = x.to(device, non_blocking=True), y.to(device, non_blocking=True), attr.to(device, non_blocking=True)
            y_s = y * (1 - C.LABEL_SMOOTHING) + 0.5 * C.LABEL_SMOOTHING
            with autocast_ctx(device):
                out = model(x)
                loss_bin = F.binary_cross_entropy_with_logits(out["logit"].float(), y_s)
                loss_attr = ce(out["attr_logits"].float(), attr) if (attr >= 0).any() else torch.zeros((), device=device)
                loss = loss_bin + C.ATTR_LOSS_WEIGHT * loss_attr
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
            if step % 50 == 0:
                el = time.time() - t0
                print(f"ep {epoch} [{step}/{len(dl_tr)}] loss {loss.item():.4f} (bin {loss_bin.item():.4f} attr {float(loss_attr):.4f}) "
                      f"lr {sched.get_last_lr()[0]:.2e} {n/max(el,1e-6):.0f} img/s", flush=True)
        ev = run_eval(model, dl_va, device)
        rec = {"epoch": epoch, "train_loss": tot / max(n, 1), **{f"val_{k}": v for k, v in ev.items()}, "time_s": time.time() - t0}
        history.append(rec)
        print(f"==> epoch {epoch}: {json.dumps(rec)}", flush=True)
        if not math.isnan(ev["auc"]) and ev["auc"] > best_auc:
            best_auc = ev["auc"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_auc": best_auc,
                        "attr_classes": C.ATTRIBUTION_CLASSES, "held_out_generators": C.HELD_OUT_GENERATORS,
                        "config": {"backbone": C.BACKBONE, "img_size": a.img, "lr": a.lr, "batch": a.batch, "epochs": a.epochs,
                                   "sources": sorted({r["source"] for r in train_rows}), "n_train": len(train_rows)}},
                       ckpt_path)
            print(f"    saved best -> {ckpt_path} (val AUC {best_auc:.4f})")
        (a.out / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"done. best val AUC {best_auc:.4f}")


if __name__ == "__main__":
    main()
