"""Evaluate on the TEST split with the unseen-generator protocol. Writes report/metrics.json + a markdown table.

Groups:
  overall  : every test row (CIFAKE test + GenImage/Imagenette test)
  cifake   : CIFAKE 20k test (32px, SD1.4 fakes)
  seen     : high-res test reals + fakes from generators SEEN in training
  unseen   : the SAME high-res test reals + fakes from HELD-OUT generators (never trained on)
  per-generator AUC uses each generator's fakes vs the high-res test reals.
Also: attribution accuracy / macro-F1 (seen test fakes), calibration (ECE, reliability bins), ROC points.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from model import config as C
from model.utils import autocast_ctx
from model.calibrate import ece
from model.data.datasets import ManifestDataset, eval_transform, load_manifest
from model.nets import load_checkpoint
from model.predict import load_calibration


@torch.no_grad()
def infer(model, rows, device, img_size, batch=128, workers=4):
    ds = ManifestDataset(rows, eval_transform(img_size))
    dl = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
    logits, attrs = [], []
    for i, (x, _, _, _) in enumerate(dl):
        with autocast_ctx(device):
            out = model(x.to(device))
        logits.append(out["logit"].float().cpu().numpy()); attrs.append(out["attr_logits"].float().cpu().numpy())
        if i % 20 == 0:
            print(f"  batch {i}/{len(dl)}", flush=True)
    return np.concatenate(logits), np.concatenate(attrs)


def group_metrics(p: np.ndarray, y: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr_c, tpr_c, _ = roc_curve(y, p) if len(np.unique(y)) > 1 else (np.array([0, 1]), np.array([0, 1]), None)
    idx = np.linspace(0, len(fpr_c) - 1, min(200, len(fpr_c))).astype(int)
    return {
        "n": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float((pred == y).mean()),
        "fpr": float(fp / max(fp + tn, 1)), "tpr": float(tp / max(tp + fn, 1)),
        "precision_fake": float(tp / max(tp + fp, 1)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "labels": ["real", "ai_generated"]},
        "threshold": thr,
        "roc": {"fpr": fpr_c[idx].round(4).tolist(), "tpr": tpr_c[idx].round(4).tolist()},
    }


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        out.append({"bin": [round(float(lo), 2), round(float(hi), 2)], "count": int(m.sum()),
                    "mean_confidence": float(p[m].mean()) if m.any() else None, "fraction_ai": float(y[m].mean()) if m.any() else None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=C.DEFAULT_CHECKPOINT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=C.METRICS_PATH)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(a.checkpoint, device)
    img_size = ckpt.get("config", {}).get("img_size", C.IMG_SIZE)
    calib = load_calibration()
    T, thr = float(calib.get("temperature", 1.0)), float(calib.get("threshold", 0.5))
    attr_classes = list(ckpt.get("attr_classes", C.ATTRIBUTION_CLASSES))

    rows = load_manifest(split="test")
    print(f"test rows: {len(rows)}  (T={T:.3f}, thr={thr:.3f})")
    logits, attr_logits = infer(model, rows, device, img_size, workers=a.workers)
    p = 1 / (1 + np.exp(-logits / T))
    y = np.array([r["label"] for r in rows])
    gen = np.array([r["generator"] for r in rows])
    src = np.array([r["source"] for r in rows])
    proto = np.array([r.get("protocol", "") for r in rows])

    hi_real = (y == 0) & (src != "cifake")
    seen_fake = (y == 1) & (proto == "seen")
    unseen_fake = (y == 1) & (proto == "unseen")
    groups = {
        "overall": np.ones(len(rows), bool),
        "cifake": src == "cifake",
        "seen": hi_real | seen_fake,
        "unseen": hi_real | unseen_fake,
        "highres_all": hi_real | seen_fake | unseen_fake,
    }
    metrics = {"groups": {}, "per_generator": {}, "threshold": thr, "temperature": T, "checkpoint": str(a.checkpoint),
               "held_out_generators": C.HELD_OUT_GENERATORS, "seen_generators": [g for g in C.SEEN_GENERATORS if g != "Real"],
               "n_test": len(rows), "config": ckpt.get("config", {}), "epoch": ckpt.get("epoch")}
    for name, m in groups.items():
        if m.sum() and len(np.unique(y[m])) > 1:
            metrics["groups"][name] = group_metrics(p[m], y[m], thr)
            metrics["groups"][name]["at_0.5"] = {k: v for k, v in group_metrics(p[m], y[m], 0.5).items() if k in ("accuracy", "fpr", "tpr", "macro_f1", "confusion_matrix")}
    # per generator: that generator's fakes vs high-res reals
    for g in sorted(set(gen[y == 1])):
        gm = (gen == g) & (y == 1)
        reals = (src == "cifake") & (y == 0) if g == "SD14" and (src[gm] == "cifake").all() else hi_real
        mm = gm | reals
        metrics["per_generator"][g] = {
            "n_fake": int(gm.sum()), "held_out": g in C.HELD_OUT_GENERATORS, "family": C.GENERATOR_FAMILY.get(g, "?"),
            "auc": float(roc_auc_score(y[mm], p[mm])), "detection_rate": float((p[gm] >= thr).mean()),
            "mean_p_ai": float(p[gm].mean()),
        }
    # attribution on seen test fakes
    am = seen_fake & np.isin(gen, attr_classes)
    if am.any():
        true_idx = np.array([attr_classes.index(g) for g in gen[am]])
        pred_idx = attr_logits[am].argmax(1)
        pr, rc, f1, sup = precision_recall_fscore_support(true_idx, pred_idx, labels=list(range(len(attr_classes))), zero_division=0)
        cm = confusion_matrix(true_idx, pred_idx, labels=list(range(len(attr_classes))))
        # family-level accuracy
        fam_true = np.array([C.GENERATOR_FAMILY[attr_classes[i]] for i in true_idx])
        fam_pred = np.array([C.GENERATOR_FAMILY[attr_classes[i]] for i in pred_idx])
        metrics["attribution"] = {
            "classes": attr_classes, "n": int(am.sum()),
            "accuracy": float((pred_idx == true_idx).mean()), "macro_f1": float(np.mean([f for f, s in zip(f1, sup) if s > 0])),
            "family_accuracy": float((fam_true == fam_pred).mean()),
            "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])} for i, c in enumerate(attr_classes)},
            "confusion_matrix": cm.tolist(),
        }
        # how are UNSEEN generators attributed? (family-level hint quality)
        um = unseen_fake
        if um.any():
            pred_u = attr_logits[um].argmax(1)
            fam_u = defaultdict(lambda: defaultdict(int))
            for g, pi in zip(gen[um], pred_u):
                fam_u[g][attr_classes[pi]] += 1
            metrics["attribution"]["unseen_mapping"] = {g: dict(d) for g, d in fam_u.items()}
    metrics["calibration"] = {"ece": ece(p, y), "reliability": reliability(p, y), "ece_highres": ece(p[groups["highres_all"]], y[groups["highres_all"]])}

    out = a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    # save raw predictions for robustness / inspection
    import csv
    with (out.parent / "test_predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "generator", "source", "protocol", "p_ai"])
        for r, pi in zip(rows, p):
            w.writerow([r["path"], r["label"], r["generator"], r["source"], r.get("protocol", ""), f"{pi:.6f}"])

    print("\n| group | n | AUC | macro-F1 | acc | FPR | TPR |")
    print("|---|---|---|---|---|---|---|")
    for name, g in metrics["groups"].items():
        print(f"| {name} | {g['n']} | {g['auc']:.4f} | {g['macro_f1']:.4f} | {g['accuracy']:.4f} | {g['fpr']:.4f} | {g['tpr']:.4f} |")
    print("\n| generator | held-out | n | AUC | detection rate |")
    print("|---|---|---|---|---|")
    for g, m in metrics["per_generator"].items():
        print(f"| {g} | {'yes' if m['held_out'] else 'no'} | {m['n_fake']} | {m['auc']:.4f} | {m['detection_rate']:.3f} |")
    if "attribution" in metrics:
        at = metrics["attribution"]
        print(f"\nattribution: acc {at['accuracy']:.3f}  macro-F1 {at['macro_f1']:.3f}  family-acc {at['family_accuracy']:.3f}  (n={at['n']})")
    print(f"\nECE {metrics['calibration']['ece']:.4f}   -> {out}")


if __name__ == "__main__":
    main()
