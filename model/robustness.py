"""Bonus Module C: robustness to degradation.

Two uses:
  1. Offline study  : `python -m model.robustness` -> report/robustness.json (AUC / accuracy per degradation,
                       split by seen / unseen generators), plus a markdown table.
  2. Live stress test: `stress_test(detector, img)` -> per-degradation probabilities for one image (API/UI).
"""
from __future__ import annotations

import argparse
import json
from functools import partial

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from model import config as C
from model.utils import autocast_ctx
from model.data.datasets import ManifestDataset, add_noise, eval_transform, gaussian_blur, jpeg_recompress, load_manifest, rescale, screenshot
from model.nets import load_checkpoint
from model.predict import load_calibration

DEGRADATIONS = {
    "original": None,
    "jpeg_q90": partial(jpeg_recompress, quality=90),
    "jpeg_q75": partial(jpeg_recompress, quality=75),
    "jpeg_q50": partial(jpeg_recompress, quality=50),
    "jpeg_q30": partial(jpeg_recompress, quality=30),
    "resize_0.5x": partial(rescale, factor=0.5),
    "resize_0.25x": partial(rescale, factor=0.25),
    "blur_s1": partial(gaussian_blur, sigma=1.0),
    "blur_s2": partial(gaussian_blur, sigma=2.0),
    "noise_5": partial(add_noise, std=5.0),
    "screenshot": partial(screenshot, scale=0.8, quality=80),
    "screenshot_jpeg50": lambda im: jpeg_recompress(screenshot(im, 0.7, 85), 50),
}
LIVE_SET = ["jpeg_q75", "jpeg_q30", "resize_0.5x", "blur_s1", "screenshot", "noise_5"]
PRETTY = {
    "original": "Original", "jpeg_q90": "JPEG q90", "jpeg_q75": "JPEG q75", "jpeg_q50": "JPEG q50", "jpeg_q30": "JPEG q30",
    "resize_0.5x": "Downscale 0.5x", "resize_0.25x": "Downscale 0.25x", "blur_s1": "Blur s=1", "blur_s2": "Blur s=2",
    "noise_5": "Noise s=5", "screenshot": "Screenshot", "screenshot_jpeg50": "Screenshot + JPEG q50",
}


@torch.no_grad()
def _probs(model, rows, device, img_size, degrade, T, workers=4, batch=96) -> np.ndarray:
    ds = ManifestDataset(rows, eval_transform(img_size), degrade=degrade)
    dl = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
    out = []
    for x, _, _, _ in dl:
        with autocast_ctx(device):
            o = model(x.to(device))
        out.append(o["logit"].float().cpu().numpy())
    return 1 / (1 + np.exp(-np.concatenate(out) / T))


def study(checkpoint=C.DEFAULT_CHECKPOINT, n_per_group: int = 500, workers: int = 4, seed: int = C.SEED) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(checkpoint, device)
    img_size = ckpt.get("config", {}).get("img_size", C.IMG_SIZE)
    calib = load_calibration()
    T, thr = float(calib.get("temperature", 1.0)), float(calib.get("threshold", 0.5))
    rng = np.random.default_rng(seed)
    test = load_manifest(split="test")
    reals = [r for r in test if r["label"] == 0 and r["source"] != "cifake"]
    seen = [r for r in test if r["label"] == 1 and r.get("protocol") == "seen"]
    unseen = [r for r in test if r["label"] == 1 and r.get("protocol") == "unseen"]
    pick = lambda rows, n: [rows[i] for i in rng.permutation(len(rows))[:n]]  # noqa: E731
    reals, seen, unseen = pick(reals, n_per_group), pick(seen, n_per_group), pick(unseen, n_per_group)
    rows = reals + seen + unseen
    y = np.array([r["label"] for r in rows])
    is_seen = np.array([r.get("protocol") == "seen" or r["label"] == 0 for r in rows])
    is_unseen = np.array([r.get("protocol") == "unseen" or r["label"] == 0 for r in rows])
    results = {}
    base = None
    for name, fn in DEGRADATIONS.items():
        print(f"[{name}] ...", flush=True)
        p = _probs(model, rows, device, img_size, fn, T, workers)
        if base is None:
            base = p
        pred = p >= thr
        results[name] = {
            "label": PRETTY[name],
            "auc_all": float(roc_auc_score(y, p)),
            "auc_seen": float(roc_auc_score(y[is_seen], p[is_seen])),
            "auc_unseen": float(roc_auc_score(y[is_unseen], p[is_unseen])),
            "accuracy": float((pred == y).mean()),
            "fpr": float(pred[y == 0].mean()),
            "tpr_seen": float(pred[(y == 1) & is_seen].mean()),
            "tpr_unseen": float(pred[(y == 1) & is_unseen].mean()),
            "verdict_flip_rate": float((pred != (base >= thr)).mean()),
            "mean_abs_prob_shift": float(np.abs(p - base).mean()),
        }
        print(f"   AUC all {results[name]['auc_all']:.4f} seen {results[name]['auc_seen']:.4f} unseen {results[name]['auc_unseen']:.4f} acc {results[name]['accuracy']:.3f}", flush=True)
    out = {"n_real": len(reals), "n_seen_fake": len(seen), "n_unseen_fake": len(unseen), "threshold": thr, "temperature": T, "results": results}
    path = C.REPORT_DIR / "robustness.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("\n| degradation | AUC all | AUC seen | AUC unseen | acc | FPR | flip rate |\n|---|---|---|---|---|---|---|")
    for k, r in results.items():
        print(f"| {r['label']} | {r['auc_all']:.4f} | {r['auc_seen']:.4f} | {r['auc_unseen']:.4f} | {r['accuracy']:.3f} | {r['fpr']:.3f} | {r['verdict_flip_rate']:.3f} |")
    print(f"-> {path}")
    return out


def stress_test(detector, img: Image.Image, names: list[str] | None = None) -> dict:
    """Per-image live stress test used by the API: how stable is the verdict under common degradations?"""
    from model.data.datasets import to_input_tensor
    names = names or LIVE_SET
    img = img.convert("RGB")
    base_logit = float(detector.logits(to_input_tensor(img, detector.img_size))["logit"][0])
    base_p = detector.calibrated_prob(base_logit)
    base_v = base_p >= detector.threshold
    items = []
    for n in names:
        deg = DEGRADATIONS[n](img)
        lg = float(detector.logits(to_input_tensor(deg, detector.img_size))["logit"][0])
        p = detector.calibrated_prob(lg)
        items.append({"id": n, "label": PRETTY[n], "probability_ai": round(p, 4), "delta": round(p - base_p, 4),
                      "verdict": "ai_generated" if p >= detector.threshold else "real", "flipped": bool((p >= detector.threshold) != base_v)})
    flips = sum(i["flipped"] for i in items)
    max_shift = max(abs(i["delta"]) for i in items) if items else 0.0
    if flips == 0 and max_shift < 0.10:
        stability, note = "stable", f"Verdict unchanged under all {len(items)} degradations; largest likelihood shift {max_shift:.2f}."
    elif flips == 0:
        stability, note = "moderate", f"Verdict unchanged, but the likelihood moved by up to {max_shift:.2f} under degradation - treat the confidence as softer."
    else:
        worst = max(items, key=lambda i: abs(i["delta"]))
        stability, note = "fragile", f"Verdict flipped under {flips} of {len(items)} degradations (worst: {worst['label']}, shift {worst['delta']:+.2f}). Do not rely on this verdict for a degraded copy."
    return {"base_probability_ai": round(base_p, 4), "items": items, "flips": flips, "max_shift": round(max_shift, 4), "stability": stability, "note": note}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=C.DEFAULT_CHECKPOINT)
    ap.add_argument("--n", type=int, default=500, help="images per group (real / seen fake / unseen fake)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    study(a.checkpoint, a.n, a.workers)
