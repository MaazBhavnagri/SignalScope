"""Bonus Module G: active-defence analysis - how easily is the detector fooled, and what helps?

Attacks (white-box, on AI-generated test images, goal = make the detector say "real"):
  * FGSM / PGD in pixel space with epsilon in {1,2,4}/255 (imperceptible-to-mild perturbations)
  * post-processing attacks are covered by robustness.py (JPEG, resize, blur, screenshot)
Mitigations evaluated:
  * JPEG q75 pre-filter at inference (destroys high-frequency adversarial noise)
  * horizontal-flip TTA averaging
Outputs report/attacks.json. Honest by design: we report the detection rate before/after each attack and mitigation.
"""
from __future__ import annotations

import argparse
import io
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from model import config as C
from model.data.datasets import ManifestDataset, eval_transform, load_manifest
from model.nets import load_checkpoint
from model.predict import load_calibration

MEAN = torch.tensor(C.IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(C.IMAGENET_STD).view(1, 3, 1, 1)


def _denorm(x):
    return x * STD.to(x.device) + MEAN.to(x.device)


def _norm(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def pgd(model, x_norm, eps, steps, alpha):
    """Minimise the AI logit (push towards 'real'); x in normalised space, epsilon in [0,1] pixel space."""
    x0 = _denorm(x_norm).clamp(0, 1)
    delta = torch.zeros_like(x0)
    if steps > 1:
        delta.uniform_(-eps, eps)
    for _ in range(steps):
        delta.requires_grad_(True)
        logit = model(_norm((x0 + delta).clamp(0, 1)))["logit"]
        loss = logit.sum()  # we want to decrease it
        grad, = torch.autograd.grad(loss, delta)
        delta = (delta.detach() - alpha * grad.sign()).clamp(-eps, eps)
    return _norm((x0 + delta.detach()).clamp(0, 1))


def jpeg_filter(x_norm, quality=75):
    x = (_denorm(x_norm).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    out = []
    for a in x:
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=quality)
        buf.seek(0)
        out.append(torch.from_numpy(np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1))
    return _norm(torch.stack(out).to(x_norm.device))


@torch.no_grad()
def det_rate(model, x, T, thr, jpeg=False, tta=False):
    if jpeg:
        x = jpeg_filter(x)
    logit = model(x)["logit"]
    if tta:
        logit = 0.5 * (logit + model(torch.flip(x, dims=[3]))["logit"])
    p = torch.sigmoid(logit / T)
    return (p >= thr).float().cpu().numpy(), p.cpu().numpy()


def study(checkpoint=C.DEFAULT_CHECKPOINT, n_fake=300, n_real=300, workers=4, seed=C.SEED) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(checkpoint, device)
    img_size = ckpt.get("config", {}).get("img_size", C.IMG_SIZE)
    calib = load_calibration()
    T, thr = float(calib.get("temperature", 1.0)), float(calib.get("threshold", 0.5))
    rng = np.random.default_rng(seed)
    test = load_manifest(split="test")
    fakes = [r for r in test if r["label"] == 1 and r["source"] != "cifake"]
    reals = [r for r in test if r["label"] == 0 and r["source"] != "cifake"]
    fakes = [fakes[i] for i in rng.permutation(len(fakes))[:n_fake]]
    reals = [reals[i] for i in rng.permutation(len(reals))[:n_real]]
    xs = torch.cat([x for x, _, _, _ in DataLoader(ManifestDataset(fakes, eval_transform(img_size)), batch_size=32, num_workers=workers)])
    xr = torch.cat([x for x, _, _, _ in DataLoader(ManifestDataset(reals, eval_transform(img_size)), batch_size=32, num_workers=workers)])
    attacks = {"none": (0, 0), "fgsm_1/255": (1 / 255, 1), "fgsm_2/255": (2 / 255, 1), "pgd_1/255": (1 / 255, 5),
               "pgd_2/255": (2 / 255, 5), "pgd_4/255": (4 / 255, 5)}
    results = {}
    for name, (eps, steps) in attacks.items():
        print(f"[{name}]", flush=True)
        rates = {"plain": [], "jpeg75": [], "tta": [], "jpeg75+tta": []}
        probs = []
        for i in range(0, len(xs), 16):
            xb = xs[i:i + 16].to(device)
            xa = pgd(model, xb, eps, steps, alpha=eps / max(1, steps) * 2.5) if eps > 0 else xb
            for k, (j, t) in {"plain": (False, False), "jpeg75": (True, False), "tta": (False, True), "jpeg75+tta": (True, True)}.items():
                d, p = det_rate(model, xa, T, thr, jpeg=j, tta=t)
                rates[k].append(d)
                if k == "plain":
                    probs.append(p)
        results[name] = {"epsilon": eps, "steps": steps, **{k: float(np.concatenate(v).mean()) for k, v in rates.items()},
                         "mean_p_ai": float(np.concatenate(probs).mean())}
        print("   " + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in results[name].items()}), flush=True)
    # cost of mitigation on real images (false-positive rate)
    fpr = {}
    for k, (j, t) in {"plain": (False, False), "jpeg75": (True, False), "tta": (False, True), "jpeg75+tta": (True, True)}.items():
        ds = []
        for i in range(0, len(xr), 32):
            d, _ = det_rate(model, xr[i:i + 32].to(device), T, thr, jpeg=j, tta=t)
            ds.append(d)
        fpr[k] = float(np.concatenate(ds).mean())
    out = {"n_fake": len(fakes), "n_real": len(reals), "threshold": thr, "temperature": T, "detection_rate": results, "fpr_on_reals": fpr,
           "notes": ["White-box attacks assume the attacker has the model weights - a worst case.",
                     "Detection rate = share of AI-generated images still flagged after the attack.",
                     "JPEG pre-filtering and TTA are cheap inference-time mitigations; their cost is measured as FPR on real images."]}
    path = C.REPORT_DIR / "attacks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("\n| attack | detection (plain) | +JPEG75 | +TTA | +both |\n|---|---|---|---|---|")
    for k, r in results.items():
        print(f"| {k} | {r['plain']:.3f} | {r['jpeg75']:.3f} | {r['tta']:.3f} | {r['jpeg75+tta']:.3f} |")
    print(f"FPR on reals: {json.dumps({k: round(v, 3) for k, v in fpr.items()})}\n-> {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=C.DEFAULT_CHECKPOINT)
    ap.add_argument("--n-fake", type=int, default=300)
    ap.add_argument("--n-real", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    study(a.checkpoint, a.n_fake, a.n_real, a.workers)
