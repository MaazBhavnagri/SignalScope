"""Grounded forensic cue statistics.

Each cue is a *measurable* image statistic. A cue "fires" when its value falls outside the range observed on
real photographs (reference percentiles fitted on the validation REAL images by `python -m model.cues fit`).
This keeps the natural-language explanation verifiable: every sentence maps to a number the judge can inspect.

Cues (all computed on a <=512px grayscale/RGB copy):
  hf_energy            share of spectral energy in the top third of frequencies
  spectral_slope       slope of log-power vs log-frequency (natural photos ~ -2 .. -3)
  noise_level          median std of the noise residual (image - 3x3 median) over 16x16 cells
  residual_uniformity  coefficient of variation of residual std across cells (camera noise is spatially consistent)
  residual_periodicity peak normalised autocorrelation of the residual at non-zero lag (upsampling patterns)
  channel_correlation  mean correlation between R/G/B noise residuals (camera demosaicing leaves a signature)
  saturation_extreme   fraction of pixels with HSV saturation > 0.9
  laplacian_var        sharpness measure
  blockiness           8x8 JPEG grid strength (compression history, used for the robustness note)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from model import config as C

CUE_REFERENCE_PATH = C.WEIGHTS_DIR / "cue_reference.json"

# Human-readable metadata. `direction` = which side of the real-photo range counts as a synthetic-leaning cue.
#   "high" -> firing when value > p95, "low" -> value < p5, "both" -> either side.
CUE_META = {
    "hf_energy": dict(name="High-frequency detail", direction="both",
                      high="unusually high high-frequency energy (over-sharpened / synthetic micro-texture)",
                      low="unusually little fine high-frequency detail (over-smooth, 'airbrushed' surfaces)"),
    "spectral_slope": dict(name="Spectral slope", direction="both",
                           high="power spectrum falls off slower than natural photographs",
                           low="power spectrum falls off faster than natural photographs"),
    "noise_level": dict(name="Sensor-noise level", direction="low",
                        low="almost no sensor-like noise - real cameras leave a faint grain everywhere",
                        high="very strong noise"),
    "residual_uniformity": dict(name="Noise consistency across regions", direction="high",
                                high="noise is patchy: some regions are far smoother than others, which cameras rarely produce",
                                low="noise is unusually uniform"),
    "residual_periodicity": dict(name="Repeating micro-pattern", direction="high",
                                 high="a repeating grid-like micro-pattern in the noise (typical of upsampling layers in generators)",
                                 low=""),
    "channel_correlation": dict(name="Colour-channel noise coupling", direction="both",
                                high="noise in R/G/B channels is almost identical - camera demosaicing normally decorrelates it",
                                low="noise in colour channels is unusually independent"),
    "saturation_extreme": dict(name="Colour saturation", direction="high",
                               high="a large share of extremely saturated pixels (a common generator look)",
                               low=""),
    "laplacian_var": dict(name="Edge sharpness", direction="both",
                          high="edges are uniformly razor-sharp across the whole frame (no depth-of-field falloff)",
                          low="the whole image is soft"),
    "blockiness": dict(name="JPEG block grid", direction="high",
                       high="strong 8x8 compression grid (heavily re-compressed image; forensic traces may be weakened)",
                       low=""),
}

# Fallback ranges (5th, 95th percentile) used only if cue_reference.json has not been fitted yet.
_DEFAULT_REF = {
    "hf_energy": [0.02, 0.35], "spectral_slope": [-3.2, -1.6], "noise_level": [1.2, 9.0],
    "residual_uniformity": [0.25, 1.1], "residual_periodicity": [0.0, 0.35], "channel_correlation": [0.2, 0.9],
    "saturation_extreme": [0.0, 0.12], "laplacian_var": [20.0, 2500.0], "blockiness": [0.0, 1.4],
}


# --------------------------------------------------------------------------------------------------------
def _prep(img: Image.Image, max_side: int = 512) -> tuple[np.ndarray, np.ndarray]:
    img = img.convert("RGB")
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((max(16, int(w * s)), max(16, int(h * s))), Image.BICUBIC)
    rgb = np.asarray(img, dtype=np.float32)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return rgb, gray


def _residual(img_arr: np.ndarray) -> np.ndarray:
    """Noise residual = image - 3x3 median filter (per channel)."""
    if img_arr.ndim == 2:
        pil = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
        med = np.asarray(pil.filter(ImageFilter.MedianFilter(3)), dtype=np.float32)
        return img_arr - med
    out = np.empty_like(img_arr)
    for c in range(img_arr.shape[2]):
        out[..., c] = _residual(img_arr[..., c])
    return out


def _cell_std(res: np.ndarray, cells: int = 16) -> np.ndarray:
    h, w = res.shape
    ch, cw = max(1, h // cells), max(1, w // cells)
    res = res[: ch * cells, : cw * cells].reshape(cells, ch, cells, cw)
    return res.std(axis=(1, 3))


def _radial_profile(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = gray - gray.mean()
    f = np.fft.fftshift(np.fft.fft2(g))
    p = np.abs(f) ** 2
    h, w = p.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(p.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    rmax = min(cy, cx)
    tbin = np.bincount(r.ravel(), p.ravel())[:rmax]
    nr = np.bincount(r.ravel())[:rmax]
    prof = tbin / np.maximum(nr, 1)
    return np.arange(rmax), prof


def compute_cues(img: Image.Image) -> dict[str, float]:
    rgb, gray = _prep(img)
    h, w = gray.shape
    out: dict[str, float] = {}

    # spectral cues
    freqs, prof = _radial_profile(gray)
    total = prof[1:].sum() + 1e-9
    cut = int(len(prof) * 2 / 3)
    out["hf_energy"] = float(prof[cut:].sum() / total)
    lo, hi = max(2, int(len(prof) * 0.05)), max(3, int(len(prof) * 0.9))
    x, y = np.log(freqs[lo:hi] + 1e-9), np.log(prof[lo:hi] + 1e-9)
    out["spectral_slope"] = float(np.polyfit(x, y, 1)[0]) if len(x) > 2 else -2.0

    # residual cues
    res_rgb = _residual(rgb)
    res = res_rgb.mean(axis=2)
    cs = _cell_std(res)
    out["noise_level"] = float(np.median(cs))
    out["residual_uniformity"] = float(cs.std() / (cs.mean() + 1e-6))
    # periodicity via autocorrelation of residual (normalised, exclude a 3px centre)
    rz = res - res.mean()
    fr = np.fft.fft2(rz)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(fr * np.conj(fr))))
    ac /= (ac.max() + 1e-9)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    ac[cy - 2: cy + 3, cx - 2: cx + 3] = 0
    win = ac[max(0, cy - 40): cy + 41, max(0, cx - 40): cx + 41]
    out["residual_periodicity"] = float(np.clip(win.max(), 0, 1))
    # channel coupling
    flat = res_rgb.reshape(-1, 3)
    cc = np.corrcoef(flat.T)
    out["channel_correlation"] = float(np.nanmean([cc[0, 1], cc[0, 2], cc[1, 2]]))

    # colour & sharpness
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    out["saturation_extreme"] = float((sat > 0.9).mean())
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
    out["laplacian_var"] = float(lap.var())

    # JPEG blockiness: boundary vs interior horizontal differences on the 8-px grid
    d = np.abs(np.diff(gray, axis=1))
    if d.shape[1] >= 16:
        cols = np.arange(d.shape[1])
        b = d[:, (cols % 8) == 7].mean()
        i = d[:, (cols % 8) != 7].mean()
        out["blockiness"] = float(b / (i + 1e-6))
    else:
        out["blockiness"] = 1.0
    return out


# --------------------------------------------------------------------------------------------------------
def load_reference(path: Path = CUE_REFERENCE_PATH) -> dict:
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {"ranges": _DEFAULT_REF, "fitted_on": None}


def interpret_cues(values: dict[str, float], reference: dict | None = None) -> list[dict]:
    """Turn raw values into ranked, human-readable cue records with a 0..1 strength."""
    ref = (reference or load_reference())["ranges"]
    recs = []
    for k, v in values.items():
        meta = CUE_META.get(k)
        if meta is None or k not in ref:
            continue
        p5, p95 = ref[k]
        span = max(p95 - p5, 1e-6)
        side, strength = None, 0.0
        if v > p95 and meta["direction"] in ("high", "both") and meta["high"]:
            side, strength = "high", min(1.0, (v - p95) / span)
        elif v < p5 and meta["direction"] in ("low", "both") and meta["low"]:
            side, strength = "low", min(1.0, (p5 - v) / span)
        recs.append({
            "id": k, "name": meta["name"], "value": round(float(v), 4),
            "typical_range": [round(float(p5), 4), round(float(p95), 4)],
            "fired": side is not None, "side": side, "strength": round(float(strength), 3),
            "finding": meta[side] if side else "within the range seen on real photographs",
        })
    recs.sort(key=lambda r: (-int(r["fired"]), -r["strength"]))
    return recs


def fit_reference(max_images: int = 1500, seed: int = C.SEED) -> dict:
    """Fit 5th/95th percentiles of every cue on REAL validation images (never test images)."""
    from model.data.datasets import load_manifest
    rows = [r for r in load_manifest(split="val") if r["label"] == C.LABEL_REAL]
    # prefer high-res reals; 32px CIFAKE images are not representative for pixel statistics
    hi = [r for r in rows if r["source"] != "cifake"] or rows
    rng = np.random.default_rng(seed)
    rng.shuffle(hi)
    hi = hi[:max_images]
    vals: dict[str, list[float]] = {k: [] for k in CUE_META}
    for i, r in enumerate(hi):
        c = compute_cues(Image.open(r["path"]))
        for k, v in c.items():
            if np.isfinite(v):
                vals[k].append(v)
        if i % 200 == 0:
            print(f"  {i}/{len(hi)}", flush=True)
    ranges = {k: [float(np.percentile(v, 5)), float(np.percentile(v, 95))] for k, v in vals.items() if v}
    ref = {"ranges": ranges, "fitted_on": {"n_images": len(hi), "sources": sorted({r['source'] for r in hi}), "split": "val", "label": "real"}}
    CUE_REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUE_REFERENCE_PATH.write_text(json.dumps(ref, indent=2))
    print(f"cue reference -> {CUE_REFERENCE_PATH}")
    return ref


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--max-images", type=int, default=1500)
    s = sub.add_parser("show"); s.add_argument("image")
    a = ap.parse_args()
    if a.cmd == "fit":
        fit_reference(a.max_images)
    else:
        for rec in interpret_cues(compute_cues(Image.open(a.image))):
            print(json.dumps(rec))
