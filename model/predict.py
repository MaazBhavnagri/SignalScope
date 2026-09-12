"""SignalScope predict interface - the single inference path used by the CLI, the API and the evaluator.

CLI:
  python -m model.predict path/to/image.jpg                 # verdict + explanation JSON to stdout
  python -m model.predict img.jpg --heatmap out.png --json out.json --caption "a ceramic mug"
  python -m model.predict --dir folder/ --csv predictions.csv   # batch, one row per image (organiser format)

The CSV has: filename, p_ai_generated (calibrated), label (real|ai_generated), threshold
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import config as C
from model.utils import autocast_ctx
from model.cues import compute_cues, interpret_cues, load_reference
from model.data.datasets import to_input_tensor
from model.explain import GradCAM, build_explanation, cam_to_png_base64, probability_band, region_analysis, BAND_LABEL
from model.nets import load_checkpoint
from model.provenance import analyze_provenance, combine_with_visual

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def load_calibration(path: Path = C.CALIBRATION_PATH) -> dict:
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {"temperature": 1.0, "threshold": 0.5, "fitted": False}


class Detector:
    """Loads weights + calibration once; thread-safe for inference (a lock guards Grad-CAM's hooks)."""

    def __init__(self, checkpoint: Path = C.DEFAULT_CHECKPOINT, calibration: Path = C.CALIBRATION_PATH, device: str | None = None):
        import threading
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.ckpt = load_checkpoint(checkpoint, self.device)
        self.attr_classes: list[str] = list(self.ckpt.get("attr_classes", C.ATTRIBUTION_CLASSES))
        self.calib = load_calibration(calibration)
        self.T = float(self.calib.get("temperature", 1.0))
        self.threshold = float(self.calib.get("threshold", 0.5))
        self.cue_ref = load_reference()
        self.img_size = int(self.ckpt.get("config", {}).get("img_size", C.IMG_SIZE))
        self._lock = threading.Lock()
        self._cam = GradCAM(self.model, self.model.cam_layer())
        self._clip = None  # lazy (Bonus E)

    # ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def logits(self, x: torch.Tensor) -> dict:
        x = x.to(self.device)
        with autocast_ctx(self.device):
            out = self.model(x)
        return {"logit": out["logit"].float().cpu().numpy(), "attr_logits": out["attr_logits"].float().cpu().numpy()}

    def calibrated_prob(self, logit: float) -> float:
        return float(1.0 / (1.0 + np.exp(-logit / self.T)))

    def attribution(self, attr_logits: np.ndarray) -> dict:
        p = np.exp(attr_logits - attr_logits.max())
        p /= p.sum()
        dist = {g: float(p[i]) for i, g in enumerate(self.attr_classes)}
        fake_only = {g: v for g, v in dist.items() if g != "Real"}
        s = sum(fake_only.values()) or 1.0
        fake_only = {g: v / s for g, v in fake_only.items()}
        top = max(fake_only, key=fake_only.get)
        fam: dict[str, float] = {}
        for g, v in fake_only.items():
            fam[C.GENERATOR_FAMILY.get(g, "unknown")] = fam.get(C.GENERATOR_FAMILY.get(g, "unknown"), 0.0) + v
        top_fam = max(fam, key=fam.get)
        return {"generator": top, "family": top_fam, "confidence": float(fake_only[top]), "family_confidence": float(fam[top_fam]),
                "distribution": {g: round(v, 4) for g, v in sorted(fake_only.items(), key=lambda kv: -kv[1])},
                "family_distribution": {g: round(v, 4) for g, v in sorted(fam.items(), key=lambda kv: -kv[1])},
                "p_real_head": float(dist.get("Real", 0.0)),
                "known_generators": [g for g in self.attr_classes if g != "Real"],
                "note": "Generators never seen in training (e.g. Midjourney, VQDM in our protocol) are mapped to the closest known family; treat as a hint."}

    # ------------------------------------------------------------------------------------------------
    def analyze(self, img: Image.Image, raw_bytes: bytes | None = None, caption: str | None = None,
                heatmap: bool = True, tta: bool = True) -> dict:
        t0 = time.perf_counter()
        img = img.convert("RGB") if img.mode != "RGB" else img
        x = to_input_tensor(img, self.img_size)
        views = [x, torch.flip(x, dims=[3])] if tta else [x]
        out = self.logits(torch.cat(views))
        logit = float(out["logit"].mean())
        attr_logits = out["attr_logits"].mean(axis=0)
        prob = self.calibrated_prob(logit)
        prob_raw = float(1 / (1 + np.exp(-logit)))
        band = probability_band(prob, self.threshold)
        verdict = "ai_generated" if prob >= self.threshold else "real"
        t_model = time.perf_counter() - t0

        # explanation lanes
        cam_png, regions = None, None
        if heatmap:
            with self._lock:
                cam, _ = self._cam(x.to(self.device), target="fake")
            regions = region_analysis(cam)
            cam_png = cam_to_png_base64(cam, img.size)
        cues = interpret_cues(compute_cues(img), self.cue_ref)
        attribution = self.attribution(attr_logits)
        provenance = analyze_provenance(raw_bytes, img) if raw_bytes is not None else None
        combined = combine_with_visual(prob, provenance) if provenance else None
        consistency = self.caption_consistency(img, caption) if caption else None
        explanation = build_explanation(prob, self.threshold, cues, regions or _empty_regions(), attribution, provenance, consistency)

        return {
            "verdict": verdict,
            "label": explanation["label"],
            "band": band,
            "probability_ai": round(prob, 4),
            "probability_ai_uncalibrated": round(prob_raw, 4),
            "confidence": round(max(prob, 1 - prob), 4),
            "threshold": self.threshold,
            "temperature": self.T,
            "logit": round(logit, 4),
            "attribution": attribution if verdict == "ai_generated" or band in ("uncertain", "leaning_ai") else {**attribution, "applicable": False},
            "heatmap_png_base64": cam_png,
            "regions": regions,
            "cues": cues,
            "provenance": provenance,
            "combined": combined,
            "caption_consistency": consistency,
            "explanation": explanation,
            "image": {"width": img.size[0], "height": img.size[1]},
            "timing_ms": {"model": round(t_model * 1000, 1), "total": round((time.perf_counter() - t0) * 1000, 1)},
            "model": {"backbone": self.ckpt.get("config", {}).get("backbone", C.BACKBONE), "epoch": self.ckpt.get("epoch"),
                      "val_auc": self.ckpt.get("val_auc"), "held_out_generators": self.ckpt.get("held_out_generators", C.HELD_OUT_GENERATORS),
                      "calibrated": bool(self.calib.get("fitted", True)), "tta": tta, "device": self.device.type},
        }

    # ------------------------------------------------------------------------------------------------
    def caption_consistency(self, img: Image.Image, caption: str) -> dict | None:
        """Bonus E: OpenCLIP image-text agreement (generic captions only)."""
        try:
            from model.multimodal import CaptionChecker
            if self._clip is None:
                self._clip = CaptionChecker(device=self.device)
            return self._clip.check(img, caption)
        except Exception as e:  # noqa: BLE001 - optional module must never break the core verdict
            return {"band": None, "error": f"caption check unavailable: {e}"}

    # ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def predict_paths(self, paths: list[Path], batch_size: int = 64) -> np.ndarray:
        """Calibrated P(ai) for many files (no TTA, no explanation) - used for CSV batch mode."""
        probs = []
        for i in range(0, len(paths), batch_size):
            xs = []
            for p in paths[i: i + batch_size]:
                try:
                    xs.append(to_input_tensor(Image.open(p).convert("RGB"), self.img_size))
                except Exception:  # noqa: BLE001
                    xs.append(torch.zeros(1, 3, self.img_size, self.img_size))
            out = self.logits(torch.cat(xs))
            probs.extend([self.calibrated_prob(float(l)) for l in out["logit"]])
        return np.array(probs)


def _empty_regions() -> dict:
    return {"cells": [], "top_regions": ["-", "-"], "top2_share": 0.0, "concentration": "diffuse", "hot_bbox": None, "hot_area_fraction": 0.0}


# --------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="image file")
    ap.add_argument("--dir", type=Path, help="batch: folder of images")
    ap.add_argument("--csv", type=Path, help="batch: output CSV path")
    ap.add_argument("--json", type=Path, help="write full result JSON here")
    ap.add_argument("--heatmap", type=Path, help="write heat-map overlay PNG here")
    ap.add_argument("--caption", default=None)
    ap.add_argument("--checkpoint", type=Path, default=C.DEFAULT_CHECKPOINT)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    det = Detector(a.checkpoint, device=a.device)
    if a.dir:
        paths = sorted(p for p in a.dir.rglob("*") if p.suffix.lower() in IMG_EXT)
        probs = det.predict_paths(paths)
        out = a.csv or (a.dir / "signalscope_predictions.csv")
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "p_ai_generated", "label", "threshold"])
            for p, pr in zip(paths, probs):
                w.writerow([p.name, f"{pr:.6f}", "ai_generated" if pr >= det.threshold else "real", det.threshold])
        print(f"{len(paths)} images -> {out}")
        return
    if not a.image:
        ap.error("provide an image path or --dir")
    raw = Path(a.image).read_bytes()
    img = Image.open(io.BytesIO(raw))
    res = det.analyze(img, raw_bytes=raw, caption=a.caption, heatmap=True, tta=not a.no_tta)
    if a.heatmap and res["heatmap_png_base64"]:
        overlay = Image.open(io.BytesIO(base64.b64decode(res["heatmap_png_base64"]))).convert("RGBA")
        base = img.convert("RGBA")
        Image.alpha_composite(base, overlay).convert("RGB").save(a.heatmap)
    printable = {k: v for k, v in res.items() if k != "heatmap_png_base64"}
    if a.json:
        a.json.write_text(json.dumps(res, indent=2))
    print(json.dumps(printable, indent=2))
    print(f"\n==> {res['label']}  (P(ai) = {res['probability_ai']:.3f}, threshold {res['threshold']:.3f})", file=sys.stderr)
    for s in res["explanation"]["sentences"]:
        print(" - " + s, file=sys.stderr)


if __name__ == "__main__":
    main()
