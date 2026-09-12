"""Faithful explanation: Grad-CAM localisation + region analysis + grounded text (Bonus Module A).

Faithfulness rules implemented here:
  * The heat-map is Grad-CAM of the *AI-generated* logit on the RGB stream's last conv layer - it shows where
    the evidence for the verdict actually came from, not a generic saliency map.
  * Text only cites cues whose measured value is outside the real-photo reference range (see cues.py).
  * Localisation statements are derived from the heat-map mass distribution (concentrated vs diffuse).
  * Wording is driven by the calibrated probability band; nothing is asserted as certain.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import config as C

REGION_NAMES = [
    ["upper-left", "upper-centre", "upper-right"],
    ["middle-left", "centre", "middle-right"],
    ["lower-left", "lower-centre", "lower-right"],
]


class GradCAM:
    """Minimal, dependency-free Grad-CAM on one conv layer."""

    def __init__(self, model, layer: torch.nn.Module):
        self.model = model
        self.acts: torch.Tensor | None = None
        self.grads: torch.Tensor | None = None
        self._h1 = layer.register_forward_hook(self._fwd)
        self._h2 = layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def remove(self):
        self._h1.remove(); self._h2.remove()

    def __call__(self, x: torch.Tensor, target: str = "fake") -> tuple[np.ndarray, dict]:
        """Returns (cam HxW in [0,1] at input resolution, model outputs dict)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        out = self.model(x)
        score = out["logit"] if target == "fake" else -out["logit"]
        score.sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        raw_max = float(cam.max())
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), {"logit": float(out["logit"][0]), "attr_logits": out["attr_logits"][0].detach().float().cpu().numpy(), "cam_raw_max": raw_max}


# --------------------------------------------------------------------------------------------------------
def region_analysis(cam: np.ndarray) -> dict:
    """Split the heat-map into a 3x3 grid; report the share of activation mass per region + concentration."""
    h, w = cam.shape
    total = cam.sum() + 1e-8
    cells = []
    for r in range(3):
        for c in range(3):
            block = cam[r * h // 3:(r + 1) * h // 3, c * w // 3:(c + 1) * w // 3]
            cells.append({"region": REGION_NAMES[r][c], "row": r, "col": c, "share": float(block.sum() / total),
                          "peak": float(block.max())})
    cells.sort(key=lambda d: -d["share"])
    top2 = cells[0]["share"] + cells[1]["share"]
    # 9 equal cells -> uniform share 0.111; >0.45 in two cells means clearly concentrated
    concentration = "concentrated" if top2 > 0.45 else ("moderate" if top2 > 0.32 else "diffuse")
    # bounding box of the >0.6 activation area (for the UI reticle), normalised 0..1
    ys, xs = np.where(cam >= 0.6)
    bbox = None
    if len(xs):
        bbox = [float(xs.min() / w), float(ys.min() / h), float((xs.max() + 1) / w), float((ys.max() + 1) / h)]
    return {"cells": cells, "top_regions": [c["region"] for c in cells[:2]], "top2_share": float(top2),
            "concentration": concentration, "hot_bbox": bbox, "hot_area_fraction": float(len(xs) / (h * w))}


def cam_to_png_base64(cam: np.ndarray, size: tuple[int, int]) -> str:
    """Heat-map as a transparent RGBA PNG (amber->red ramp, alpha ~ activation) resized to the image size."""
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    a = np.asarray(cam_img, dtype=np.float32) / 255.0
    # colour ramp: low = cool transparent, high = hot amber/red
    r = np.clip(1.2 * a + 0.2, 0, 1)
    g = np.clip(0.9 - 0.8 * a, 0, 1) * (0.4 + 0.6 * a)
    b = np.clip(0.6 - 1.2 * a, 0, 1) * 0.6
    alpha = np.clip(a ** 1.3 * 0.85, 0, 0.85)
    rgba = np.stack([r, g, b, alpha], axis=-1)
    out = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")
    buf = io.BytesIO()
    out.save(buf, "PNG", compress_level=6)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------------------------------------
def probability_band(p: float, threshold: float) -> str:
    """likely_real | leaning_real | uncertain | leaning_ai | likely_ai"""
    d = p - threshold
    if abs(d) < C.UNCERTAIN_BAND * 0.5:
        return "uncertain"
    if d < 0:
        return "leaning_real" if abs(d) < C.UNCERTAIN_BAND else "likely_real"
    return "leaning_ai" if d < C.UNCERTAIN_BAND else "likely_ai"


BAND_LABEL = {
    "likely_ai": "Likely AI-generated",
    "leaning_ai": "Possibly AI-generated",
    "uncertain": "Inconclusive",
    "leaning_real": "Possibly a real photograph",
    "likely_real": "Likely a real photograph",
}


def build_explanation(prob: float, threshold: float, cues: list[dict], regions: dict, attribution: dict | None,
                      provenance: dict | None = None, consistency: dict | None = None) -> dict:
    band = probability_band(prob, threshold)
    fired = [c for c in cues if c["fired"]]
    fired_syn = [c for c in fired if c["id"] != "blockiness"]
    sentences: list[str] = []

    # 1. headline (hedged, never an accusation)
    pct = round(prob * 100)
    if band == "likely_ai":
        sentences.append(f"The detector assigns a {pct}% calibrated likelihood that this image is AI-generated. This is a likelihood estimate, not proof.")
    elif band == "leaning_ai":
        sentences.append(f"The detector leans towards AI-generated ({pct}% calibrated likelihood), but the margin is modest - treat this as a prompt for closer inspection.")
    elif band == "uncertain":
        sentences.append(f"The detector cannot separate this image from real photographs with confidence ({pct}% likelihood of AI generation, close to the decision threshold).")
    elif band == "leaning_real":
        sentences.append(f"The detector leans towards a real photograph ({100 - pct}% calibrated likelihood of being real), with limited margin.")
    else:
        sentences.append(f"The detector found no strong synthetic signature; it estimates a {100 - pct}% calibrated likelihood that this is a real photograph.")

    # 2. localisation from the heat-map
    if band in ("likely_ai", "leaning_ai", "uncertain"):
        top = " and ".join(regions["top_regions"])
        if regions["concentration"] == "concentrated":
            sentences.append(f"The evidence the model relied on is concentrated in the {top} of the frame (about {round(regions['top2_share'] * 100)}% of the heat-map mass); inspect that area first.")
        elif regions["concentration"] == "moderate":
            sentences.append(f"The strongest signal is around the {top}, though the model also drew on other areas.")
        else:
            sentences.append("The signal is spread across the whole frame rather than one region - consistent with a global texture/noise pattern rather than a single local flaw.")
    else:
        sentences.append("The heat-map shows only weak, scattered activation - no region stood out as synthetic.")

    # 3. measurable cues that actually fired
    if fired_syn:
        lead = "Measurable cues outside the real-photo range: " if band in ("likely_ai", "leaning_ai", "uncertain") else "Note: some pixel statistics are atypical for photographs: "
        parts = [f"{c['name'].lower()} - {c['finding']} (measured {c['value']}, real photos typically {c['typical_range'][0]} to {c['typical_range'][1]})" for c in fired_syn[:3]]
        sentences.append(lead + "; ".join(parts) + ".")
    elif band in ("likely_ai", "leaning_ai"):
        sentences.append("None of the hand-crafted statistics fell outside the real-photo range; the verdict rests on learned texture and noise patterns that are not captured by simple statistics, so weigh it accordingly.")
    else:
        sentences.append("All measured pixel statistics (noise, spectrum, colour, sharpness) are within the range of real photographs.")

    # 4. compression caveat
    blk = next((c for c in cues if c["id"] == "blockiness"), None)
    if blk and blk["fired"]:
        sentences.append("The file shows heavy JPEG re-compression, which weakens forensic traces; confidence should be read as lower than the number alone suggests.")

    # 5. attribution (hedged)
    if attribution and band in ("likely_ai", "leaning_ai"):
        if attribution.get("confidence", 0) >= 0.5:
            sentences.append(f"Artefact pattern is most consistent with a {attribution['family']} generator (closest known: {attribution['generator']}, {round(attribution['confidence'] * 100)}% of attribution mass). Unseen generators map to their nearest known family, so treat this as a family-level hint.")
        else:
            sentences.append("The artefact pattern does not match any generator seen in training closely - possibly a newer or unseen generator.")

    # 6. provenance and caption lanes
    if provenance:
        if provenance.get("signal") == "supports_ai":
            sentences.append("Embedded metadata independently indicates AI generation: " + "; ".join(provenance.get("reasons", [])[:2]) + ".")
        elif provenance.get("signal") == "supports_real":
            sentences.append("Metadata is consistent with a camera capture (" + "; ".join(provenance.get("reasons", [])[:2]) + "), but metadata can be copied or forged and does not override the visual analysis.")
        elif provenance.get("signal") == "stripped":
            sentences.append("No camera or software metadata is present - common after social-media re-uploads, so it neither supports nor weakens the verdict.")
    if consistency and consistency.get("band"):
        sentences.append(consistency["sentence"])

    return {"band": band, "label": BAND_LABEL[band], "summary": sentences[0], "sentences": sentences,
            "fired_cues": [c["id"] for c in fired_syn]}
