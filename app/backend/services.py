"""Application services: image validation, storage, running the detector, persistence helpers."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import threading
from pathlib import Path

from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.backend import models
from app.backend.db import HEATMAPS_DIR, ORIGINALS_DIR
from model import config as C

log = logging.getLogger("signalscope")

MAX_BYTES = int(os.environ.get("SIGNALSCOPE_MAX_UPLOAD_MB", "20")) * 1024 * 1024
MAX_PIXELS = 40_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF"}
ALLOWED_CT = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff", "image/gif", "application/octet-stream", None, ""}


class ModelState:
    """Lazily loaded singleton detector so the API can start (and report health) even without weights."""

    def __init__(self):
        self.detector = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self.checkpoint = Path(os.environ.get("SIGNALSCOPE_CHECKPOINT", C.DEFAULT_CHECKPOINT))

    def load(self):
        with self._lock:
            if self.detector is not None:
                return self.detector
            try:
                from model.predict import Detector
                if not self.checkpoint.exists():
                    raise FileNotFoundError(f"checkpoint not found: {self.checkpoint} (train with `python -m model.train` or download weights, see README)")
                self.detector = Detector(self.checkpoint)
                self.error = None
                log.info("model loaded from %s on %s", self.checkpoint, self.detector.device)
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
                log.exception("model load failed")
            return self.detector

    def require(self):
        det = self.load()
        if det is None:
            raise HTTPException(status_code=503, detail={"error": "model_unavailable", "message": self.error})
        return det

    def info(self) -> dict:
        det = self.detector
        return {"loaded": det is not None, "error": self.error, "checkpoint": str(self.checkpoint),
                "device": det.device.type if det else None, "threshold": det.threshold if det else None,
                "temperature": det.T if det else None, "epoch": det.ckpt.get("epoch") if det else None,
                "val_auc": det.ckpt.get("val_auc") if det else None}


MODEL = ModelState()


# --------------------------------------------------------------------------------------------------------
def validate_and_open(raw: bytes, filename: str, content_type: str | None) -> Image.Image:
    if not raw:
        raise HTTPException(400, detail={"error": "empty_file", "message": "The uploaded file is empty."})
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, detail={"error": "file_too_large", "message": f"File exceeds {MAX_BYTES // (1024*1024)} MB limit."})
    if content_type not in ALLOWED_CT and not (content_type or "").startswith("image/"):
        raise HTTPException(415, detail={"error": "unsupported_type", "message": f"Unsupported content type '{content_type}'."})
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(415, detail={"error": "not_an_image", "message": f"'{filename}' could not be decoded as an image ({e})."})
    if img.format not in ALLOWED_FORMATS:
        raise HTTPException(415, detail={"error": "unsupported_format", "message": f"Format {img.format} is not supported."})
    if img.width * img.height > MAX_PIXELS:
        raise HTTPException(413, detail={"error": "image_too_large", "message": "Image has too many pixels (limit 40 MP)."})
    if img.width < 16 or img.height < 16:
        raise HTTPException(400, detail={"error": "image_too_small", "message": "Image must be at least 16x16 pixels."})
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    img = ImageOps.exif_transpose(img)
    return img


def run_analysis(db: Session, raw: bytes, filename: str, content_type: str | None, caption: str | None = None,
                 want_heatmap: bool = True, batch_id: str | None = None) -> models.Analysis:
    det = MODEL.require()
    img = validate_and_open(raw, filename, content_type)
    sha = hashlib.sha256(raw).hexdigest()
    res = det.analyze(img, raw_bytes=raw, caption=caption or None, heatmap=want_heatmap, tta=True)

    a = models.Analysis(
        filename=filename[:300], content_type=content_type, sha256=sha, size_bytes=len(raw), width=img.width, height=img.height,
        verdict=res["verdict"], label=res["label"], band=res["band"], probability_ai=res["probability_ai"],
        probability_raw=res["probability_ai_uncalibrated"], confidence=res["confidence"], threshold=res["threshold"],
        attribution=res["attribution"], cues=res["cues"], regions=res["regions"], provenance=res["provenance"],
        combined=res["combined"], caption=caption or None, caption_consistency=res["caption_consistency"],
        explanation=res["explanation"], timing_ms=res["timing_ms"], model_info=res["model"], batch_id=batch_id,
    )
    db.add(a)
    db.flush()  # get id
    # persist files
    ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp", "TIFF": ".tif"}.get(img.format or "", ".img")
    orig = ORIGINALS_DIR / f"{a.id}{ext}"
    orig.write_bytes(raw)
    a.original_path = str(orig)
    if res.get("heatmap_png_base64"):
        hm = HEATMAPS_DIR / f"{a.id}.png"
        hm.write_bytes(base64.b64decode(res["heatmap_png_base64"]))
        a.heatmap_path = str(hm)
    db.commit()
    return a


def load_report_assets() -> dict:
    """Metrics / robustness / attacks / calibration / training history for the model card."""
    out = {}
    for key, path in {"metrics": C.METRICS_PATH, "robustness": C.REPORT_DIR / "robustness.json", "attacks": C.REPORT_DIR / "attacks.json",
                      "calibration": C.CALIBRATION_PATH, "train_history": C.WEIGHTS_DIR / "train_history.json",
                      "cue_reference": C.WEIGHTS_DIR / "cue_reference.json", "dataset_card": C.REPORT_DIR / "dataset_card.json"}.items():
        try:
            out[key] = json.loads(Path(path).read_text()) if Path(path).exists() else None
        except Exception as e:  # noqa: BLE001
            out[key] = {"error": str(e)}
    return out
