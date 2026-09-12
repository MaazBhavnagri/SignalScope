from __future__ import annotations

import io
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.backend import models
from app.backend.db import get_db
from app.backend.report import render_report_html
from app.backend.services import MODEL, load_report_assets, run_analysis
from model import config as C

router = APIRouter()


# --------------------------------------------------------------------------------------------------------
@router.get("/health")
def health():
    return {"status": "ok", "model": MODEL.info(), "time": datetime.now(timezone.utc).isoformat()}


@router.get("/model")
def model_card():
    assets = load_report_assets()
    return {"model": MODEL.info(), **assets, "protocol": {"held_out_generators": C.HELD_OUT_GENERATORS, "seen_generators": [g for g in C.SEEN_GENERATORS if g != "Real"],
            "attribution_classes": C.ATTRIBUTION_CLASSES, "families": C.GENERATOR_FAMILY, "target_fpr": C.TARGET_FPR, "uncertain_band": C.UNCERTAIN_BAND}}


# --------------------------------------------------------------------------------------------------------
@router.post("/analyze")
async def analyze(file: UploadFile = File(...), caption: str | None = Form(default=None), heatmap: bool = Form(default=True),
                  db: Session = Depends(get_db)):
    raw = await file.read()
    # heavy, blocking work runs in the thread pool so the event loop keeps serving other requests
    a = await run_in_threadpool(run_analysis, db, raw, file.filename or "upload", file.content_type, caption=caption, want_heatmap=heatmap)
    return a.to_full()


@router.post("/analyze/batch")
async def analyze_batch(files: list[UploadFile] = File(...), name: str | None = Form(default=None), db: Session = Depends(get_db)):
    if not files:
        raise HTTPException(400, detail={"error": "no_files", "message": "Upload at least one image."})
    if len(files) > 50:
        raise HTTPException(400, detail={"error": "too_many_files", "message": "Batch limit is 50 images."})
    batch = models.Batch(name=name)
    db.add(batch)
    db.flush()
    t0 = time.perf_counter()
    items, errors = [], []
    for f in files:
        raw = await f.read()
        try:
            a = await run_in_threadpool(run_analysis, db, raw, f.filename or "upload", f.content_type, want_heatmap=True, batch_id=batch.id)
            items.append(a.to_summary())
        except HTTPException as e:
            errors.append({"filename": f.filename, "detail": e.detail})
    batch.n_items = len(items)
    batch.n_ai = sum(1 for i in items if i["verdict"] == "ai_generated" and i["band"] != "uncertain")
    batch.n_real = sum(1 for i in items if i["verdict"] == "real" and i["band"] != "uncertain")
    batch.n_uncertain = sum(1 for i in items if i["band"] == "uncertain")
    batch.total_ms = (time.perf_counter() - t0) * 1000
    db.commit()
    return {"batch": _batch_dict(batch), "items": items, "errors": errors}


def _batch_dict(b: models.Batch) -> dict:
    return {"id": b.id, "name": b.name, "created_at": b.created_at.isoformat() if b.created_at else None, "n_items": b.n_items,
            "n_ai": b.n_ai, "n_real": b.n_real, "n_uncertain": b.n_uncertain, "total_ms": b.total_ms}


@router.get("/batches")
def list_batches(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    rows = db.scalars(select(models.Batch).order_by(models.Batch.created_at.desc()).limit(limit)).all()
    return {"items": [_batch_dict(b) for b in rows]}


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str, db: Session = Depends(get_db)):
    b = db.get(models.Batch, batch_id)
    if not b:
        raise HTTPException(404, detail={"error": "not_found", "message": "Batch not found."})
    return {"batch": _batch_dict(b), "items": [a.to_summary() for a in sorted(b.analyses, key=lambda a: a.created_at)]}


# --------------------------------------------------------------------------------------------------------
@router.get("/analyses")
def list_analyses(limit: int = Query(24, ge=1, le=200), offset: int = Query(0, ge=0),
                  verdict: Literal["real", "ai_generated"] | None = None, band: str | None = None,
                  q: str | None = Query(None, max_length=100), batch_id: str | None = None, review: str | None = None,
                  sort: Literal["newest", "oldest", "p_desc", "p_asc"] = "newest", db: Session = Depends(get_db)):
    stmt = select(models.Analysis)
    if verdict:
        stmt = stmt.where(models.Analysis.verdict == verdict)
    if band:
        stmt = stmt.where(models.Analysis.band.in_(band.split(",")))
    if q:
        stmt = stmt.where(models.Analysis.filename.ilike(f"%{q}%"))
    if batch_id:
        stmt = stmt.where(models.Analysis.batch_id == batch_id)
    if review == "pending":
        stmt = stmt.where(models.Analysis.review_decision.is_(None))
    elif review:
        stmt = stmt.where(models.Analysis.review_decision == review)
    total = db.scalar(select(func.count()).select_from(stmt.subquery()))
    order = {"newest": models.Analysis.created_at.desc(), "oldest": models.Analysis.created_at.asc(),
             "p_desc": models.Analysis.probability_ai.desc(), "p_asc": models.Analysis.probability_ai.asc()}[sort]
    rows = db.scalars(stmt.order_by(order).offset(offset).limit(limit)).all()
    return {"total": total, "items": [a.to_summary() for a in rows], "limit": limit, "offset": offset}


def _get(db: Session, analysis_id: str) -> models.Analysis:
    a = db.get(models.Analysis, analysis_id)
    if not a:
        raise HTTPException(404, detail={"error": "not_found", "message": "Analysis not found."})
    return a


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    return _get(db, analysis_id).to_full()


@router.delete("/analyses/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: str, db: Session = Depends(get_db)):
    a = _get(db, analysis_id)
    for p in (a.original_path, a.heatmap_path):
        if p and Path(p).exists():
            try:
                Path(p).unlink()
            except OSError:
                pass
    db.delete(a)
    db.commit()
    return JSONResponse(status_code=204, content=None)


class ReviewIn(BaseModel):
    decision: Literal["confirmed_ai", "confirmed_real", "unresolved"] | None = None
    note: str | None = Field(default=None, max_length=2000)


@router.patch("/analyses/{analysis_id}/review")
def review(analysis_id: str, body: ReviewIn, db: Session = Depends(get_db)):
    a = _get(db, analysis_id)
    a.review_decision = body.decision
    a.review_note = body.note
    a.reviewed_at = datetime.now(timezone.utc) if body.decision else None
    db.commit()
    return a.to_full()


@router.post("/analyses/{analysis_id}/robustness")
def robustness(analysis_id: str, db: Session = Depends(get_db)):
    from PIL import Image, ImageOps
    from model.robustness import stress_test
    a = _get(db, analysis_id)
    if not a.original_path or not Path(a.original_path).exists():
        raise HTTPException(410, detail={"error": "original_missing", "message": "Original file no longer stored."})
    det = MODEL.require()
    img = ImageOps.exif_transpose(Image.open(a.original_path)).convert("RGB")
    res = stress_test(det, img)
    a.robustness = res
    db.commit()
    return res


@router.get("/analyses/{analysis_id}/image")
def get_image(analysis_id: str, db: Session = Depends(get_db)):
    a = _get(db, analysis_id)
    if not a.original_path or not Path(a.original_path).exists():
        raise HTTPException(404, detail={"error": "not_found", "message": "Original file not found."})
    return FileResponse(a.original_path)


@router.get("/analyses/{analysis_id}/heatmap")
def get_heatmap(analysis_id: str, db: Session = Depends(get_db)):
    a = _get(db, analysis_id)
    if not a.heatmap_path or not Path(a.heatmap_path).exists():
        raise HTTPException(404, detail={"error": "not_found", "message": "Heat-map not available."})
    return FileResponse(a.heatmap_path, media_type="image/png")


@router.get("/analyses/{analysis_id}/report")
def report(analysis_id: str, format: Literal["html", "json"] = "html", db: Session = Depends(get_db)):
    a = _get(db, analysis_id)
    if format == "json":
        return JSONResponse(content={"report": a.to_full(), "generated_at": datetime.now(timezone.utc).isoformat(), "tool": "SignalScope 1.0"},
                            headers={"Content-Disposition": f'attachment; filename="signalscope-{a.id}.json"'})
    return HTMLResponse(render_report_html(a))


# --------------------------------------------------------------------------------------------------------
@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(models.Analysis.id))) or 0
    by_verdict = dict(db.execute(select(models.Analysis.verdict, func.count()).group_by(models.Analysis.verdict)).all())
    by_band = dict(db.execute(select(models.Analysis.band, func.count()).group_by(models.Analysis.band)).all())
    by_review = dict(db.execute(select(models.Analysis.review_decision, func.count()).group_by(models.Analysis.review_decision)).all())
    avg_p = db.scalar(select(func.avg(models.Analysis.probability_ai))) or 0.0
    since = datetime.now(timezone.utc) - timedelta(days=14)
    recent = db.execute(select(models.Analysis.created_at, models.Analysis.verdict).where(models.Analysis.created_at >= since)).all()
    per_day: dict[str, dict[str, int]] = {}
    for created, verdict in recent:
        d = created.date().isoformat() if created else "unknown"
        per_day.setdefault(d, {"real": 0, "ai_generated": 0})
        per_day[d][verdict] = per_day[d].get(verdict, 0) + 1
    lat = [t.get("total") for (t,) in db.execute(select(models.Analysis.timing_ms).order_by(models.Analysis.created_at.desc()).limit(200)).all() if t and t.get("total")]
    fam = {}
    for (attr, verdict) in db.execute(select(models.Analysis.attribution, models.Analysis.verdict).where(models.Analysis.verdict == "ai_generated")).all():
        if attr and attr.get("family"):
            fam[attr["family"]] = fam.get(attr["family"], 0) + 1
    return {"total": total, "by_verdict": by_verdict, "by_band": by_band, "by_review": {str(k): v for k, v in by_review.items()},
            "mean_probability_ai": avg_p, "per_day": dict(sorted(per_day.items())),
            "latency_ms": {"median": sorted(lat)[len(lat) // 2] if lat else None, "n": len(lat)}, "families": fam,
            "batches": db.scalar(select(func.count(models.Batch.id))) or 0}
