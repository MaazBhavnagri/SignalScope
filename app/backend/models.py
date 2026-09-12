from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.backend.db import Base


def _uuid() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Batch(Base):
    __tablename__ = "batches"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    n_items: Mapped[int] = mapped_column(Integer, default=0)
    n_ai: Mapped[int] = mapped_column(Integer, default=0)
    n_real: Mapped[int] = mapped_column(Integer, default=0)
    n_uncertain: Mapped[int] = mapped_column(Integer, default=0)
    total_ms: Mapped[float] = mapped_column(Float, default=0.0)
    analyses: Mapped[list["Analysis"]] = relationship(back_populates="batch", cascade="all, delete-orphan")


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    batch_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("batches.id", ondelete="CASCADE"), nullable=True, index=True)

    filename: Mapped[str] = mapped_column(String(300))
    content_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)

    verdict: Mapped[str] = mapped_column(String(20), index=True)   # real | ai_generated
    label: Mapped[str] = mapped_column(String(60))
    band: Mapped[str] = mapped_column(String(20), index=True)
    probability_ai: Mapped[float] = mapped_column(Float, index=True)
    probability_raw: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)

    attribution: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cues: Mapped[list | None] = mapped_column(JSON, nullable=True)
    regions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    provenance: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    combined: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption_consistency: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    explanation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    robustness: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timing_ms: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_info: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    original_path: Mapped[str | None] = mapped_column(String(400), nullable=True)
    heatmap_path: Mapped[str | None] = mapped_column(String(400), nullable=True)

    # human-in-the-loop review
    review_decision: Mapped[str | None] = mapped_column(String(30), nullable=True)  # confirmed_ai | confirmed_real | unresolved
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    batch: Mapped[Batch | None] = relationship(back_populates="analyses")

    def to_summary(self) -> dict:
        return {
            "id": self.id, "created_at": self.created_at.isoformat() if self.created_at else None, "batch_id": self.batch_id,
            "filename": self.filename, "width": self.width, "height": self.height, "size_bytes": self.size_bytes,
            "verdict": self.verdict, "label": self.label, "band": self.band, "probability_ai": self.probability_ai,
            "confidence": self.confidence, "threshold": self.threshold,
            "attribution_family": (self.attribution or {}).get("family") if self.verdict == "ai_generated" else None,
            "provenance_signal": (self.provenance or {}).get("signal"),
            "fired_cues": (self.explanation or {}).get("fired_cues", []),
            "review_decision": self.review_decision,
            "has_robustness": self.robustness is not None,
            "stability": (self.robustness or {}).get("stability"),
            "latency_ms": (self.timing_ms or {}).get("total"),
            "image_url": f"/api/analyses/{self.id}/image", "heatmap_url": f"/api/analyses/{self.id}/heatmap" if self.heatmap_path else None,
        }

    def to_full(self) -> dict:
        d = self.to_summary()
        d.update({
            "content_type": self.content_type, "sha256": self.sha256, "probability_ai_uncalibrated": self.probability_raw,
            "attribution": self.attribution, "cues": self.cues, "regions": self.regions, "provenance": self.provenance,
            "combined": self.combined, "caption": self.caption, "caption_consistency": self.caption_consistency,
            "explanation": self.explanation, "robustness": self.robustness, "timing_ms": self.timing_ms, "model": self.model_info,
            "review_note": self.review_note, "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        })
        return d
