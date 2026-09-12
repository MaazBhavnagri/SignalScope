"""SQLite persistence (SQLAlchemy 2.x). DB + file storage live under app/backend/storage/."""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

BACKEND_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(os.environ.get("SIGNALSCOPE_STORAGE", BACKEND_DIR / "storage"))
ORIGINALS_DIR = STORAGE_DIR / "originals"
HEATMAPS_DIR = STORAGE_DIR / "heatmaps"
DB_PATH = STORAGE_DIR / "signalscope.db"
for d in (ORIGINALS_DIR, HEATMAPS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.environ.get("SIGNALSCOPE_DATABASE_URL", f"sqlite:///{DB_PATH.as_posix()}")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from app.backend import models  # noqa: F401 - register tables
    Base.metadata.create_all(engine)
    if DATABASE_URL.startswith("sqlite"):
        with engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
