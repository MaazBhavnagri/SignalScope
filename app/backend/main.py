"""SignalScope API - FastAPI application entry point.

Run (dev):   uvicorn app.backend.main:app --reload --port 8000
Run (prod):  uvicorn app.backend.main:app --host 0.0.0.0 --port 8000   (serves the built frontend from app/frontend/dist)
"""
from __future__ import annotations

import logging
import os
import time

# Plain-HTTP downloads for optional Hugging Face weights (the xet transport stalled on some networks).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.backend.db import init_db
from app.backend.routes import router
from app.backend.services import MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("signalscope")

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if os.environ.get("SIGNALSCOPE_LAZY_MODEL", "0") != "1":
        MODEL.load()  # warm start; failures are reported via /api/health rather than crashing
    yield


app = FastAPI(title="SignalScope API", version="1.0.0", description="Real vs AI-generated image forensics with faithful explanations.", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("SIGNALSCOPE_CORS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def timing(request: Request, call_next):
    t = time.perf_counter()
    resp = await call_next(request)
    resp.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t) * 1000:.1f}"
    return resp


@app.exception_handler(StarletteHTTPException)
async def http_exc(request: Request, exc: StarletteHTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"error": "http_error", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(RequestValidationError)
async def validation_exc(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": {"error": "validation_error", "message": "Invalid request.", "errors": exc.errors()}})


@app.exception_handler(Exception)
async def unhandled_exc(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": {"error": "internal_error", "message": str(exc)[:300]}})


app.include_router(router, prefix="/api")

# ---- static frontend (production) ----
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": "SignalScope API", "docs": "/docs", "health": "/api/health", "note": "frontend not built - run `npm run build` in app/frontend or use the Vite dev server"}
