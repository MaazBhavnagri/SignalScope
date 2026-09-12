"""Test fixtures. Tests never require trained weights or network access: a randomly initialised
SignalScopeNet is saved to a temporary checkpoint and the API is pointed at it."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

@pytest.fixture(scope="session")
def tmp_root(tmp_path_factory):
    return tmp_path_factory.mktemp("signalscope")


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_root):
    from model import config as C
    from model.nets import SignalScopeNet
    torch.manual_seed(0)
    net = SignalScopeNet(pretrained=False)
    p = tmp_root / "tiny.pt"
    torch.save({"state_dict": net.state_dict(), "epoch": 0, "val_auc": 0.5, "attr_classes": C.ATTRIBUTION_CLASSES,
                "held_out_generators": C.HELD_OUT_GENERATORS, "config": {"backbone": C.BACKBONE, "img_size": 224}}, p)
    return p


@pytest.fixture(scope="session")
def sample_image() -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    # add smooth structure so cues are not degenerate
    yy, xx = np.mgrid[0:240, 0:320]
    arr[..., 0] = (arr[..., 0] * 0.3 + 120 + 60 * np.sin(xx / 40)).clip(0, 255)
    return Image.fromarray(arr)


@pytest.fixture(scope="session")
def sample_jpeg_bytes(sample_image) -> bytes:
    buf = io.BytesIO()
    ex = Image.Exif()
    ex[0x010F] = "TestCam"; ex[0x0110] = "Model X"
    sample_image.save(buf, "JPEG", quality=90, exif=ex.tobytes())
    return buf.getvalue()


@pytest.fixture(scope="session")
def client(tiny_checkpoint, tmp_root):
    os.environ["SIGNALSCOPE_CHECKPOINT"] = str(tiny_checkpoint)
    os.environ["SIGNALSCOPE_STORAGE"] = str(tmp_root / "storage")
    os.environ["SIGNALSCOPE_DATABASE_URL"] = f"sqlite:///{(tmp_root / 'test.db').as_posix()}"
    os.environ["SIGNALSCOPE_LAZY_MODEL"] = "0"
    from fastapi.testclient import TestClient
    from app.backend.main import app
    with TestClient(app) as c:
        yield c
