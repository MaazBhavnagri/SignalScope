"""Small shared helpers."""
from __future__ import annotations

import contextlib

import torch


def autocast_ctx(device: torch.device | str):
    """fp16 autocast on CUDA, no-op elsewhere (CPU autocast in older torch only supports bf16)."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()
