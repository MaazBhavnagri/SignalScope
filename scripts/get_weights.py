"""Download the released SignalScope weights + calibration into weights/ (used by setup scripts).

The URLs point at the GitHub release attached to this repository (see weights/README.md). If the download
fails (offline judging), train locally instead:  python -m model.data.download && python -m model.data.prepare
&& python -m model.train && python -m model.calibrate && python -m model.evaluate
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
MANIFEST = WEIGHTS / "release.json"  # {"base_url": "...", "files": ["signalscope_best.pt", "calibration.json", "cue_reference.json"]}


def main() -> int:
    if not MANIFEST.exists():
        print("weights/release.json not found - no release configured yet. Train locally (see README).")
        return 1
    meta = json.loads(MANIFEST.read_text())
    base = meta["base_url"].rstrip("/")
    ok = True
    for name in meta["files"]:
        dest = WEIGHTS / name
        if dest.exists():
            print(f"[skip] {name} exists")
            continue
        url = f"{base}/{name}"
        print(f"[get ] {url}")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[ ok ] {name} ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {name}: {e}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
