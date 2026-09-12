#!/usr/bin/env bash
# SignalScope one-shot setup (Linux/macOS). Run from the repo root:  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Python dependencies =="
python3 -m pip install --upgrade pip
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected -> CUDA 12.1 PyTorch build"
  python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
  echo "No NVIDIA GPU -> CPU PyTorch build"
  python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
python3 -m pip install -r requirements.txt

echo "== Frontend =="
(cd app/frontend && npm install && npm run build)

echo "== Weights =="
[ -f weights/signalscope_best.pt ] || python3 scripts/get_weights.py || echo "No weights release configured - train locally (README section 4)."
echo "Setup complete. Start with: bash scripts/run.sh"
