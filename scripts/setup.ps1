# SignalScope one-shot setup (Windows PowerShell). Run from the repo root:  .\scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "== Python dependencies ==" -ForegroundColor Cyan
python -m pip install --upgrade pip
$hasCuda = $false
try { $null = & nvidia-smi 2>$null; $hasCuda = $LASTEXITCODE -eq 0 } catch {}
if ($hasCuda) {
  Write-Host "NVIDIA GPU detected -> installing CUDA 12.1 PyTorch build"
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
} else {
  Write-Host "No NVIDIA GPU detected -> installing CPU PyTorch build"
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}
python -m pip install -r requirements.txt

Write-Host "== Frontend dependencies + build ==" -ForegroundColor Cyan
Push-Location app/frontend
npm install
npm run build
Pop-Location

Write-Host "== Model weights ==" -ForegroundColor Cyan
if (-not (Test-Path "weights/signalscope_best.pt")) {
  python scripts/get_weights.py
}
Write-Host "Setup complete. Start the app with .\scripts\run.ps1" -ForegroundColor Green
