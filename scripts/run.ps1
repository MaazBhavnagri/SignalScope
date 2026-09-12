# Start the SignalScope API (serves the built frontend at http://localhost:8000)
param([int]$Port = 8000, [switch]$Dev)
Set-Location (Join-Path $PSScriptRoot "..")
if ($Dev) {
  # API with auto-reload + Vite dev server with HMR (http://localhost:5173)
  Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; uvicorn app.backend.main:app --reload --port 8010"
  Push-Location app/frontend
  npm run dev
  Pop-Location
} else {
  if (-not (Test-Path "app/frontend/dist/index.html")) { Push-Location app/frontend; npm run build; Pop-Location }
  Write-Host "SignalScope -> http://localhost:$Port" -ForegroundColor Green
  uvicorn app.backend.main:app --host 0.0.0.0 --port $Port
}
