# restart.ps1 — kill API process on port 8000, clear pyc cache, start fresh.
# Usage:  .\restart.ps1
#         .\restart.ps1 -Reload          (uvicorn --reload watch mode)

param(
    [switch]$Reload,
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$appDir      = Join-Path $projectRoot "langraph pipeline"
$python      = Join-Path $projectRoot "venv\Scripts\python.exe"

# ── 1. Kill anything on the port ────────────────────────────────
$pids = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue `
        | Select-Object -ExpandProperty OwningProcess -Unique
if ($pids) {
    foreach ($targetPid in $pids) {
        try {
            Stop-Process -Id $targetPid -Force -ErrorAction Stop
            Write-Host "[restart] killed PID $targetPid on :$Port" -ForegroundColor Yellow
        } catch {
            Write-Host "[restart] could not kill PID $targetPid : $_" -ForegroundColor Red
        }
    }
    Start-Sleep -Milliseconds 500
} else {
    Write-Host "[restart] nothing running on :$Port" -ForegroundColor DarkGray
}

# ── 2. Clear __pycache__ ────────────────────────────────────────
Get-ChildItem -Path $appDir -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue `
    | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "[restart] cleared __pycache__" -ForegroundColor DarkGray

# ── 3. Start ────────────────────────────────────────────────────
Push-Location $appDir
try {
    if ($Reload) {
        Write-Host "[restart] starting uvicorn --reload on :$Port ..." -ForegroundColor Green
        & $python -m uvicorn api:app --reload --host 0.0.0.0 --port $Port
    } else {
        Write-Host "[restart] starting api.py on :$Port ..." -ForegroundColor Green
        & $python api.py
    }
} finally {
    Pop-Location
}
