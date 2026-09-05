#!/usr/bin/env pwsh
# GeoSemantic Platform -- Windows Startup Script
# Run from project root: .\scripts\start.ps1

param(
    [switch]$SetupOnly,
    [switch]$BackendOnly,
    [switch]$FrontendOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  GeoSemantic Satellite Intelligence  " -ForegroundColor Cyan
Write-Host "  Platform - SIH 2026                " -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# -- Verify prerequisites --------------------------------------
function Test-Command($cmd) {
    try { Get-Command $cmd -ErrorAction Stop | Out-Null; return $true }
    catch { return $false }
}

if (-not (Test-Command "python")) {
    Write-Host "[ERROR] Python not found. Install Python 3.11+" -ForegroundColor Red
    exit 1
}
if (-not (Test-Command "node")) {
    Write-Host "[ERROR] Node.js not found. Install Node.js 20+" -ForegroundColor Red
    exit 1
}

# -- Backend setup ---------------------------------------------
$BackendDir = "$Root\backend"
$VenvDir = "$BackendDir\.venv"

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv $VenvDir
}

$PythonExe = "$VenvDir\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "[OK] Virtual environment ready" -ForegroundColor Green

# Check if deps installed
$uvicorn = "$VenvDir\Scripts\uvicorn.exe"
if (-not (Test-Path $uvicorn)) {
    Write-Host "Installing backend dependencies..." -ForegroundColor Yellow
    & $PythonExe -m pip install -r "$BackendDir\requirements.txt" --quiet
    Write-Host "[OK] Backend dependencies installed" -ForegroundColor Green
}

# Check .env
$EnvFile = "$BackendDir\.env"
if (-not (Test-Path $EnvFile)) {
    Copy-Item "$BackendDir\.env.example" $EnvFile
    Write-Host ""
    Write-Host "[WARN] Created backend\.env from template." -ForegroundColor Yellow
    Write-Host "  Edit it to set your PostgreSQL password, then re-run this script." -ForegroundColor Yellow
    Write-Host ""
    exit 0
}

# Database setup
if ($SetupOnly -or -not $FrontendOnly) {
    Write-Host "Setting up database..." -ForegroundColor Yellow
    Set-Location $BackendDir
    & $PythonExe "$Root\scripts\setup_db.py" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Database setup failed. Check PostgreSQL is running and credentials in .env" -ForegroundColor Yellow
    } else {
        Write-Host "[OK] Database ready" -ForegroundColor Green
    }
}

if ($SetupOnly) {
    Write-Host ""
    Write-Host "[OK] Setup complete. Run without -SetupOnly to start servers." -ForegroundColor Green
    exit 0
}

# -- Start servers ---------------------------------------------
Set-Location $Root

if (-not $FrontendOnly) {
    Write-Host ""
    Write-Host "Starting FastAPI backend on http://localhost:8000 ..." -ForegroundColor Green
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "cd '$BackendDir'; .venv\Scripts\uvicorn.exe app.main:app --reload --port 8000"
    ) -WindowStyle Normal
    Start-Sleep -Seconds 2
}

if (-not $BackendOnly) {
    Write-Host "Starting Next.js frontend on http://localhost:3000 ..." -ForegroundColor Green
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "cd '$Root\frontend'; npm run dev"
    ) -WindowStyle Normal
}

Write-Host ""
Write-Host "[OK] Platform starting up." -ForegroundColor Green
Write-Host "  Backend:  http://localhost:8000/api/docs" -ForegroundColor Cyan
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
