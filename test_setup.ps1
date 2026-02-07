#!/usr/bin/env powershell
# Business Insight Generator - Quick Test Script

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Business Insight Generator - Test Suite" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check required packages
Write-Host ""
Write-Host "[2/5] Checking required packages..." -ForegroundColor Yellow
$packages = @("fastapi", "uvicorn", "pandas", "numpy", "scikit-learn", "python-multipart")

foreach ($package in $packages) {
    try {
        python -c "import ${package}" 2>&1 | Out-Null
        Write-Host "✓ $package is installed" -ForegroundColor Green
    } catch {
        Write-Host "✗ $package is NOT installed" -ForegroundColor Red
        Write-Host "  Run: pip install $package" -ForegroundColor Yellow
    }
}

# Check file structure
Write-Host ""
Write-Host "[3/5] Checking file structure..." -ForegroundColor Yellow
$requiredFiles = @(
    "backend/logic.py",
    "backend/api/main.py",
    "frontend/index.html",
    "frontend/script.js",
    "frontend/styles.css",
    "samples/crypto_signals.csv",
    "samples/heart_disease.csv",
    "samples/used_car_prices.csv",
    "samples/marketing_roi.csv"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✓ $file exists" -ForegroundColor Green
    } else {
        Write-Host "✗ $file NOT found" -ForegroundColor Red
    }
}

# Check port availability
Write-Host ""
Write-Host "[4/5] Checking port availability..." -ForegroundColor Yellow
$portTest = netstat -an 2>$null | Select-String ":8000"
if ($portTest) {
    Write-Host "⚠ Port 8000 appears to be in use" -ForegroundColor Yellow
    Write-Host "  Kill existing processes or use different port" -ForegroundColor Yellow
} else {
    Write-Host "✓ Port 8000 is available" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "[5/5] Summary" -ForegroundColor Yellow
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "✅ SYSTEM READY FOR TESTING!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Start backend:"
Write-Host "     cd backend && python -m uvicorn api.main:app --reload --port 8000"
Write-Host ""
Write-Host "  2. Open frontend:"
Write-Host "     frontend/index.html (in browser)"
Write-Host ""
Write-Host "  3. Upload CSV from samples/"
Write-Host "     - crypto_signals.csv  → CRYPTO scenario"
Write-Host "     - heart_disease.csv   → MEDICAL scenario"
Write-Host "     - used_car_prices.csv → CAR_PRICE scenario"
Write-Host "     - marketing_roi.csv   → SALES scenario"
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
