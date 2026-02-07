# Business Insight Generator - Complete Server Startup
# Starts both Backend (FastAPI) and Frontend (Static Server)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Business Insight Generator - Server Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the project root
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $projectRoot "backend"
$frontendDir = Join-Path $projectRoot "frontend"

Write-Host "Project Root: $projectRoot" -ForegroundColor Yellow
Write-Host "Backend Dir: $backendDir" -ForegroundColor Yellow
Write-Host "Frontend Dir: $frontendDir" -ForegroundColor Yellow
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Green
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Python found" -ForegroundColor Green
Write-Host ""

# Check requirements
Write-Host "Checking required packages..." -ForegroundColor Green
$requiredPackages = @("fastapi", "uvicorn", "pandas", "numpy", "scikit-learn")
foreach ($pkg in $requiredPackages) {
    python -c "import $pkg" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $pkg installed" -ForegroundColor Green
    } else {
        Write-Host "❌ $pkg NOT installed - installing now..." -ForegroundColor Yellow
        pip install $pkg -q
    }
}
Write-Host ""

# Display instructions
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 STARTUP INSTRUCTIONS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "TERMINAL 1: Start FastAPI Backend" -ForegroundColor Green
Write-Host "Command:" -ForegroundColor White
Write-Host "cd $backendDir" -ForegroundColor Cyan
Write-Host "python -m uvicorn api.main:app --reload --port 8000" -ForegroundColor Cyan
Write-Host ""

Write-Host "TERMINAL 2: Start Simple HTTP Server for Frontend" -ForegroundColor Green
Write-Host "Command:" -ForegroundColor White
Write-Host "cd $frontendDir" -ForegroundColor Cyan
Write-Host "python -m http.server 3000 --directory ." -ForegroundColor Cyan
Write-Host ""

Write-Host "TERMINAL 3: Open Frontend in Browser" -ForegroundColor Green
Write-Host "Open: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "📝 TESTING" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Drag one of these CSVs into the browser:" -ForegroundColor White
Write-Host "   - samples/crypto_signals.csv" -ForegroundColor Yellow
Write-Host "   - samples/heart_disease.csv" -ForegroundColor Yellow
Write-Host "   - samples/used_car_prices.csv" -ForegroundColor Yellow
Write-Host "   - samples/marketing_roi.csv" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. The UI will load based on detected scenario" -ForegroundColor White
Write-Host "3. Use sliders and inputs to make predictions" -ForegroundColor White
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ SETUP COMPLETE - Ready to start servers!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
