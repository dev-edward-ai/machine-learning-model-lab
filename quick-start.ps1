# AutoML Intelligence Platform v2.0 - Quick Start
Write-Host "🚀 Starting AutoML Intelligence Platform v2.0..." -ForegroundColor Green

# Stop any running containers
Write-Host "Stopping existing containers..." -ForegroundColor Yellow
try {
    docker compose down 2>$null
} catch {
    # Ignore errors if containers aren't running
}

# Build and start containers
Write-Host "Building and starting containers..." -ForegroundColor Yellow
docker compose up -d --build

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Platform started successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Frontend: http://localhost:3000" -ForegroundColor Cyan
    Write-Host "🔧 Backend API: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5

    # Test if services are running
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/ping" -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Backend is running" -ForegroundColor Green
        } else {
            Write-Host "❌ Backend failed to start" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ Backend failed to start" -ForegroundColor Red
    }

    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3000" -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Frontend is running" -ForegroundColor Green
        } else {
            Write-Host "❌ Frontend failed to start" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ Frontend failed to start" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "🎉 AutoML Intelligence Platform is ready!" -ForegroundColor Magenta
    Write-Host "🔗 Open http://localhost:3000 to get started!" -ForegroundColor Cyan
} else {
    Write-Host "❌ Failed to start platform. Check Docker is running." -ForegroundColor Red
}