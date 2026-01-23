#!/bin/bash
# Setup Script for Linux/Mac
# AutoML Platform Quick Start

echo "========================================"
echo "   AutoML Platform Setup"
echo "========================================"
echo ""

# Check if Docker is installed
echo "Checking Docker installation..."
if command -v docker &> /dev/null; then
    echo "✅ Docker found: $(docker --version)"
else
    echo "❌ Docker not found! Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
echo "Checking Docker Compose..."
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose found: $(docker-compose --version)"
else
    echo "❌ Docker Compose not found! Please install it first."
    exit 1
fi

# Check if Docker is running
echo "Checking if Docker is running..."
if docker ps &> /dev/null; then
    echo "✅ Docker is running"
else
    echo "❌ Docker is not running! Please start Docker."
    exit 1
fi

echo ""
echo "Building Docker containers..."
docker-compose build

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "Starting services..."
docker-compose up -d

if [ $? -eq 0 ]; then
    echo "✅ Services started successfully!"
else
    echo "❌ Failed to start services!"
    exit 1
fi

echo ""
echo "Waiting for services to be ready..."
sleep 5

echo ""
echo "========================================"
echo "   Setup Complete! 🎉"
echo "========================================"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "To stop the services, run: docker-compose down"
echo "To view logs, run: docker-compose logs -f"
echo ""

# Try to open browser (works on most systems)
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:3000
elif command -v open &> /dev/null; then
    open http://localhost:3000
fi
