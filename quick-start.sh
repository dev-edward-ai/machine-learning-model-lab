#!/bin/bash
echo "🚀 Starting AutoML Intelligence Platform v2.0..."

# Stop any running containers
echo "Stopping existing containers..."
docker compose down 2>/dev/null || true

# Build and start containers
echo "Building and starting containers..."
docker compose up -d --build

echo "✅ Platform started successfully!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Waiting for services to be ready..."
sleep 5

# Test if services are running
if curl -s http://localhost:8000/ping > /dev/null; then
    echo "✅ Backend is running"
else
    echo "❌ Backend failed to start"
fi

if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Frontend is running"
else
    echo "❌ Frontend failed to start"
fi

echo ""
echo "🎉 AutoML Intelligence Platform is ready!"