# Backend Dockerfile - Python ML API (Render-Ready)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy samples folder for CSV data
COPY samples/ ./samples/

# Copy frontend for serving static files
COPY frontend/ ./frontend/

# Expose port (Render will use PORT env variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/ping || exit 1

# Run the FastAPI server - Render sets PORT dynamically
CMD uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
