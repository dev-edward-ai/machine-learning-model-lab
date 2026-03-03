# InsightAI - Single-container Dockerfile
# Serves FastAPI backend + static frontend on one port

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY backend/  ./backend/
COPY frontend/ ./frontend/
COPY samples/  ./samples/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://localhost:'+os.environ.get('PORT','8000')+'/ping')" || exit 1

# Run from /app/backend so relative paths ../frontend and ../samples resolve correctly
CMD sh -c "cd /app/backend && uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"
