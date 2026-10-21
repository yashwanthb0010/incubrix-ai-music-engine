# Dockerfile for IncuBrix AI Background Music Engine
FROM python:3.11-slim

# Install system dependencies (ffmpeg, libsndfile for audio processing, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and tests
COPY src/ ./src/
COPY tests/ ./tests/
COPY openapi.yaml .
COPY .env.example .env

ENV PYTHONPATH=/app/src \
    PORT=8000 \
    HOST=0.0.0.0 \
    MUSIC_PROVIDER=mock \
    STEM_SEPARATOR=mock \
    STORAGE_TYPE=local

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI application
CMD ["uvicorn", "incubrix.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
