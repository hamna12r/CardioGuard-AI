# =========================================================================
# CardioGuard AI - Production Multi-Stage Dockerfile
# =========================================================================

# Stage 1: Build & Dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Lean Production Runtime
FROM python:3.11-slim AS runtime

# Create non-root system user for security
RUN useradd -m -u 1001 cardioguard

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/cardioguard/.local

# Copy application source and model artifacts
COPY --chown=cardioguard:cardioguard app/ ./app/
COPY --chown=cardioguard:cardioguard model/ ./model/
COPY --chown=cardioguard:cardioguard data/ ./data/

# Set environment variables
ENV PATH=/home/cardioguard/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

USER cardioguard

# Expose FastAPI service port
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start FastAPI production server with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
