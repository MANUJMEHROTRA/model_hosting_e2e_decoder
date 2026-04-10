# ── Stage 1: dependency builder ───────────────────────────────────────────
# Using multi-stage build keeps the final image lean.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
# Install into a prefix we can copy later
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security (Kubernetes best practice)
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY api/ ./api/
# Models are mounted as a volume or loaded from an object store.
# We create the mount point here.
RUN mkdir -p /app/models && chown -R appuser:appuser /app

USER appuser

# Environment variables — override at runtime or via Kubernetes ConfigMap/Secret
ENV MODEL_PATH=/app/models/lora_merged \
    PORT=8080 \
    WORKERS=1 \
    LOG_LEVEL=info

# Expose the port
EXPOSE 8080

# Health check — Docker will mark container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Start uvicorn (ASGI server for FastAPI)
CMD uvicorn api.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers ${WORKERS} \
    --log-level ${LOG_LEVEL}
