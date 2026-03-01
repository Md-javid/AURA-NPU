# ══════════════════════════════════════════════════════════════════════════════
#  Aura-NPU  Dockerfile
#  CPU-safe container (Tier 2: Ollama + Tier 3: PIL)
#
#  NPU acceleration (VitisAI EP) requires native Windows / bare-metal host.
#  This container is the reproducibility / hackathon-demo / CI target.
#
#  Build:
#    docker build -f docker/Dockerfile -t aura-npu:latest .
#
#  Run (with Ollama on host):
#    docker compose -f docker/docker-compose.yml up
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# ── Labels ─────────────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="Aura-NPU" \
      org.opencontainers.image.description="Neuro-inclusive offline AI cognitive assistant" \
      org.opencontainers.image.version="4.1.0" \
      org.opencontainers.image.licenses="Apache-2.0"

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python environment ─────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Install Python dependencies ────────────────────────────────────────────
#  Copy requirements first for layer-cache efficiency.
#  NOTE: onnxruntime-vitisai is Windows-only — use plain onnxruntime in container.
COPY requirements.docker.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ────────────────────────────────────────────────
#  Only the app/ package — models are volume-mounted at runtime.
COPY app/ ./app/

# ── Runtime configuration ──────────────────────────────────────────────────
#  All values can be overridden via docker-compose environment.
ENV AURA_OFFLINE=true \
    AURA_FORCE_CPU=true \
    AURA_ECO_MODE=false \
    AURA_LOG_LEVEL=INFO \
    AURA_HOST=0.0.0.0 \
    AURA_PORT=8080 \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_MODEL=llava

# ── Expose NiceGUI port ────────────────────────────────────────────────────
EXPOSE 8080

# ── Health check ──────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080')" \
    || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────
CMD ["python", "-m", "app.main"]
