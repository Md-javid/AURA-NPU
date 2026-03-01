# ══════════════════════════════════════════════════════════════════════════════
#  Aura-NPU  Makefile
#  Reproducible developer workflow — AMD Slingshot 2026
# ══════════════════════════════════════════════════════════════════════════════
#  Usage:
#    make              → show help
#    make run          → start app natively (Ollama required)
#    make docker-build → build Docker image (CPU-safe)
#    make docker-run   → start with docker compose
#    make test         → run test suite
#    make lint         → run ruff linter
#    make benchmark    → run hardware benchmark
#    make models       → download model configs / weights
#    make clean        → remove caches
# ══════════════════════════════════════════════════════════════════════════════

.PHONY: help run docker-build docker-run docker-stop test lint typecheck benchmark models clean

# ── Variables ──────────────────────────────────────────────────────────────
PYTHON      ?= python
COMPOSE     := docker compose -f docker/docker-compose.yml
IMAGE_TAG   ?= aura-npu:latest
PORT        ?= 8765

# ── Help ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Aura-NPU — AMD Slingshot 2026"
	@echo "  ==============================="
	@echo ""
	@echo "  make run            Start app natively (requires Ollama on localhost)"
	@echo "  make docker-build   Build CPU-safe Docker image"
	@echo "  make docker-run     Start via docker compose (port 8080)"
	@echo "  make docker-stop    Stop docker compose stack"
	@echo "  make test           Run pytest suite"
	@echo "  make lint           Run ruff linter"
	@echo "  make typecheck      Run mypy"
	@echo "  make benchmark      Run offline hardware benchmark"
	@echo "  make models         Download model configs (no large weights)"
	@echo "  make clean          Remove __pycache__, .ruff_cache, logs"
	@echo ""

# ── Native run ─────────────────────────────────────────────────────────────
run:
	@echo ">> Starting Aura-NPU on http://127.0.0.1:$(PORT) ..."
	$(PYTHON) -m app.main

# ── Docker ─────────────────────────────────────────────────────────────────
docker-build:
	@echo ">> Building Docker image $(IMAGE_TAG) ..."
	docker build -f docker/Dockerfile -t $(IMAGE_TAG) .

docker-run:
	@echo ">> Starting Aura-NPU via Docker Compose (port 8080) ..."
	@echo "   Make sure Ollama is running: ollama serve && ollama pull llava"
	$(COMPOSE) up --build -d
	@echo ">> Open http://localhost:8080"

docker-stop:
	$(COMPOSE) down

docker-logs:
	$(COMPOSE) logs -f

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	@echo ">> Running test suite ..."
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

# ── Code quality ───────────────────────────────────────────────────────────
lint:
	@echo ">> Running ruff linter ..."
	$(PYTHON) -m ruff check app/ scripts/ tests/

lint-fix:
	$(PYTHON) -m ruff check --fix app/ scripts/ tests/

typecheck:
	$(PYTHON) -m mypy app/ --ignore-missing-imports

# ── Benchmark ──────────────────────────────────────────────────────────────
benchmark:
	@echo ">> Running offline hardware benchmark ..."
	$(PYTHON) scripts/benchmark.py

# ── Model setup ────────────────────────────────────────────────────────────
models:
	@echo ">> Downloading model configs ..."
	$(PYTHON) scripts/download_models.py

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:
	@echo ">> Cleaning build artifacts ..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache .pytest_cache
	rm -f app_out.log app_err.log logs/*.log
	@echo ">> Done."

clean-models:
	@echo ">> WARNING: This will delete downloaded model weights!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	find models/ -name "*.onnx" -delete
	find models/ -name "*.bin" -delete
	find models/ -name "*.safetensors" -delete
	@echo ">> Model weights deleted. Run 'make models' to re-download."
