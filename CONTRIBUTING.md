# Contributing to Aura-NPU

Thank you for your interest in contributing to Aura-NPU.

## Development Setup

```bash
git clone https://github.com/Md-javid/AURA-NPU.git
cd AURA-NPU
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m app.main
```

## Code Style

- Python 3.11+, type hints required on all public functions
- Line length: 100 characters (enforced by `ruff`)
- Formatter: `ruff format`
- Linter: `ruff check`
- Type checker: `mypy app/`

```bash
make lint          # ruff check + mypy
make lint-fix      # auto-fix ruff issues
```

## Offline-First Constraint

**No PR will be merged that introduces an outbound network call at inference time.**

All inference must follow the priority chain:
1. VitisAI EP / AMD NPU
2. Ollama `localhost` only
3. PIL offline fallback

## Testing

```bash
make test                        # run all tests
pytest -m "not npu and not slow" # skip hardware-dependent tests
```

Tests requiring AMD NPU hardware are marked `@pytest.mark.npu` and are skipped in CI.

## Pull Request Checklist

- [ ] `make lint` passes with zero errors
- [ ] `python -c "import ast; ast.parse(open('app/main.py', encoding='utf-8').read())"` — no syntax errors
- [ ] New functions have docstrings
- [ ] No new outbound HTTP calls outside `localhost`
- [ ] No model weights committed (`*.onnx`, `*.bin`, `*.safetensors`)
- [ ] No `.env` files committed
- [ ] No absolute Windows paths hardcoded (use `app/config.py` constants)

## Reporting Issues

Open a GitHub issue with:
- OS and Python version
- AMD hardware model (if applicable)
- Full traceback
- Steps to reproduce

## License

By contributing, you agree your changes will be licensed under the [Apache 2.0 License](LICENSE).
