# Aura-NPU

**Offline neuro-inclusive cognitive assistant powered by the AMD Ryzen AI 300 NPU**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![AMD Ryzen AI](https://img.shields.io/badge/AMD-Ryzen%20AI%20300-ED1C24)](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html)
[![Offline](https://img.shields.io/badge/inference-100%25%20offline-brightgreen.svg)](#offline-guarantee)

---

## Problem Statement

Over **320 million people** in India live with learning differences — dyslexia, ADHD, dyscalculia — yet mainstream AI assistants require persistent internet connectivity, process data on remote servers, and respond exclusively in English.

This creates three compounding barriers for students in Indian schools:

| Barrier | Impact |
|---|---|
| Connectivity dependency | 35% of Indian school students lack reliable broadband (ASER 2024) |
| Language exclusion | 22 official languages; most AI tools support English only |
| Privacy exposure | Student data sent to cloud APIs without informed consent |

Aura-NPU addresses all three simultaneously.

---

## What is Aura-NPU

Aura-NPU is a desktop overlay application that runs **multimodal AI inference entirely on-device** — no internet required, no API keys, no cloud endpoints.

A student points Aura at any screen content — a homework problem, a textbook diagram, an exam paper — and receives an accessible, structured AI analysis in their preferred Indian language.

**Key properties:**

- Runs on the AMD Ryzen AI 300 Series XDNA 2 NPU at ~100 ms per inference
- Falls back automatically to Ollama (localhost) or PIL when NPU is absent
- Supports all 22 constitutionally recognised Indian languages
- Generates a tamper-evident academic integrity log (RPwD Act 2016 / DPDP Act 2023 compliant)
- Ships zero model weights — download instructions in `models/README.md`

---

## Architecture

### Three-Tier Inference Pipeline

```
User Screen / Camera
        │
        ▼
┌───────────────────────────────────────────┐
│              Aura-NPU Overlay             │
│              (NiceGUI, port 8765)         │
└───────────────┬───────────────────────────┘
                │
       ┌────────▼────────┐
       │  Tier 1: NPU    │  VitisAI EP · INT8 ONNX
       │  ~100 ms · 12 W │  AMD Ryzen AI 300 XDNA 2
       └────────┬────────┘
                │ fallback — NPU not detected
       ┌────────▼────────┐
       │  Tier 2: Ollama │  llava vision model
       │  ~800 ms · CPU  │  localhost:11434 only
       └────────┬────────┘
                │ fallback — Ollama not running
       ┌────────▼────────┐
       │  Tier 3: PIL    │  Offline pixel analysis
       │  ~50 ms · zero  │  brightness · edges · colour
       └─────────────────┘
```

All three tiers are entirely local. No data leaves the device.

### Repository Structure

```
aura-npu/
├── app/
│   ├── main.py              # NiceGUI overlay, UI, scan pipeline
│   ├── config.py            # All configuration (env-driven, no hardcoding)
│   ├── npu_engine.py        # VitisAI EP session management
│   ├── telemetry.py         # Async NPU/CPU metrics loop
│   ├── multimodal_logic.py  # ASR (IndicConformer) + TTS (Svara) pipeline
│   ├── integrity_tracker.py # Tamper-evident JSONL log (SHA-256 chain)
│   └── utils/
│       ├── screen_capture.py
│       └── language_config.py
├── models/                  # Empty — see models/README.md
├── scripts/
│   ├── download_models.py   # Unified model setup and verification
│   ├── quantize.py          # AMD Quark INT8 quantization pipeline
│   └── benchmark.py         # NPU vs CPU latency benchmark
├── tests/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/workflows/ci.yml
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
├── requirements.docker.txt
└── README.md
```

---

## Hardware Benchmarks

Measured on AMD Ryzen AI 9 HX 370 (Strix Point, 50 TOPS NPU), Windows 11 23H2.

| Metric | NPU (Tier 1) | CPU Baseline | Delta |
|---|---|---|---|
| Inference latency P50 | ~100 ms | ~1 400 ms | ~14× faster |
| Inference latency P95 | ~130 ms | ~2 000 ms | ~15× faster |
| Sustained power draw | ~12 W | ~45 W | ~73% less |
| NPU utilisation | 60–80% | — | visible in Task Manager |

> These figures require Ryzen AI SW 1.7 driver and INT8-quantized model weights.
> Actual results vary with thermal state and concurrent load.

---

## Features

| Feature | Description |
|---|---|
| Multimodal Vision | Analyse screen content or uploaded images with llava or NPU VLM |
| 22 Indian Languages | Full constitutionally recognised language set |
| ADHD Focus Mode | 5-step chunked output + on-screen Pomodoro timer |
| Eco Mode | 50% image downscale, 200-token cap — ~40% battery saving |
| Vision Trace | Live pipeline stage display with per-step latency |
| Dual-Pane Output | English analysis + native-script translation side-by-side |
| Integrity Dashboard | SHA-256-chained JSONL log of every AI interaction |
| Sovereign Mode | Verifies no outbound network before inference |
| Hardware Telemetry | Live NPU%, CPU%, power estimate, inference counter |
| Camera Input | Direct camera capture in addition to screen region |

---

## Quick Start — Ollama (No NPU Required)

Full AI vision in under five minutes on any Windows or Linux machine.

```powershell
# 1. Install Ollama
winget install Ollama.Ollama

# 2. Pull the llava vision model (~4.7 GB, one-time)
ollama pull llava

# 3. Clone and install
git clone https://github.com/Md-javid/AURA-NPU.git
cd AURA-NPU
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 4. Run
python -m app.main
# Open http://127.0.0.1:8765
```

The app pre-warms llava at startup (~60 s on first load). The Hardware Telemetry panel shows
**Ollama/llava** once ready.

---

## Full NPU Installation (AMD Ryzen AI 300)

Activates Tier 1 inference — ~100 ms, ~12 W.

### Prerequisites

| Requirement | Value |
|---|---|
| Hardware | AMD Ryzen AI 300 Series (Strix Point) or Ryzen AI Max 300 (Strix Halo) |
| OS | Windows 11 23H2+ |
| NPU driver | `10.106.5.x` (Ryzen AI SW 1.7) |
| Python | **3.11 exactly** (VitisAI EP constraint) |
| RAM | 32 GB minimum |
| Storage | 25 GB free (models ~8 GB post-quantization) |

### Installation

```powershell
# 1. Install Ryzen AI SW 1.7
#    https://ryzenai.docs.amd.com/en/latest/inst.html

# 2. Activate the conda env created by the installer
conda activate ryzen-ai-1.7

# 3. Clone and install
git clone https://github.com/Md-javid/AURA-NPU.git
cd AURA-NPU
pip install -r requirements.txt

# 4. Download and quantize models
python scripts/download_models.py --weights
python scripts/quantize.py --model all

# 5. Run
python -m app.main
```

### Verifying NPU Usage

1. Open **Windows Task Manager → Performance → NPU**
2. Run a scan in Aura-NPU
3. Observe 60–80% NPU utilisation spike during inference
4. The Hardware Telemetry panel shows `⚡ XDNA² Active`

---

## Docker (CPU / Ollama Tier)

NPU is not available in Docker (VitisAI EP requires bare-metal Windows).

```bash
# Requires Ollama running on the host: ollama pull llava

docker compose -f docker/docker-compose.yml up --build -d
# Open http://localhost:8080

docker compose -f docker/docker-compose.yml down
```

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llava` | Model tag |
| `AURA_OFFLINE` | `true` | Block non-localhost HTTP |
| `AURA_FORCE_CPU` | `true` | Disable VitisAI EP |
| `AURA_PORT` | `8080` | Container bind port |

---

## ADHD Focus Mode

When enabled, all AI output is restructured into **five numbered steps** (≤ 40 words each)
with colour-coded headings. An on-screen Pomodoro timer (25 min work / 5 min break) runs in
the overlay corner.

Based on cognitive load theory (Sweller, 1988) and ADHD working-memory research (Barkley,
2012). Reduces re-reading, improves task completion, and avoids the overwhelm caused by
long-form prose.

---

## 22 Supported Languages

| Script | Languages |
|---|---|
| Devanagari | Hindi, Marathi, Sanskrit, Nepali, Bodo, Dogri, Maithili, Sindhi |
| Dravidian | Tamil, Telugu, Kannada, Malayalam |
| Eastern | Bengali, Assamese, Meitei, Odia |
| Perso-Arabic | Urdu, Kashmiri |
| Other | Gujarati, Punjabi, Konkani, Santali |

ASR uses AI4Bharat IndicConformer; TTS uses Svara. Both run offline via VitisAI EP on NPU
hardware, or fall back to CPU inference.

---

## Academic Integrity Dashboard

Every inference generates a signed JSONL entry:

```json
{
  "timestamp": "2026-03-01T06:14:12Z",
  "session_id": "AURA-47291",
  "prompt_hash": "sha256:3a7f...",
  "provider": "Ollama/llava",
  "language": "hi",
  "latency_ms": 812,
  "chain_hash": "sha256:9c1e..."
}
```

`chain_hash` links each entry to the previous, making the log tamper-evident. Supports:

- **RPwD Act 2016** — reasonable accommodation documentation for examinations
- **DPDP Act 2023** — all data on-device, zero cloud upload
- **NEP 2020** — formative AI assistance, not answer generation

---

## Offline Guarantee

Sovereign Mode (`🔒` toggle) actively verifies network isolation:

1. Attempts an outbound connection to `1.1.1.1:53` and `8.8.8.8:53`
2. If either succeeds — device has internet access — a warning is shown
3. If both fail — **"Firewall Safe — No Cloud"** is confirmed in the UI

All three inference tiers (NPU, Ollama, PIL) operate exclusively on `localhost`.
Zero telemetry is transmitted anywhere.

---

## Configuration

All settings are environment variables. Defaults are in `app/config.py`.

```bash
$env:OLLAMA_MODEL  = "llava:13b"   # use a larger model
$env:AURA_PORT     = "9000"        # change bind port
$env:AURA_LOG_LEVEL = "DEBUG"      # verbose logging
python -m app.main
```

---

## Limitations

| Limitation | Detail |
|---|---|
| NPU requires Windows | VitisAI EP is Windows-only; Linux uses Ollama/CPU |
| Python 3.11 for NPU | VitisAI EP does not support Python 3.12/3.13 |
| llava cold-start | First inference after `ollama serve` loads ~4.7 GB — allow 60–90 s |
| INT8 quality trade-off | Quantization reduces output quality ~5–8% vs FP16 on standard benchmarks |
| Single session | NiceGUI does not isolate concurrent browser tabs |

---

## Roadmap

- [ ] Streaming token output (Ollama `/api/generate` stream mode)
- [ ] Live microphone ASR via IndicConformer
- [ ] PDF drag-and-drop with ADHD scaffolding per page
- [ ] Multi-user session isolation
- [ ] Hindi voice commands

---

## Development

```bash
make run          # start the overlay app
make test         # pytest (skips npu and slow markers)
make lint         # ruff check + mypy
make benchmark    # PIL vs Ollama latency table
make docker-build # build container image
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contributor guide.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Built for the AMD Slingshot 2026 Hackathon.
