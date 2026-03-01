"""
config.py — Aura-NPU centralised runtime configuration
=======================================================
All tuneable knobs live here.  Values are read once at import time from
environment variables with sensible defaults so the app runs out-of-the-box
with zero configuration required.

No API keys.  No cloud endpoints.  100% offline.

Environment variables (all optional):

  AURA_OFFLINE           bool   Refuse any non-localhost HTTP (default: true)
  AURA_ECO_MODE          bool   50 % image downscale + 200 max tokens (default: false)
  AURA_FORCE_CPU         bool   Disable NPU; use CPU-only ONNX session (default: false)
  AURA_LOG_LEVEL         str    Python logging level name (default: INFO)
  AURA_HOST              str    NiceGUI bind host (default: 127.0.0.1)
  AURA_PORT              int    NiceGUI bind port (default: 8765)
  OLLAMA_HOST            str    Ollama base URL (default: http://localhost:11434)
  OLLAMA_MODEL           str    Model tag to use via Ollama (default: llava)
  AURA_MAX_TOKENS        int    Default max tokens for inference (default: 600)
  AURA_ECO_MAX_TOKENS    int    Max tokens when eco mode is on (default: 200)
  AURA_TELEMETRY_INTERVAL float Telemetry poll interval seconds (default: 1.0)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


# ─────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = ROOT_DIR / "models"
LOGS_DIR: Path = ROOT_DIR / "logs"

LOGS_DIR.mkdir(exist_ok=True)

# Model sub-paths
VLM_MODEL_DIR: Path = MODELS_DIR / "gemma3_4b_vlm"
VLM_MODEL_PATH: Path = VLM_MODEL_DIR / "model_quantized.onnx"
VLM_CONFIG_PATH: Path = VLM_MODEL_DIR / "config.json"

ASR_MODEL_DIR: Path = MODELS_DIR / "indic_conformer"
ASR_MODEL_PATH: Path = ASR_MODEL_DIR / "model_quantized.onnx"

TTS_MODEL_DIR: Path = MODELS_DIR / "svara_tts"
TTS_MODEL_PATH: Path = TTS_MODEL_DIR / "model_quantized.onnx"

NPU_VAIP_CONFIG: Path = MODELS_DIR / "vaip_config.json"

# ─────────────────────────────────────────────────────────────
#  Runtime flags
# ─────────────────────────────────────────────────────────────

OFFLINE: bool = _bool("AURA_OFFLINE", True)
ECO_MODE: bool = _bool("AURA_ECO_MODE", False)
FORCE_CPU: bool = _bool("AURA_FORCE_CPU", False)
LOG_LEVEL: str = _str("AURA_LOG_LEVEL", "INFO").upper()

# ─────────────────────────────────────────────────────────────
#  Server
# ─────────────────────────────────────────────────────────────

HOST: str = _str("AURA_HOST", "127.0.0.1")
PORT: int = _int("AURA_PORT", 8765)

# ─────────────────────────────────────────────────────────────
#  Ollama
# ─────────────────────────────────────────────────────────────

OLLAMA_HOST: str = _str("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = _str("OLLAMA_MODEL", "llava")

# Guard: OFFLINE mode must only allow localhost Ollama
if OFFLINE and OLLAMA_HOST and not OLLAMA_HOST.startswith("http://localhost"):
    raise RuntimeError(
        f"AURA_OFFLINE=true but OLLAMA_HOST={OLLAMA_HOST!r} is not localhost. "
        "Set OLLAMA_HOST=http://localhost:11434 or AURA_OFFLINE=false."
    )

# ─────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────

MAX_TOKENS: int = _int("AURA_MAX_TOKENS", 600)
ECO_MAX_TOKENS: int = _int("AURA_ECO_MAX_TOKENS", 200)
ECO_IMAGE_SCALE: float = _float("AURA_ECO_IMAGE_SCALE", 0.5)
OLLAMA_TIMEOUT_S: float = _float("AURA_OLLAMA_TIMEOUT", 300.0)  # llava cold-start ~2-3 min

# ─────────────────────────────────────────────────────────────
#  Hardware / VitisAI
# ─────────────────────────────────────────────────────────────

XCLBIN_PATH: Path = Path(
    _str(
        "AURA_XCLBIN_PATH",
        r"C:\Windows\System32\AMD\xclbin\AMD_AIE2P_Nx4_Overlay.xclbin",
    )
)
NPU_CACHE_DIR: Path = ROOT_DIR / ".npu_cache"
NPU_CACHE_DIR.mkdir(exist_ok=True)

ORT_EP_PRIORITY: list[str] = (
    ["CPUExecutionProvider"]
    if FORCE_CPU
    else ["VitisAIExecutionProvider", "CPUExecutionProvider"]
)

# ─────────────────────────────────────────────────────────────
#  Telemetry
# ─────────────────────────────────────────────────────────────

TELEMETRY_INTERVAL: float = _float("AURA_TELEMETRY_INTERVAL", 1.0)

# ─────────────────────────────────────────────────────────────
#  Logging bootstrap
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────
#  Convenience summary for startup
# ─────────────────────────────────────────────────────────────

def log_config() -> None:
    """Emit a one-shot configuration summary at startup."""
    log = logging.getLogger("aura.config")
    log.info("Aura-NPU configuration")
    log.info("  offline    : %s", OFFLINE)
    log.info("  eco_mode   : %s", ECO_MODE)
    log.info("  force_cpu  : %s", FORCE_CPU)
    log.info("  log_level  : %s", LOG_LEVEL)
    log.info("  host:port  : %s:%s", HOST, PORT)
    log.info("  ollama     : %s  model=%s", OLLAMA_HOST, OLLAMA_MODEL)
    log.info("  max_tokens : %s (eco: %s)", MAX_TOKENS, ECO_MAX_TOKENS)
    log.info("  models_dir : %s", MODELS_DIR)
    log.info("  vlm_ready  : %s", VLM_MODEL_PATH.exists())
    log.info("  asr_ready  : %s", ASR_MODEL_PATH.exists())
    log.info("  tts_ready  : %s", TTS_MODEL_PATH.exists())


@dataclass
class RuntimeConfig:
    """Structured snapshot of current config — useful for passing into subsystems."""
    offline: bool = field(default_factory=lambda: OFFLINE)
    eco_mode: bool = field(default_factory=lambda: ECO_MODE)
    force_cpu: bool = field(default_factory=lambda: FORCE_CPU)
    host: str = field(default_factory=lambda: HOST)
    port: int = field(default_factory=lambda: PORT)
    ollama_host: str = field(default_factory=lambda: OLLAMA_HOST)
    ollama_model: str = field(default_factory=lambda: OLLAMA_MODEL)
    max_tokens: int = field(default_factory=lambda: MAX_TOKENS)
    eco_max_tokens: int = field(default_factory=lambda: ECO_MAX_TOKENS)
    telemetry_interval: float = field(default_factory=lambda: TELEMETRY_INTERVAL)
