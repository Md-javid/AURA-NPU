#!/usr/bin/env python3
"""
scripts/download_models.py
==========================
Download and verify Aura-NPU model configs and (optionally) INT8 ONNX weights.

Usage:
    python scripts/download_models.py             # create dir structure + configs only
    python scripts/download_models.py --weights   # also download ONNX weights
    python scripts/download_models.py --verify    # verify existing downloads

All models run 100% offline after download.  No cloud calls at inference time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

# Keep imports minimal — this script must run outside the venv to bootstrap it
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
)
log = logging.getLogger("aura.download")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# ─────────────────────────────────────────────────────────────
#  Model registry
# ─────────────────────────────────────────────────────────────

MODEL_CONFIGS: dict[str, dict] = {
    "gemma3_4b_vlm": {
        "description": "Gemma-3 4B Vision Language Model (INT8 quantized for NPU)",
        "dir": MODELS_DIR / "gemma3_4b_vlm",
        "config": {
            "model_type": "gemma3_vlm",
            "quant_bits": 8,
            "provider": "VitisAIExecutionProvider",
            "input_size": [1, 3, 336, 336],
            "max_tokens": 512,
            "tokenizer": "google/gemma-3-4b-it",
            "note": "Download model_quantized.onnx from AMD Ryzen AI model zoo"
        },
        "weight_filename": "model_quantized.onnx",
        "weight_size_gb": 3.8,
        "weight_source": "AMD Ryzen AI Model Zoo — https://ryzenai.docs.amd.com/en/latest/modelzoo.html",
        "sha256": None,  # set after official release
    },
    "indic_conformer": {
        "description": "AI4Bharat IndicConformer ASR — 22 Indian languages (INT8)",
        "dir": MODELS_DIR / "indic_conformer",
        "config": {
            "model_type": "conformer_ctc",
            "quant_bits": 8,
            "provider": "VitisAIExecutionProvider",
            "sample_rate": 16000,
            "languages": ["hi", "ta", "te", "bn", "mr", "gu", "kn", "ml",
                          "pa", "or", "as", "ur", "sa", "ks", "ne", "sd",
                          "mai", "kok", "doi", "mni", "sat", "bho"],
            "note": "Download from AI4Bharat HuggingFace — ai4bharat/indicconformer"
        },
        "weight_filename": "model_quantized.onnx",
        "weight_size_gb": 0.8,
        "weight_source": "https://huggingface.co/ai4bharat/indicconformer",
        "sha256": None,
    },
    "svara_tts": {
        "description": "Svara TTS — multilingual Indian language voice synthesis (INT8)",
        "dir": MODELS_DIR / "svara_tts",
        "config": {
            "model_type": "vits_tts",
            "quant_bits": 8,
            "provider": "VitisAIExecutionProvider",
            "sample_rate": 22050,
            "languages": ["hi", "ta", "te", "bn", "mr", "gu", "kn", "ml"],
            "note": "Download from AI4Bharat HuggingFace — ai4bharat/indic-tts"
        },
        "weight_filename": "model_quantized.onnx",
        "weight_size_gb": 0.4,
        "weight_source": "https://huggingface.co/ai4bharat/indic-tts",
        "sha256": None,
    },
}


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_config(model_dir: Path, config_dict: dict) -> None:
    config_path = model_dir / "config.json"
    config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    log.info("  ✅ Config written: %s", config_path)


def create_readme(model_dir: Path, meta: dict) -> None:
    readme = model_dir / "README.md"
    size = meta.get("weight_size_gb", "?")
    source = meta.get("weight_source", "See AMD Ryzen AI documentation")
    desc = meta.get("description", model_dir.name)
    filename = meta.get("weight_filename", "model_quantized.onnx")
    content = (
        f"# {model_dir.name}\n\n"
        f"{desc}\n\n"
        f"## Model weight download\n\n"
        f"**File**: `{filename}` (~{size} GB)\n\n"
        f"**Source**: {source}\n\n"
        f"Place the downloaded file at:\n```\n{model_dir / filename}\n```\n\n"
        f"## Quantization\n\n"
        f"Models are quantized to INT8 using AMD Quark for VitisAI EP deployment.\n"
        f"See `scripts/quantize.py` for the quantization pipeline.\n"
    )
    readme.write_text(content, encoding="utf-8")


# ─────────────────────────────────────────────────────────────
#  Main actions
# ─────────────────────────────────────────────────────────────

def create_structure() -> None:
    """Create directory structure and write config.json files."""
    log.info("Creating model directory structure...")
    MODELS_DIR.mkdir(exist_ok=True)

    for name, meta in MODEL_CONFIGS.items():
        model_dir: Path = meta["dir"]
        model_dir.mkdir(exist_ok=True)
        write_config(model_dir, meta["config"])
        create_readme(model_dir, meta)
        log.info("  📁 %s ready  (weight: %s GB — download separately)",
                 name, meta["weight_size_gb"])

    # Write top-level vaip config if missing
    vaip = MODELS_DIR / "vaip_config.json"
    if not vaip.exists():
        vaip.write_text(json.dumps({
            "target_name": "AMD_AIE2P_Nx4_Overlay",
            "cache_dir": str(ROOT / ".npu_cache"),
            "enable_cache": True,
            "optimization_level": 3,
            "note": "VitisAI Execution Provider configuration for AMD XDNA 2"
        }, indent=2), encoding="utf-8")
        log.info("  ✅ vaip_config.json written")

    log.info("Structure created. To download model weights, run:")
    log.info("  python scripts/download_models.py --weights")


def verify() -> bool:
    """Check which models are downloaded and report status."""
    log.info("Verifying model files...")
    all_ok = True
    for name, meta in MODEL_CONFIGS.items():
        weight_path = meta["dir"] / meta["weight_filename"]
        config_path = meta["dir"] / "config.json"

        config_ok = config_path.exists()
        weight_ok = weight_path.exists()
        sha_ok = True

        if weight_ok and meta.get("sha256"):
            actual = sha256_file(weight_path)
            sha_ok = actual == meta["sha256"]

        status = "✅" if (config_ok and weight_ok and sha_ok) else "⚠ "
        log.info("  %s %s", status, name)
        log.info("      config : %s", "✅" if config_ok else "❌ missing")
        log.info("      weights: %s", "✅" if weight_ok else f"❌ missing ({meta['weight_size_gb']} GB)")
        if weight_ok and meta.get("sha256"):
            log.info("      sha256 : %s", "✅" if sha_ok else "❌ MISMATCH")

        if not (config_ok and weight_ok and sha_ok):
            all_ok = False

    return all_ok


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aura-NPU model setup utility"
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Attempt to download model weights (requires weight_source URLs)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded model files",
    )
    args = parser.parse_args()

    if args.verify:
        ok = verify()
        return 0 if ok else 1

    if args.weights:
        log.warning(
            "Automatic weight download is not yet implemented.\n"
            "  Model weights must be downloaded manually from the sources listed below:\n"
        )
        for name, meta in MODEL_CONFIGS.items():
            log.warning("  %s: %s", name, meta["weight_source"])
        log.warning(
            "\nAfter downloading, place each file at:\n"
            "  models/<model_name>/model_quantized.onnx"
        )
        return 1

    create_structure()
    return 0


if __name__ == "__main__":
    sys.exit(main())
