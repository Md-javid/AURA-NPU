#!/usr/bin/env python3
"""
scripts/quantize.py
===================
Unified INT8 PTQ quantization pipeline using AMD Quark.

Quantizes:
  • Gemma-3 4B VLM (Vision Language Model)
  • IndicConformer ASR (22-language speech recognition)
  • Svara TTS (multilingual text-to-speech)

Produces ONNX INT8 models optimised for AMD VitisAI Execution Provider.

Requirements:
    pip install amd-quark onnxruntime-vitisai transformers

Usage:
    python scripts/quantize.py --model vlm
    python scripts/quantize.py --model asr
    python scripts/quantize.py --model tts
    python scripts/quantize.py --model all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
)
log = logging.getLogger("aura.quantize")

MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / ".npu_cache"


# ─────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────

def _check_quark() -> None:
    try:
        import quark  # noqa: F401
    except ImportError:
        log.error(
            "AMD Quark is not installed.\n"
            "  Install: pip install amd-quark\n"
            "  Source:  https://quark.docs.amd.com"
        )
        sys.exit(1)


def _vaip_config() -> dict:
    return {
        "target_name": "AMD_AIE2P_Nx4_Overlay",
        "cache_dir": str(CACHE_DIR),
        "optimization_level": 3,
    }


# ─────────────────────────────────────────────────────────────
#  VLM quantization
# ─────────────────────────────────────────────────────────────

def quantize_vlm() -> None:
    """INT8 PTQ for Gemma-3 4B VLM."""
    model_dir = MODELS_DIR / "gemma3_4b_vlm"
    fp32_path = model_dir / "model_fp32.onnx"
    out_path = model_dir / "model_quantized.onnx"

    if not fp32_path.exists():
        log.error(
            "FP32 model not found: %s\n"
            "  Export Gemma-3 4B to ONNX first:\n"
            "    optimum-cli export onnx --model google/gemma-3-4b-it %s",
            fp32_path, model_dir,
        )
        return

    log.info("Quantizing VLM: %s → %s", fp32_path, out_path)
    _check_quark()

    from quark.onnx import ModelQuantizer, QuantizationConfig  # type: ignore
    from quark.onnx.quantization.config.config import (  # type: ignore
        Config, get_default_config
    )

    quant_config = get_default_config("XINT8")
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(str(fp32_path), str(out_path), calibration_data_path=None)
    log.info("VLM quantization complete: %s", out_path)


# ─────────────────────────────────────────────────────────────
#  ASR quantization
# ─────────────────────────────────────────────────────────────

def quantize_asr() -> None:
    """INT8 PTQ for IndicConformer ASR."""
    model_dir = MODELS_DIR / "indic_conformer"
    fp32_path = model_dir / "model_fp32.onnx"
    out_path = model_dir / "model_quantized.onnx"

    if not fp32_path.exists():
        log.error(
            "FP32 ASR model not found: %s\n"
            "  Convert from NeMo: nemo_to_onnx.py or use HuggingFace exporters.\n"
            "  Source: https://huggingface.co/ai4bharat/indicconformer",
            fp32_path,
        )
        return

    log.info("Quantizing ASR: %s → %s", fp32_path, out_path)
    _check_quark()

    from quark.onnx import ModelQuantizer  # type: ignore
    from quark.onnx.quantization.config.config import get_default_config  # type: ignore

    quant_config = get_default_config("XINT8")
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(str(fp32_path), str(out_path), calibration_data_path=None)
    log.info("ASR quantization complete: %s", out_path)


# ─────────────────────────────────────────────────────────────
#  TTS quantization
# ─────────────────────────────────────────────────────────────

def quantize_tts() -> None:
    """INT8 PTQ for Svara TTS."""
    model_dir = MODELS_DIR / "svara_tts"
    fp32_path = model_dir / "model_fp32.onnx"
    out_path = model_dir / "model_quantized.onnx"

    if not fp32_path.exists():
        log.error(
            "FP32 TTS model not found: %s\n"
            "  Source: https://huggingface.co/ai4bharat/indic-tts",
            fp32_path,
        )
        return

    log.info("Quantizing TTS: %s → %s", fp32_path, out_path)
    _check_quark()

    from quark.onnx import ModelQuantizer  # type: ignore
    from quark.onnx.quantization.config.config import get_default_config  # type: ignore

    quant_config = get_default_config("XINT8")
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(str(fp32_path), str(out_path), calibration_data_path=None)
    log.info("TTS quantization complete: %s", out_path)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

TARGETS = {
    "vlm": quantize_vlm,
    "asr": quantize_asr,
    "tts": quantize_tts,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AMD Quark INT8 quantization pipeline for Aura-NPU models"
    )
    parser.add_argument(
        "--model",
        choices=["vlm", "asr", "tts", "all"],
        required=True,
        help="Model to quantize",
    )
    args = parser.parse_args()

    if args.model == "all":
        for fn in TARGETS.values():
            fn()
    else:
        TARGETS[args.model]()

    return 0


if __name__ == "__main__":
    sys.exit(main())
