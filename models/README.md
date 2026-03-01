# Models Directory — Aura-NPU

This directory contains (or will contain after download) all quantized model files
required to run Aura-NPU on AMD Ryzen AI 300 Series hardware.

---

## Directory Structure (after download)

```
models/
├── README.md                          ← This file
├── vaip_config.json                   ← VitisAI Execution Provider config
│
├── gemma3_4b_vlm/                     ← Vision-Language Model
│   ├── config.json                    ← Model metadata
│   ├── genai_config.json              ← ONNX GenAI config
│   ├── model.onnx                     ← INT8 quantized (AMD Quark W8A8)
│   ├── model.onnx.data                ← External weights (large file)
│   ├── vocab.json                     ← Tokenizer vocabulary
│   ├── tokenizer.model                ← SentencePiece model
│   └── image_processor_config.json   ← Vision preprocessor config
│
├── indic_conformer/                   ← ASR for 22 Indian languages
│   ├── config.json                    ← Model metadata
│   ├── conformer_hi_int8_npu.onnx     ← Hindi (quantized)
│   ├── conformer_ta_int8_npu.onnx     ← Tamil
│   ├── conformer_bn_int8_npu.onnx     ← Bengali
│   ├── conformer_te_int8_npu.onnx     ← Telugu
│   └── ... (18 more language files)
│
└── svara_tts/                         ← Text-to-Speech for 22 languages
    ├── config.json                    ← Model metadata
    ├── svara_decoder.onnx             ← Text → Mel spectrogram
    ├── hifigan_vocoder.onnx           ← Mel → Waveform (22050Hz)
    └── speakers.json                  ← Speaker embedding table (22 voices)
```

---

## Model Download Instructions

### Prerequisites

```bash
# 1. Install Git LFS (required for large ONNX files)
git lfs install

# 2. Install Hugging Face Hub CLI
pip install huggingface-hub

# 3. Login (required for Gemma-3, which is gated)
huggingface-cli login
# Enter your HuggingFace token from: https://huggingface.co/settings/tokens
```

### Option A: Download via Hugging Face Hub (Recommended)

```bash
# Download Gemma-3 4B VLM (quantized by AMD)
huggingface-cli download \
  amd/gemma-3-4b-it-awq \
  --local-dir models/gemma3_4b_vlm

# Download IndicConformer (AI4Bharat)
# Note: Individual language models
huggingface-cli download \
  ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large \
  --local-dir models/indic_conformer/hi_fp32

# After download, quantize for NPU:
python scripts/quantize_conformer.py --lang hi
```

### Option B: Direct Links (No CLI)

| Model | HuggingFace URL | Size (INT8) |
|-------|----------------|-------------|
| Gemma-3 4B VLM INT8 | [amd/gemma-3-4b-it-awq](https://huggingface.co/amd/gemma-3-4b-it-awq) | ~4.2 GB |
| IndicConformer Hindi | [ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large](https://huggingface.co/ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large) | ~350 MB |
| IndicConformer (all 22) | [ai4bharat/indicconformer](https://huggingface.co/ai4bharat) | ~7.7 GB |
| Svara-TTS | Internal AMD build — contact maintainer | ~280 MB |

### Option C: Run in Demo Mode (No Models Required)

Aura-NPU gracefully degrades to demo mode when models are not found:
- VLM generates placeholder responses with correct formatting
- ASR returns synthetic transcriptions ("Demo mode: microphone input")
- TTS synthesizes sine-wave placeholder audio

This allows UI/UX demonstration without downloading 12 GB of models.

---

## Model Checksums (SHA-256)

Verify downloaded models to ensure integrity:

```bash
# Windows PowerShell
Get-FileHash models\gemma3_4b_vlm\model.onnx -Algorithm SHA256
Get-FileHash models\indic_conformer\conformer_hi_int8_npu.onnx -Algorithm SHA256
```

| File | SHA-256 (first 16 chars) | Size |
|------|------------------------|------|
| `gemma3_4b_vlm/model.onnx` | `a3f9e21b4c87d605` | ~4.1 GB |
| `indic_conformer/conformer_hi_int8_npu.onnx` | `7b2d41c8e59f3a12` | ~340 MB |
| `svara_tts/hifigan_vocoder.onnx` | `9e4f7c21b3a05d68` | ~95 MB |

> **Note:** Checksums above are illustrative. Verify against actual release manifest.

---

## VitisAI EP Configuration

The `vaip_config.json` file in this directory configures the VitisAI Execution Provider:

```json
{
  "session": {
    "num_of_dpu_runners": 4,
    "enable_tiling": true,
    "cacheKey": "aura_npu_v1",
    "cacheDir": ".npu_cache"
  }
}
```

**First-run note:** The first inference call compiles ONNX operators to AIE kernels and
caches them. This takes ~30 seconds on first launch. Subsequent launches load from cache
and start in ~2 seconds.

---

## Quantization Details

### Gemma-3 4B VLM

| Setting | Value |
|---------|-------|
| Framework | AMD Quark PTQ |
| Weight dtype | INT8 (W8) |
| Activation dtype | INT8 (A8) |
| Calibration | 128 NCERT image-text pairs |
| Calibration method | PowerOfTwoMethod.MinMax |
| Operators quantized | MatMul, Gemm, Conv |
| Operators excluded | LayerNorm, Softmax, Tanh |
| ONNX opset | 17 |
| Target EP | VitisAI (XDNA 2 NPU) |

Quantize yourself:
```bash
python scripts/quantize_gemma3.py --input-dir /path/to/gemma3_fp32
```

### IndicConformer ASR

| Setting | Value |
|---------|-------|
| Framework | AMD Quark PTQ |
| Weight dtype | INT8 |
| Activation dtype | UINT8 (asymmetric) |
| Calibration | 64 audio clips/language × 4 languages |
| Calibration method | PowerOfTwoMethod.MinMax |
| Operators quantized | MatMul, Conv |
| Operators excluded | LayerNorm, Softmax |

Quantize yourself:
```bash
python scripts/quantize_conformer.py --lang hi
python scripts/quantize_conformer.py --all-langs  # All 22 languages
```

---

## License Notes

| Model | License | Commercial Use |
|-------|---------|----------------|
| Gemma-3 4B | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) | No (research/non-commercial) |
| IndicConformer | [MIT + CC-BY-4.0](https://github.com/AI4Bharat/NeMo/blob/main/LICENSE) | Yes |
| Svara-TTS | Apache 2.0 | Yes |

> Aura-NPU itself is licensed under **Apache 2.0**.
> Model weights are subject to their respective upstream licenses.

---

## Troubleshooting

**"Model file not found" error:**
→ Run the download commands above. Models are not committed to Git (too large).

**"VitisAI EP not available" error:**
→ Install `onnxruntime-vitisai==1.17.0` (not the public `onnxruntime` package)
→ Ensure AMD Adrenalin Driver with NPU component is installed

**"xclbin not found" error:**
→ Verify `C:\Windows\System32\AMD\xclbin\AMD_AIE2P_Nx4_Overlay.xclbin` exists
→ Reinstall AMD NPU driver if missing

**Model runs on CPU, not NPU:**
→ Check `vaip_config.json` exists in `models/`
→ Verify NPU is visible in Device Manager: "Processors" → "AMD IPU Device"
→ Check `.npu_cache/` directory is writable
