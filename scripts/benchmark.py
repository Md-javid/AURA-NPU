#!/usr/bin/env python3
"""
scripts/benchmark.py
====================
Offline hardware benchmark for Aura-NPU.

Measures:
  • PIL offline baseline (always available)
  • Ollama latency (if running)
  • NPU inference (if model present)

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --runs 10
    python scripts/benchmark.py --json results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
)
log = logging.getLogger("aura.benchmark")


def _make_test_image_b64() -> str:
    """Generate a small solid-colour JPEG as a benchmark payload."""
    import base64
    import io
    from PIL import Image

    img = Image.new("RGB", (336, 336), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _benchmark_pil(n: int, image_b64: str) -> dict:
    from app.npu_engine import AuraNPUEngine
    engine = AuraNPUEngine()

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        engine._analyse_with_pil(image_b64, "Describe this image.")
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "provider": "PIL/Offline",
        "runs": n,
        "mean_ms": round(sum(times) / n, 1),
        "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1),
    }


async def _benchmark_ollama(n: int, image_b64: str) -> dict:
    import os
    import httpx
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llava")

    # Check Ollama is up
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(host + "/api/tags")
            if r.status_code != 200:
                return {"provider": "Ollama", "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"provider": "Ollama", "error": str(e)}

    times = []
    for i in range(n):
        t0 = time.perf_counter()
        payload = {
            "model": model,
            "prompt": "Describe this image in one sentence.",
            "images": [image_b64],
            "stream": False,
            "options": {"num_predict": 80},
        }
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(host + "/api/generate", json=payload)
            r.raise_for_status()
            times.append((time.perf_counter() - t0) * 1000)
            log.info("  Ollama run %d/%d — %.0f ms", i + 1, n, times[-1])
        except Exception as e:
            log.warning("  Ollama run %d failed: %s", i + 1, e)

    if not times:
        return {"provider": "Ollama", "error": "All runs failed"}

    return {
        "provider": f"Ollama ({model})",
        "runs": len(times),
        "mean_ms": round(sum(times) / len(times), 1),
        "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1),
    }


def _print_table(results: list[dict]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Aura-NPU Benchmark Results", style="bold")
    table.add_column("Provider", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Mean ms", justify="right", style="green")
    table.add_column("Min ms", justify="right")
    table.add_column("Max ms", justify="right")
    table.add_column("Status", justify="center")

    baseline = None
    for r in results:
        if "error" in r:
            table.add_row(r["provider"], "-", "-", "-", "-", "❌ " + r["error"])
        else:
            if r["provider"].startswith("PIL"):
                baseline = r["mean_ms"]
            speedup = ""
            if baseline and not r["provider"].startswith("PIL") and r.get("mean_ms"):
                ratio = r["mean_ms"] / baseline
                speedup = f"  ({ratio:.1f}x vs PIL)"
            table.add_row(
                r["provider"],
                str(r["runs"]),
                f"{r['mean_ms']:.1f}" + speedup,
                f"{r['min_ms']:.1f}",
                f"{r['max_ms']:.1f}",
                "✅",
            )

    console.print(table)


async def _main(runs: int, output: str | None) -> None:
    log.info("Generating test image...")
    image_b64 = _make_test_image_b64()

    results: list[dict] = []

    log.info("Benchmarking PIL offline baseline (%d runs)...", runs)
    try:
        pil_result = _benchmark_pil(runs, image_b64)
        results.append(pil_result)
        log.info("  PIL mean: %.1f ms", pil_result["mean_ms"])
    except Exception as e:
        results.append({"provider": "PIL/Offline", "error": str(e)})
        log.error("  PIL failed: %s", e)

    log.info("Benchmarking Ollama (%d runs)...", min(runs, 3))
    ollama_result = await _benchmark_ollama(min(runs, 3), image_b64)
    results.append(ollama_result)

    _print_table(results)

    if output:
        Path(output).write_text(json.dumps(results, indent=2), encoding="utf-8")
        log.info("Results saved to %s", output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aura-NPU offline benchmark")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per provider")
    parser.add_argument("--json", dest="output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    asyncio.run(_main(args.runs, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
