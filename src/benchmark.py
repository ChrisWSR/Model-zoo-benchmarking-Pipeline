"""
Autor CWSR
Model Zoo Benchmarking Pipeline
Core benchmarking engine — runs latency, throughput, and accuracy tests
across PyTorch and ONNX Runtime backends.
"""

import time
import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import sys
from pathlib import Path
# sys.path.append(str(Path(__file__).parent))  # makes src/ imports work from anywhere

import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

from exporter import export_to_onnx
from model_registry import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    model_name: str
    backend: str           # "pytorch" | "onnx_cpu" | "onnx_gpu"
    task: str              # "classification" | "detection" | "text-generation"
    batch_size: int
    input_shape: tuple
    # Latency
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    # Throughput
    throughput_samples_per_sec: float
    # Quality
    top1_accuracy: Optional[float] = None
    # Meta
    model_size_mb: Optional[float] = None
    device: str = "cpu"


def measure_latency(run_fn, warmup: int = 10, iterations: int = 100, is_cuda: bool = False) -> dict:
    """Warm up then measure latency over N iterations."""
    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    for _ in range(warmup):
        run_fn()
    sync()  # flush warmup queue before measuring

    latencies = []
    for _ in range(iterations):
        sync()                                          # GPU idle before start
        start = time.perf_counter()
        run_fn()
        sync()                                          # GPU done before stop
        latencies.append((time.perf_counter() - start) * 1000)

    arr = np.array(latencies)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "throughput": 1000.0 / arr.mean(),  # samples/sec (batch_size=1 baseline)
    }


def benchmark_pytorch(model_cfg: dict, batch_size: int, device: str) -> BenchmarkResult:
    log.info(f"[PyTorch] Benchmarking {model_cfg['name']} on {device}")
    model = AutoModelForImageClassification.from_pretrained(model_cfg["hf_id"])
    model = model.to(device).eval()

    dummy = torch.randn(batch_size, 3, model_cfg["img_size"], model_cfg["img_size"]).to(device)

    with torch.no_grad():
        metrics = measure_latency(lambda: model(dummy), is_cuda=(device == "cuda"))

    return BenchmarkResult(
        model_name=model_cfg["name"],
        backend="pytorch",
        task=model_cfg["task"],
        batch_size=batch_size,
        input_shape=(batch_size, 3, model_cfg["img_size"], model_cfg["img_size"]),
        mean_latency_ms=metrics["mean"],
        p50_latency_ms=metrics["p50"],
        p95_latency_ms=metrics["p95"],
        p99_latency_ms=metrics["p99"],
        throughput_samples_per_sec=metrics["throughput"] * batch_size,
        device=device,
    )


def benchmark_onnx(onnx_path: str, model_cfg: dict, batch_size: int, use_gpu: bool) -> BenchmarkResult:
    provider = "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"
    backend_tag = "onnx_gpu" if use_gpu else "onnx_cpu"

    log.info(f"[ONNX/{provider}] Benchmarking {model_cfg['name']}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4

    # For CUDA provider, pass explicit options to avoid silent CPU fallback
    providers = (
        [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        if use_gpu else ["CPUExecutionProvider"]
    )

    session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)

    # Log which provider is actually being used (catches silent fallbacks)
    active_provider = session.get_providers()[0]
    if use_gpu and active_provider != "CUDAExecutionProvider":
        log.warning(f"  ⚠️  Requested CUDA but running on {active_provider} — check onnxruntime-gpu install")
    else:
        log.info(f"  ✅ Active provider: {active_provider}")

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(batch_size, 3, model_cfg["img_size"], model_cfg["img_size"]).astype(np.float32)

    metrics = measure_latency(lambda: session.run(None, {input_name: dummy}), is_cuda=use_gpu)

    return BenchmarkResult(
        model_name=model_cfg["name"],
        backend=backend_tag,
        task=model_cfg["task"],
        batch_size=batch_size,
        input_shape=dummy.shape,
        mean_latency_ms=metrics["mean"],
        p50_latency_ms=metrics["p50"],
        p95_latency_ms=metrics["p95"],
        p99_latency_ms=metrics["p99"],
        throughput_samples_per_sec=metrics["throughput"] * batch_size,
        model_size_mb=Path(onnx_path).stat().st_size / 1e6,
        device="cuda" if use_gpu else "cpu",
    )


def run_pipeline(models: list[str], batch_sizes: list[int], output_dir: str, gpu: bool):
    results = []
    base = output_dir.rstrip("/").rstrip("_rst")
    output_path = Path(f"{base}_rst")
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_path / "onnx_models"
    onnx_dir.mkdir(exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
        log.info(f"GPU detected: {torch.cuda.get_device_name(0)} — using CUDA")
    else:
        device = "cpu"
        log.info("No GPU detected — using CPU")


    for model_name in models:
        cfg = MODEL_REGISTRY.get(model_name)
        if not cfg:
            log.warning(f"Model '{model_name}' not found in registry. Skipping.")
            continue

        onnx_path = onnx_dir / f"{model_name}.onnx"

        # Export once
        if not onnx_path.exists():
            log.info(f"Exporting {model_name} to ONNX...")
            export_to_onnx(cfg, str(onnx_path))

        for bs in batch_sizes:
            try:
                results.append(benchmark_pytorch(cfg, bs, device))
                results.append(benchmark_onnx(str(onnx_path), cfg, bs, use_gpu=False))
                if torch.cuda.is_available():
                    results.append(benchmark_onnx(str(onnx_path), cfg, bs, use_gpu=True))
            except Exception as e:
                log.error(f"Failed on {model_name} bs={bs}: {e}")

    # Save results
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_path / "benchmark_results.csv", index=False)

    # Pretty summary table
    summary = df.pivot_table(
        index=["model_name", "batch_size"],
        columns="backend",
        values=["mean_latency_ms", "throughput_samples_per_sec"],
        aggfunc="mean",
    ).round(2)
    summary.to_csv(output_path / "summary_table.csv")

    log.info(f"\n{'='*60}")
    log.info("BENCHMARK SUMMARY")
    log.info(f"{'='*60}")
    print(summary.to_string())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Zoo Benchmarking Pipeline")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                        help="Models to benchmark (from registry)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU benchmarking")
    args = parser.parse_args()

    run_pipeline(
        models=args.models,
        batch_sizes=args.batch_sizes,
        output_dir=args.output_dir,
        gpu=args.gpu,
    )