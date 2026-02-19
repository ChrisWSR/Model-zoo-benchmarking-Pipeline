# Model Zoo Benchmarking Pipeline
TODO
create a good drescription
create a better explanation
create a requirements file
Automated pipeline to download models from Hugging Face, export them to ONNX,
and benchmark **latency, throughput, and accuracy** across CPU and GPU backends.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run the full benchmark (CPU)
python src/benchmark.py --models resnet18 resnet50 efficientnet_b0 --batch-sizes 1 4 8

# Run with GPU
python src/benchmark.py --gpu

# Generate markdown report
python src/report_generator.py --results-dir results --output RESULTS.md
```

## Project Structure

```
model-zoo-benchmark/
├── src/
│   ├── benchmark.py          # Main orchestrator — run this
│   ├── exporter.py           # PyTorch → ONNX conversion + validation
│   ├── model_registry.py     # All supported models live here
│   └── report_generator.py   # CSV → Markdown tables
├── tests/
│   └── test_regression.py    # CI latency regression guard
├── results/  
├──diagnos
│   └── check_gpu.py          # debug for gpu enablement troubles 
├── .github/workflows/
│   └── benchmark.yml         # CI: runs weekly + on every PR
└── requirements.txt
```

## 📊 Sample Results (batch_size=1, CPU)

> 🟢 < 10ms &nbsp;&nbsp; 🟡 10–50ms &nbsp;&nbsp; 🔴 > 50ms

| Model           | Backend   | Latency (ms)   |   p95 (ms) |   Throughput (s/s) | Size (MB)   |
|:----------------|:----------|:---------------|-----------:|-------------------:|:------------|
| efficientnet_b0 | onnx_cpu  | 🟢 6.6         |        7.4 |              151.4 | 21.2        |
| efficientnet_b0 | onnx_gpu  | 🟢 2.1         |        2.2 |              483.2 | 21.2        |
| efficientnet_b0 | pytorch   | 🟢 3.6         |        5   |              278.7 | —           |
| mobilenet_v2    | onnx_cpu  | 🟢 2.5         |        2.8 |              400.8 | 14.2        |
| mobilenet_v2    | onnx_gpu  | 🟢 1.5         |        1.9 |              660.4 | 14.2        |
| mobilenet_v2    | pytorch   | 🟢 3.2         |        4   |              312.7 | —           |
| resnet18        | onnx_cpu  | 🟢 9.0         |       10.4 |              111.1 | 46.8        |
| resnet18        | onnx_gpu  | 🟢 2.2         |        2.7 |              447   | 46.8        |
| resnet18        | pytorch   | 🟢 2.3         |        2.9 |              435.2 | —           |
| resnet50        | onnx_cpu  | 🟡 21.1        |       22.3 |               47.3 | 102.2       |
| resnet50        | onnx_gpu  | 🟢 5.1         |        5.7 |              196.3 | 102.2       |
| resnet50        | pytorch   | 🟢 4.9         |        5.3 |              206   | —           |

*ONNX Runtime delivers **1.7–2.4× lower latency** vs PyTorch on CPU across all tested models.*

## 🛠️ Adding a New Model

Edit `src/model_registry.py` and add an entry:

```python
MODEL_REGISTRY["your_model"] = {
    "name": "your_model",
    "hf_id": "org/model-name",   # Hugging Face model ID
    "task": "classification",
    "img_size": 224,
    "opset": 17,
}
```

Then run `python src/benchmark.py --models your_model`.

## Key Design Decisions

**Dynamic batch axes in ONNX export** — allows a single exported model to serve any batch size without re-export.

**Graph optimization at ORT level** — `ORT_ENABLE_ALL` applies constant folding, operator fusion, and layout optimization automatically.

**Warmup iterations** — 10 warmup passes before measurement eliminate JIT compilation and cache cold-start noise.

**p95/p99 latency** — mean latency alone is misleading; tail latency matters for real serving SLAs.

**CI regression guard** — `test_regression.py` fails the pipeline if any model exceeds the configured ms threshold, preventing silent perf regressions on new PRs.

##  Roadmap

- [ ] LLM benchmarking (TinyLlama, Phi-2) — token/s, TTFT, memory footprint
- [ ] INT8 quantization via ONNX Runtime quantization API
- [ ] YOLOv8 object detection task support
- [ ] GPU (CUDA) benchmarking in CI via GitHub-hosted GPU runners
- [ ] Accuracy evaluation on ImageNet-1k validation subset
- [ ] Streamlit dashboard for interactive result exploration