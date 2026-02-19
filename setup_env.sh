#!/bin/bash

# ── Model Zoo Benchmark — Environment Setup (venv) ─────────────────────────
set -e

ENV_NAME="model-zoo-env"

echo "Setting up Model Zoo Benchmark environment..."

# ── Check prerequisites ────────────────────────────────────────────────────
echo ""
echo "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "Python not found. Install Python 3.11 from https://python.org"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "  Python $PYTHON_VERSION found"

if command -v nvidia-smi &> /dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU found: $GPU"
    HAS_GPU=true
else
    echo "  ⚠️  No GPU detected — installing CPU-only packages"
    HAS_GPU=false
fi

# ── Create venv ────────────────────────────────────────────────────────────
echo ""
echo "🐍 Creating virtual environment '$ENV_NAME'..."

if [ -d "$ENV_NAME" ]; then
    echo "  ⚠️  '$ENV_NAME' folder already exists."
    read -p "  Recreate it? (y/n): " RECREATE
    if [[ "$RECREATE" == "y" ]]; then
        rm -rf $ENV_NAME
    else
        echo "  Keeping existing environment."
    fi
fi

python3 -m venv $ENV_NAME
echo "  Virtual environment created"

# ── Activate ───────────────────────────────────────────────────────────────
echo ""
echo "⚡ Activating environment..."
source $ENV_NAME/bin/activate
pip install --upgrade pip -q

# ── Install PyTorch ────────────────────────────────────────────────────────
echo ""
echo "Installing PyTorch..."
if [ "$HAS_GPU" = true ]; then
    pip install "torch>=2.6.0" torchvision --index-url https://download.pytorch.org/whl/cu124 -q
    echo "PyTorch installed (CUDA 12.4)"
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
    echo "  PyTorch installed (CPU only)"
fi

# ── Install ONNX Runtime ───────────────────────────────────────────────────
echo ""
echo "  Installing ONNX Runtime..."
if [ "$HAS_GPU" = true ]; then
    pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ -q
    echo "  onnxruntime-gpu installed"
else
    pip install onnxruntime -q
    echo "   onnxruntime (CPU) installed"
fi

# ── Install dependencies ───────────────────────────────────────────────────
echo ""
echo " Installing dependencies..."

if [ "$HAS_GPU" = true ]; then
    REQ_FILE="requirements-gpu.txt"
else
    REQ_FILE="requirements-cpu.txt"
fi

if [ ! -f "$REQ_FILE" ]; then
    echo " $REQ_FILE not found"
    exit 1
fi

pip install -r "$REQ_FILE" -q
echo "  All packages installed from $REQ_FILE"

# ── Verify ────────────────────────────────────────────────────────────────
echo ""
echo "Verifying installation..."
python3 - <<'EOF'
import torch
import onnxruntime as ort
import transformers
import pandas

print(f"  torch         : {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU           : {torch.cuda.get_device_name(0)}")
print(f"  onnxruntime   : {ort.__version__}")
print(f"  ORT providers : {ort.get_available_providers()}")
print(f"  transformers  : {transformers.__version__}")
print(f"  pandas        : {pandas.__version__}")
EOF


# ── Done ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Environment ready!"
echo ""
echo "   Activate it anytime with:"
echo "    source $ENV_NAME/bin/activate"
echo "   and Deactivate it with:"
echo "    deactivate"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"