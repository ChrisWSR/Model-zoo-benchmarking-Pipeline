"""
ONNX Exporter — handles conversion from PyTorch → ONNX with validation.
"""

import logging
import warnings
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
import numpy as np
from transformers import AutoModelForImageClassification

log = logging.getLogger(__name__)


def export_to_onnx(model_cfg: dict, output_path: str, batch_size: int = 1) -> str:
    """
    Export a HuggingFace model to ONNX format with dynamic batch axes.
    Returns the path to the exported model.
    """
    log.info(f"Loading {model_cfg['hf_id']} for ONNX export...")
    model = AutoModelForImageClassification.from_pretrained(model_cfg["hf_id"])
    model.eval()

    img_size = model_cfg["img_size"]
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    log.info(f"Exporting to {output_path} (opset={model_cfg['opset']})...")
    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", message=".*Converting a tensor to a Python boolean.*")
            torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=model_cfg["opset"],
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    # Validate the exported model
    _validate_onnx(output_path, dummy_input.numpy())
    log.info(f"✓ Export complete and validated: {output_path}")
    return output_path


def _validate_onnx(onnx_path: str, sample_input: np.ndarray):
    """Check ONNX graph is valid and ORT can run inference."""
    # Graph-level check
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Runtime check
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: sample_input.astype(np.float32)})
    assert outputs is not None and len(outputs) > 0, "ORT returned empty output"

    log.info(f"  Graph check ✓ | ORT inference check ✓ | Output shape: {outputs[0].shape}")