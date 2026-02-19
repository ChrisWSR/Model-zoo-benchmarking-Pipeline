"""
Model Registry — single source of truth for all supported models.
Add new models here and they're automatically picked up by the pipeline.
"""

MODEL_REGISTRY = {
    "resnet50": {
        "name": "resnet50",
        "hf_id": "microsoft/resnet-50",
        "task": "classification",
        "img_size": 224,
        "opset": 17,
    },
    "resnet18": {
        "name": "resnet18",
        "hf_id": "microsoft/resnet-18",
        "task": "classification",
        "img_size": 224,
        "opset": 17,
    },
    "efficientnet_b0": {
        "name": "efficientnet_b0",
        "hf_id": "google/efficientnet-b0",
        "task": "classification",
        "img_size": 224,
        "opset": 17,
    },
    "mobilenet_v2": {
        "name": "mobilenet_v2",
        "hf_id": "google/mobilenet_v2_1.0_224",
        "task": "classification",
        "img_size": 224,
        "opset": 17,
    },
    # LLMs (text tasks handled separately in llm_benchmark.py) TODO
    # "tinyllama": {
    #     "name": "tinyllama",
    #     "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     "task": "text-generation",
    #     "img_size": None,
    # },
}