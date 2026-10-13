import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Apple Silicon: fall back to CPU for unsupported ops

import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def download_and_cache_models(model_id=MODEL_ID):

    print(f"[+] Loading '{model_id}' (device: {DEVICE})")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    print("[+] Model ready.")
    return processor, model


if __name__ == "__main__":
    download_and_cache_models()
