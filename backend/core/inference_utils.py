"""Shared inference helpers: device, FP16, torch.compile, ONNX."""

from __future__ import annotations

import logging
import platform

import numpy as np
import torch

from core.ml_config import load_calibration, ONNX_MODEL_PATH

logger = logging.getLogger("voiceshield.inference")

_ONNX_SESSION = None
_ONNX_INPUT_NAMES = None
_ONNX_OUTPUT_NAME = None
_ONNX_FAILED = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_fp16() -> bool:
    cfg = load_calibration()
    return bool(cfg.get("use_fp16_on_gpu")) and torch.cuda.is_available()


def maybe_compile(model):
    cfg = load_calibration()
    if not cfg.get("use_torch_compile", False):
        return model
    if platform.system() == "Windows":
        return model
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        logger.warning(f"torch.compile unavailable: {e}")
        return model


def prepare_model(model, device: torch.device):
    model.to(device)
    model.eval()
    if use_fp16():
        model.half()
    return maybe_compile(model)


def to_model_dtype(tensor: torch.Tensor, model) -> torch.Tensor:
    if next(model.parameters()).dtype == torch.float16:
        return tensor.half()
    return tensor.float()


def get_onnx_session():
    """Load ONNX Runtime session if exported model exists."""
    global _ONNX_SESSION, _ONNX_INPUT_NAMES, _ONNX_OUTPUT_NAME, _ONNX_FAILED
    cfg = load_calibration()
    if _ONNX_FAILED or not cfg.get("prefer_onnx", True):
        return None, None, None
    if _ONNX_SESSION is not None:
        return _ONNX_SESSION, _ONNX_INPUT_NAMES, _ONNX_OUTPUT_NAME
    if not ONNX_MODEL_PATH.exists():
        return None, None, None
    try:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        _ONNX_SESSION = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=providers)
        _ONNX_INPUT_NAMES = [i.name for i in _ONNX_SESSION.get_inputs()]
        _ONNX_OUTPUT_NAME = _ONNX_SESSION.get_outputs()[0].name
        print(f"[OK] ONNX deepfake model loaded ({ONNX_MODEL_PATH.name})")
        return _ONNX_SESSION, _ONNX_INPUT_NAMES, _ONNX_OUTPUT_NAME
    except Exception as e:
        logger.warning(f"ONNX load failed ({e}); using PyTorch.")
        _ONNX_FAILED = True
        return None, None, None


def run_onnx_logits(session, input_names, output_name, input_values: dict) -> np.ndarray:
    feed = {name: input_values[name] for name in input_names if name in input_values}
    return session.run([output_name], feed)[0]
