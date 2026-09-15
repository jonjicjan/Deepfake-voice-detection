"""
VoiceShield AI — ML runtime configuration.
Tunable without code changes via backend/data/calibration.json
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CALIBRATION_PATH = DATA_DIR / "calibration.json"
ONNX_MODEL_PATH = Path(__file__).resolve().parents[1] / "pretrained_models" / "wav2vec2_deepfake.onnx"

DEFAULTS = {
    "ensemble": {"neural_weight": 0.90, "dsp_weight": 0.10},
    "skip_dsp_when_neural_confident": True,
    "neural_confidence_threshold": 0.85,
    "analysis_window_seconds": 3.0,
    "use_fp16_on_gpu": True,
    "use_torch_compile": False,
    "prefer_onnx": True,
    "deepfake_threshold": 0.50,
    "risk_weights": {
        "deepfake": 0.40,
        "speaker_mismatch": 0.24,
        "prosody_anomaly": 0.16,
        "replay": 0.10,
        "unknown_attack": 0.10,
    },
    "risk_thresholds": {"low": 30.0, "high": 70.0, "critical": 85.0},
}


def load_calibration() -> dict:
    cfg = {**DEFAULTS, "risk_weights": {**DEFAULTS["risk_weights"]}, "risk_thresholds": {**DEFAULTS["risk_thresholds"]}}
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH, encoding="utf-8") as f:
            user = json.load(f)
        for key, val in user.items():
            if isinstance(val, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(val)
            else:
                cfg[key] = val
    return cfg


def save_calibration(updates: dict) -> dict:
    current = load_calibration()
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(current.get(key), dict):
            current[key].update(val)
        else:
            current[key] = val
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
    return current
