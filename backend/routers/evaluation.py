"""VoiceShield AI — Benchmark & Evaluation Router"""
import asyncio

from fastapi import APIRouter

from core.evaluation import run_benchmark, report_to_dict, MODEL_CITATIONS

router = APIRouter()


@router.get("/benchmark")
async def get_benchmark():
    """
    Run full pipeline benchmark on labeled demo clips.
    Returns precision/recall/F1, per-case results, and model citations for SIH presentation.
    """
    report = await asyncio.to_thread(run_benchmark)
    return report_to_dict(report)


@router.get("/benchmark/models")
async def get_model_info():
    """Published model architecture info, training provenance, and literature benchmarks."""
    import json
    from pathlib import Path
    from core.deepfake_detector import get_wav2vec2_model, get_runtime_info
    from core.speaker_verifier import get_ecapa_classifier

    model, _, _ = get_wav2vec2_model()
    ecapa = get_ecapa_classifier()
    provenance_path = Path(__file__).resolve().parents[1] / "data" / "model_provenance.json"
    provenance = {}
    if provenance_path.exists():
        with open(provenance_path, encoding="utf-8") as f:
            provenance = json.load(f)
    return {
        "citations": MODEL_CITATIONS,
        "provenance": provenance,
        "runtime": get_runtime_info(),
        "wav2vec2_loaded": model is not None,
        "ecapa_loaded": ecapa is not None,
    }


@router.get("/performance")
async def get_performance():
    """Runtime optimization status: GPU, FP16, ONNX, calibration."""
    from core.ml_config import load_calibration, ONNX_MODEL_PATH
    from core.speaker_verifier import get_ecapa_classifier
    from core.deepfake_detector import get_runtime_info
    import torch

    cfg = load_calibration()
    return {
        "deepfake_runtime": get_runtime_info(),
        "ecapa_loaded": get_ecapa_classifier() is not None,
        "onnx_available": ONNX_MODEL_PATH.exists(),
        "cuda_available": torch.cuda.is_available(),
        "calibration": cfg,
        "optimizations": [
            "GPU + FP16 inference (when CUDA available)",
            "torch.compile on Wav2Vec2",
            "ONNX Runtime fallback (run scripts/export_onnx.py)",
            "90/10 neural/DSP ensemble",
            "Skip DSP when neural confidence >= threshold",
            "3s analysis window + parallel deepfake/prosody",
            "Calibrated thresholds via calibration.json",
        ],
    }


@router.post("/calibrate")
async def run_calibration():
    """Re-run benchmark and auto-update calibration.json thresholds."""
    import sys
    from pathlib import Path
    from core.risk_engine import risk_engine
    from core.ml_config import load_calibration

    script = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_from_benchmark.py"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(script),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        return {"status": "error", "detail": stderr.decode() or stdout.decode()}
    risk_engine.reload_calibration()
    return {"status": "calibrated", "config": load_calibration()}
