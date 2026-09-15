"""
VoiceShield AI — Parallel analysis pipeline.
Runs independent layers concurrently and trims audio to analysis window.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.preprocessor import preprocessor
from core.deepfake_detector import detector
from core.prosody_analyzer import analyzer
from core.attack_classifier import classifier
from core.speaker_verifier import verifier
from core.ml_config import load_calibration


@dataclass
class LayerResults:
    deepfake_result: object
    prosody_result: object
    attack_result: object
    speaker_mismatch: float
    speaker_similarity: float
    speaker_enrolled: bool
    analysis_duration_s: float


def trim_waveform(waveform: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    cfg = load_calibration()
    max_sec = float(cfg.get("analysis_window_seconds", 3.0))
    max_samples = int(max_sec * sr)
    if len(waveform) <= max_samples:
        return waveform, len(waveform) / sr
    start = (len(waveform) - max_samples) // 2
    trimmed = waveform[start : start + max_samples]
    return trimmed, max_sec


def run_parallel_layers(
    waveform: np.ndarray,
    sr: int,
    ref_embedding: Optional[np.ndarray] = None,
) -> LayerResults:
    """Deepfake + prosody in parallel; attack + speaker after."""
    waveform, duration = trim_waveform(waveform, sr)
    spectral_features = preprocessor.get_spectral_features(waveform, sr)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_deepfake = pool.submit(detector.analyze, waveform, sr, spectral_features)
        f_prosody = pool.submit(analyzer.analyze, waveform, sr)
        deepfake_result = f_deepfake.result()
        prosody_result = f_prosody.result()

    attack_result = classifier.classify(
        waveform,
        sr,
        deepfake_result.probability,
        prosody_result.anomaly_probability,
    )

    speaker_mismatch = 0.3
    speaker_similarity = 0.0
    speaker_enrolled = False
    if ref_embedding is not None:
        test_emb = verifier.extract_embedding(waveform, sr)
        verify_result = verifier.verify(test_emb, ref_embedding)
        speaker_mismatch = verify_result.mismatch_probability
        speaker_similarity = verify_result.similarity_score
        speaker_enrolled = True

    return LayerResults(
        deepfake_result=deepfake_result,
        prosody_result=prosody_result,
        attack_result=attack_result,
        speaker_mismatch=speaker_mismatch,
        speaker_similarity=speaker_similarity,
        speaker_enrolled=speaker_enrolled,
        analysis_duration_s=duration,
    )
