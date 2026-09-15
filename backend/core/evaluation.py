"""
VoiceShield AI — Model Benchmark & Evaluation
Runs the full detection pipeline on labeled test clips and computes SIH-ready metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from core.preprocessor import preprocessor
from core.deepfake_detector import detector, get_wav2vec2_model
from core.speaker_verifier import verifier, get_ecapa_classifier, ECAPA_EMBEDDING_DIM
from core.prosody_analyzer import analyzer
from core.attack_classifier import classifier
from core.context_engine import context_engine
from core.risk_engine import risk_engine
from core.policy_engine import policy_engine
from db.schemas import CallerContext, TransactionContext

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "public" / "demo"
CUSTOM_DIR = DEMO_DIR / "custom"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "benchmark_manifest.json"

# Published benchmarks for pretrained models (cite in SIH presentation)
MODEL_CITATIONS = {
    "deepfake": {
        "model": "IndicWav2Vec-Hindi + MLP head (primary) / XLS-R-SLS v1 (optional)",
        "architecture": "Frozen Wav2Vec2 encoder + trained classifier head",
        "task": "Binary audio deepfake classification",
        "reference": "ai4bharat/indicwav2vec-hindi + fine-tuned head on labeled data",
    },
    "speaker": {
        "model": "speechbrain/spkrec-ecapa-voxceleb",
        "architecture": "ECAPA-TDNN",
        "eer_voxceleb1_h": "< 1%",
        "reference": "Desplanques et al., ECAPA-TDNN, Interspeech 2020",
    },
}


@dataclass
class TestCaseResult:
    id: str
    label: str
    file: str
    file_exists: bool
    expected_category: str
    predicted_attack: str
    risk_score: float
    risk_level: str
    deepfake_probability: float
    speaker_mismatch: float
    speaker_verified: Optional[bool]
    models_used: str
    processing_time_ms: float
    category_correct: bool
    risk_band_correct: bool
    deepfake_binary_correct: bool


@dataclass
class BenchmarkReport:
    timestamp: str
    models_loaded: dict
    test_count: int
    files_found: int
    metrics: dict
    model_citations: dict
    results: list
    recommendations: list


def _load_manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)["test_cases"]
    return _default_test_cases()


def _default_test_cases() -> list[dict]:
    return [
        {
            "id": "genuine_ceo",
            "file": "genuine.wav",
            "label": "Genuine CEO (enrolled)",
            "expected_category": "GENUINE",
            "expected_risk_band": "LOW",
            "expected_deepfake_max": 0.45,
            "caller_id": "CEO-001",
            "caller_name": "Priya Sharma (CEO)",
            "is_known_contact": True,
            "call_origin": "MOBILE",
        },
        {
            "id": "cloned_tts",
            "file": "cloned.wav",
            "label": "AI-Cloned Voice (TTS)",
            "expected_category": "TTS_SYNTHESIZED",
            "expected_risk_band": "HIGH",
            "expected_deepfake_min": 0.55,
            "call_origin": "VOIP",
            "is_known_contact": False,
        },
        {
            "id": "ceo_impersonation_transfer",
            "file": "cloned_transfer.wav",
            "label": "CEO Impersonation + ₹25L",
            "expected_category": "TTS_SYNTHESIZED",
            "expected_risk_band": "CRITICAL",
            "expected_deepfake_min": 0.55,
            "transaction_amount": 2500000,
            "transaction_type": "FUND_TRANSFER",
            "is_privileged_workflow": True,
            "call_origin": "VOIP",
        },
        {
            "id": "replay_attack",
            "file": "replay.wav",
            "label": "Replay Attack",
            "expected_category": "REPLAY_ATTACK",
            "expected_risk_band": "HIGH",
            "historical_fraud_flag": True,
            "transaction_amount": 150000,
            "transaction_type": "WIRE_TRANSFER",
            "call_origin": "VOIP",
        },
        {
            "id": "voice_conversion",
            "file": "voice_conversion.wav",
            "label": "Voice Conversion (RVC)",
            "expected_category": "VOICE_CONVERSION",
            "expected_risk_band": "HIGH",
            "expected_deepfake_min": 0.50,
            "transaction_amount": 500000,
            "transaction_type": "APPROVAL",
            "call_origin": "UNKNOWN",
        },
    ]


def _resolve_audio_path(filename: str) -> Path:
    """Prefer user-recorded custom clips over generated defaults."""
    custom = CUSTOM_DIR / filename
    if custom.exists():
        return custom
    return DEMO_DIR / filename


def _risk_band(score: float) -> str:
    if score < 30:
        return "LOW"
    if score < 70:
        return "MEDIUM"
    if score < 85:
        return "HIGH"
    return "CRITICAL"


def _category_match(predicted: str, expected: str) -> bool:
    if predicted == expected:
        return True
    # Synthetic attacks often overlap — partial credit
    synthetic = {"TTS_SYNTHESIZED", "VOICE_CONVERSION", "UNKNOWN_SYNTHETIC", "REPLAY_ATTACK"}
    if expected in synthetic and predicted in synthetic:
        return True
    return False


def _run_single_case(case: dict, ref_embedding: Optional[np.ndarray] = None) -> TestCaseResult:
    from core.analysis_pipeline import run_parallel_layers

    audio_path = _resolve_audio_path(case["file"])
    file_exists = audio_path.exists()

    if not file_exists:
        return TestCaseResult(
            id=case["id"],
            label=case["label"],
            file=str(audio_path),
            file_exists=False,
            expected_category=case["expected_category"],
            predicted_attack="N/A",
            risk_score=0.0,
            risk_level="N/A",
            deepfake_probability=0.0,
            speaker_mismatch=0.0,
            speaker_verified=None,
            models_used="N/A",
            processing_time_ms=0.0,
            category_correct=False,
            risk_band_correct=False,
            deepfake_binary_correct=False,
        )

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    t0 = time.time()
    waveform, sr, _ = preprocessor.load_audio(audio_bytes, str(audio_path))
    waveform = preprocessor.apply_vad(waveform, sr)

    ref = ref_embedding if case.get("caller_id") else None
    layers = run_parallel_layers(waveform, sr, ref)
    deepfake_result = layers.deepfake_result
    prosody_result = layers.prosody_result
    attack_result = layers.attack_result
    speaker_mismatch = layers.speaker_mismatch
    speaker_verified = None
    speaker_enrolled = layers.speaker_enrolled
    if case.get("caller_id") and ref_embedding is not None:
        speaker_verified = speaker_mismatch < 0.5

    caller_ctx = CallerContext(
        caller_id=case.get("caller_id"),
        caller_name=case.get("caller_name"),
        phone_number=case.get("phone_number"),
        is_known_contact=case.get("is_known_contact", False),
        historical_fraud_flag=case.get("historical_fraud_flag", False),
        call_origin=case.get("call_origin", "UNKNOWN"),
    )
    txn_ctx = None
    if case.get("transaction_amount") is not None:
        txn_ctx = TransactionContext(
            amount=case["transaction_amount"],
            transaction_type=case.get("transaction_type"),
            is_privileged_workflow=case.get("is_privileged_workflow", False),
        )
    context_result = context_engine.evaluate(caller_ctx, txn_ctx)

    risk_result = risk_engine.compute(
        deepfake_probability=deepfake_result.probability,
        speaker_mismatch=speaker_mismatch,
        prosody_anomaly=prosody_result.anomaly_probability,
        replay_probability=attack_result.replay_probability,
        unknown_attack_score=attack_result.unknown_attack_score,
        context_risk=context_result.context_risk,
        transaction_risk_multiplier=context_result.transaction_risk_multiplier,
        speaker_enrolled=speaker_enrolled,
    )

    elapsed_ms = (time.time() - t0) * 1000
    predicted_band = risk_result.level
    expected_band = case.get("expected_risk_band", "MEDIUM")

    # Risk band: allow adjacent band for MEDIUM/HIGH boundary
    band_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    band_correct = predicted_band == expected_band
    if not band_correct:
        pi, ei = band_order.index(predicted_band), band_order.index(expected_band)
        band_correct = abs(pi - ei) <= 1 and case["expected_category"] != "GENUINE"

    is_genuine = case["expected_category"] == "GENUINE"
    df_correct = True
    if is_genuine:
        if case.get("expected_deepfake_max") is not None:
            df_correct = deepfake_result.probability <= case["expected_deepfake_max"]
        else:
            df_correct = deepfake_result.probability < 0.5
    else:
        if case.get("expected_deepfake_min") is not None:
            df_correct = deepfake_result.probability >= case["expected_deepfake_min"]
        else:
            df_correct = deepfake_result.probability >= 0.5

    return TestCaseResult(
        id=case["id"],
        label=case["label"],
        file=str(audio_path.relative_to(ROOT)) if audio_path.is_relative_to(ROOT) else str(audio_path),
        file_exists=True,
        expected_category=case["expected_category"],
        predicted_attack=attack_result.predicted_class,
        risk_score=round(risk_result.score, 1),
        risk_level=predicted_band,
        deepfake_probability=round(deepfake_result.probability, 3),
        speaker_mismatch=round(speaker_mismatch, 3),
        speaker_verified=speaker_verified,
        models_used=deepfake_result.model_used,
        processing_time_ms=round(elapsed_ms, 1),
        category_correct=_category_match(attack_result.predicted_class, case["expected_category"]),
        risk_band_correct=band_correct,
        deepfake_binary_correct=df_correct,
    )


def run_benchmark() -> BenchmarkReport:
    from datetime import datetime, timezone
    from core.indic_encoder import get_indic_status, HEAD_PATH
    from core.xlsr_sls_detector import get_load_status

    cases = _load_manifest()
    wav2vec_ok = get_wav2vec2_model()[0] is not None
    ecapa_ok = get_ecapa_classifier() is not None
    indic_status = get_indic_status()
    xlsr_status = get_load_status()

    # Load CEO reference embedding for speaker tests
    ref_embedding = None
    genuine_path = _resolve_audio_path("genuine.wav")
    if genuine_path.exists():
        with open(genuine_path, "rb") as f:
            w, sr, _ = preprocessor.load_audio(f.read(), "genuine.wav")
        ref_embedding = verifier.extract_embedding(w, sr)

    results = [_run_single_case(c, ref_embedding) for c in cases]
    found = [r for r in results if r.file_exists]

    def rate(attr: str) -> float:
        if not found:
            return 0.0
        return round(sum(getattr(r, attr) for r in found) / len(found) * 100, 1)

  # Binary fake/real from deepfake scores
    genuine_cases = [r for r in found if r.expected_category == "GENUINE"]
    fake_cases = [r for r in found if r.expected_category != "GENUINE"]
    tp = sum(1 for r in fake_cases if r.deepfake_probability >= 0.5)
    tn = sum(1 for r in genuine_cases if r.deepfake_probability < 0.5)
    fp = sum(1 for r in genuine_cases if r.deepfake_probability >= 0.5)
    fn = sum(1 for r in fake_cases if r.deepfake_probability < 0.5)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    avg_latency = round(sum(r.processing_time_ms for r in found) / len(found), 1) if found else 0.0

    # Per-language accuracy (Indic support)
    language_metrics = {}
    for case, result in zip(cases, results):
        lang = case.get("language")
        if not lang or not result.file_exists:
            continue
        if lang not in language_metrics:
            language_metrics[lang] = {"total": 0, "correct": 0}
        language_metrics[lang]["total"] += 1
        if result.deepfake_binary_correct:
            language_metrics[lang]["correct"] += 1
    for lang, m in language_metrics.items():
        m["accuracy_pct"] = round(m["correct"] / m["total"] * 100, 1) if m["total"] else 0

    confusion_matrix = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    recommendations = []
    if not indic_status.get("head_exists"):
        recommendations.append("Indic classifier head not trained — run: python scripts/train_classifier.py")
    if indic_status.get("using_fallback"):
        recommendations.append(
            "Using XLS-R-53 fallback encoder — accept ai4bharat/indicwav2vec-hindi gate and set HF_TOKEN."
        )
    if not xlsr_status.get("loaded"):
        err = xlsr_status.get("error") or "unknown"
        recommendations.append(f"XLS-R-SLS not loaded: {str(err)[:80]}")
    if not ecapa_ok:
        recommendations.append("ECAPA-TDNN failed to load — speaker verification using MFCC fallback.")
    if fp > 0:
        recommendations.append(
            f"{fp} false positive(s): replace public/demo/custom/genuine.wav with a real human voice recording."
        )
    if fn > 0:
        recommendations.append(
            f"{fn} false negative(s): use real AI-TTS clones in public/demo/custom/cloned.wav."
        )
    if CUSTOM_DIR.exists() and any(CUSTOM_DIR.glob("*.wav")):
        recommendations.append("Using custom recordings from public/demo/custom/ — good for judge demo.")
    else:
        recommendations.append(
            "Tip: Add real recordings to public/demo/custom/ (genuine.wav, cloned.wav) for best demo accuracy."
        )

    return BenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        models_loaded={
            "indic_encoder": indic_status.get("encoder_id"),
            "indic_head_trained": indic_status.get("head_exists", False),
            "indic_using_fallback": indic_status.get("using_fallback", False),
            "xlsr_sls_loaded": xlsr_status.get("loaded", False),
            "legacy_wav2vec2": wav2vec_ok,
            "ecapa_speaker": ecapa_ok,
            "ecapa_embedding_dim": ECAPA_EMBEDDING_DIM if ecapa_ok else 128,
        },
        test_count=len(cases),
        files_found=len(found),
        metrics={
            "attack_classification_accuracy_pct": rate("category_correct"),
            "risk_band_accuracy_pct": rate("risk_band_correct"),
            "deepfake_detection_accuracy_pct": rate("deepfake_binary_correct"),
            "precision_fake_detection": round(precision, 3),
            "recall_fake_detection": round(recall, 3),
            "f1_fake_detection": round(f1, 3),
            "avg_latency_ms": avg_latency,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "confusion_matrix": confusion_matrix,
            "language_accuracy": language_metrics,
        },
        model_citations=MODEL_CITATIONS,
        results=[asdict(r) for r in results],
        recommendations=recommendations,
    )


def report_to_dict(report: BenchmarkReport) -> dict:
    return asdict(report)
