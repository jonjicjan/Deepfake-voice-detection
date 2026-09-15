"""
VoiceShield AI — Deepfake Detector
Pipeline priority:
  1. IndicWav2Vec frozen encoder + trained classifier head (Indian languages)
  2. XLS-R-SLS pretrained checkpoint (English/cross-domain, requires fairseq)
  3. Legacy Wav2Vec2 / DSP fallback
"""

import numpy as np
import librosa
import torch
from dataclasses import dataclass
import logging

from core.ml_config import load_calibration
from core.inference_utils import (
    get_device,
    prepare_model,
    to_model_dtype,
    get_onnx_session,
    run_onnx_logits,
    use_fp16,
)

logger = logging.getLogger("voiceshield.deepfake_detector")

LEGACY_MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"

_WAV2VEC2_MODEL = None
_WAV2VEC2_FE = None
_WAV2VEC2_DEVICE = None
_WAV2VEC2_FAILED = False
_RUNTIME_INFO = {"backend": "none", "device": "cpu", "fp16": False, "compiled": False}


def get_runtime_info() -> dict:
    return dict(_RUNTIME_INFO)


def get_wav2vec2_model():
    global _WAV2VEC2_MODEL, _WAV2VEC2_FE, _WAV2VEC2_DEVICE, _WAV2VEC2_FAILED, _RUNTIME_INFO
    if _WAV2VEC2_FAILED:
        return None, None, None

    onnx_sess, onnx_in, onnx_out = get_onnx_session()
    if onnx_sess is not None:
        _RUNTIME_INFO.update({"backend": "onnx", "device": "cuda" if torch.cuda.is_available() else "cpu"})
        return "onnx", onnx_sess, (onnx_in, onnx_out)

    if _WAV2VEC2_MODEL is None:
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            print(f"[INFO] Loading legacy Wav2Vec2 model: {LEGACY_MODEL_ID}...")
            _WAV2VEC2_DEVICE = get_device()
            _WAV2VEC2_FE = AutoFeatureExtractor.from_pretrained(LEGACY_MODEL_ID)
            model = AutoModelForAudioClassification.from_pretrained(LEGACY_MODEL_ID)
            compiled = prepare_model(model, _WAV2VEC2_DEVICE)
            _WAV2VEC2_MODEL = compiled
            _RUNTIME_INFO.update({
                "backend": "pytorch",
                "device": str(_WAV2VEC2_DEVICE),
                "fp16": use_fp16(),
                "compiled": compiled is not model or hasattr(compiled, "_orig_mod"),
            })
            print(f"[OK] Wav2Vec2 loaded ({_RUNTIME_INFO['device']}, fp16={_RUNTIME_INFO['fp16']})")
        except Exception as e:
            logger.warning(f"Wav2Vec2 load failed ({e}). DSP fallback.")
            _WAV2VEC2_FAILED = True
            return None, None, None
    return _WAV2VEC2_MODEL, _WAV2VEC2_FE, _WAV2VEC2_DEVICE


def _fake_prob_from_logits(model_or_cfg, logits) -> float:
    if isinstance(logits, np.ndarray):
        t = torch.from_numpy(logits).float()
    elif isinstance(logits, torch.Tensor):
        t = logits.float()
    else:
        t = torch.tensor(logits).float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    probs = torch.softmax(t, dim=-1)[0]

    if isinstance(model_or_cfg, str) and model_or_cfg == "onnx":
        id2label = {0: "fake", 1: "real"}
    else:
        id2label = model_or_cfg.config.id2label

    fake_prob = 0.0
    for idx, prob in enumerate(probs.tolist()):
        label = str(id2label.get(idx, id2label.get(str(idx), ""))).lower()
        if "fake" in label or "synthetic" in label or "spoof" in label:
            fake_prob = max(fake_prob, prob)
        elif "real" in label or "genuine" in label:
            fake_prob = max(fake_prob, 1.0 - prob)
    return float(fake_prob)


_ONNX_FE = None


def _get_onnx_feature_extractor():
    global _ONNX_FE
    if _ONNX_FE is None:
        from transformers import AutoFeatureExtractor
        _ONNX_FE = AutoFeatureExtractor.from_pretrained(LEGACY_MODEL_ID)
    return _ONNX_FE


def _run_neural_inference(waveform: np.ndarray, sr: int) -> tuple[float | None, str]:
    """Try detectors in priority order. Returns (fake_prob, model_name)."""
    # 1. IndicWav2Vec + trained head
    try:
        from core.indic_encoder import predict_fake_probability as indic_predict, get_indic_status
        status = get_indic_status()
        if status["head_exists"]:
            prob = indic_predict(waveform, sr)
            if prob is not None:
                return prob, "IndicWav2Vec+MLP"
    except Exception as e:
        logger.debug(f"Indic classifier skipped: {e}")

    # 2. XLS-R-SLS
    try:
        from core.xlsr_sls_detector import predict_fake_probability as xlsr_predict, load_xlsr_sls_model, get_load_status
        load_xlsr_sls_model()
        if get_load_status()["loaded"]:
            prob = xlsr_predict(waveform, sr)
            if prob is not None:
                return prob, "XLS-R-SLS"
    except Exception as e:
        logger.debug(f"XLS-R-SLS skipped: {e}")

    # 3. Legacy Wav2Vec2 / ONNX
    prob = _run_wav2vec2_inference(waveform, sr)
    if prob is not None:
        backend = _RUNTIME_INFO.get("backend", "pytorch")
        return prob, f"Legacy-Wav2Vec2 ({backend})"
    return None, "DSP-Engine"


def _run_wav2vec2_inference(waveform: np.ndarray, sr: int) -> float | None:
    loaded = get_wav2vec2_model()
    if loaded[0] is None:
        return None

    model, fe_or_sess, device_or_meta = loaded

    if model == "onnx":
        session = fe_or_sess
        input_names, output_name = device_or_meta
        fe = _get_onnx_feature_extractor()
        inputs = fe(waveform, sampling_rate=sr, return_tensors="np", padding=True)
        feed = {k: v.astype(np.float32) for k, v in inputs.items()}
        logits = run_onnx_logits(session, input_names, output_name, feed)
        return _fake_prob_from_logits("onnx", logits[0])

    inputs = fe_or_sess(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: to_model_dtype(v.to(device_or_meta), model) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
    return _fake_prob_from_logits(model, logits[0])


@dataclass
class DeepfakeResult:
    probability: float
    confidence: float
    artifacts_detected: list
    spectral_consistency: float
    phase_coherence: float
    label: str
    explanation: str
    model_used: str
    neural_probability: float | None = None
    dsp_skipped: bool = False


class DeepfakeDetector:
    """Wav2Vec2/ONNX + optional lightweight DSP ensemble."""

    def analyze(self, waveform: np.ndarray, sr: int, spectral_features: dict) -> DeepfakeResult:
        cfg = load_calibration()
        neural_prob = None
        model_used = "DSP-Engine"
        dsp_skipped = False

        try:
            neural_prob, model_used = _run_neural_inference(waveform, sr)
        except Exception as e:
            logger.error(f"Neural inference error: {e}")
            neural_prob, model_used = None, "DSP-Engine"

        backend = _RUNTIME_INFO.get("backend", "none")
        if neural_prob is not None and "Legacy" in model_used:
            model_used = f"{model_used} + DSP"
        elif neural_prob is not None:
            pass  # keep IndicWav2Vec or XLS-R-SLS name
        else:
            model_used = "DSP-Engine"

        skip_dsp = False
        if neural_prob is not None and cfg.get("skip_dsp_when_neural_confident", True):
            thresh = float(cfg.get("neural_confidence_threshold", 0.85))
            if neural_prob >= thresh or neural_prob <= (1.0 - thresh):
                skip_dsp = True
                dsp_skipped = True

        if skip_dsp:
            dsp_probability = neural_prob
            artifacts = []
            flatness_score = phase_score = 0.0
        else:
            artifacts, dsp_scores, flatness_score, phase_score = self._run_dsp(waveform, sr)
            weights = [0.15, 0.25, 0.20, 0.15, 0.15, 0.10]
            dsp_probability = float(np.average(dsp_scores, weights=weights))

        if neural_prob is not None:
            nw = float(cfg["ensemble"]["neural_weight"])
            dw = float(cfg["ensemble"]["dsp_weight"])
            if skip_dsp:
                final_probability = neural_prob
                model_used = f"Wav2Vec2-Neural ({backend.upper()}) [DSP skipped]"
            else:
                final_probability = float(nw * neural_prob + dw * dsp_probability)
        else:
            final_probability = dsp_probability

        final_probability = float(np.clip(final_probability, 0.0, 1.0))
        threshold = float(cfg.get("indic_classifier_threshold", cfg.get("deepfake_threshold", 0.5)))
        label = "SYNTHETIC" if final_probability > threshold else "GENUINE"
        confidence = float(abs((neural_prob if neural_prob is not None else final_probability) - 0.5) * 2)

        return DeepfakeResult(
            probability=final_probability,
            confidence=confidence,
            artifacts_detected=artifacts,
            spectral_consistency=float(1.0 - flatness_score),
            phase_coherence=float(1.0 - phase_score),
            label=label,
            explanation=self._generate_explanation(final_probability, artifacts, model_used),
            model_used=model_used,
            neural_probability=neural_prob,
            dsp_skipped=dsp_skipped,
        )

    def _run_dsp(self, waveform: np.ndarray, sr: int):
        artifacts = []
        scores = []
        flatness = self._spectral_flatness_analysis(waveform, sr)
        scores.append(flatness)
        if flatness > 0.6:
            artifacts.append("spectral_flatness_anomaly")
        phase = self._phase_coherence_analysis(waveform, sr)
        scores.append(phase)
        if phase > 0.55:
            artifacts.append("phase_discontinuity")
        mfcc = self._mfcc_temporal_analysis(waveform, sr)
        scores.append(mfcc)
        if mfcc > 0.65:
            artifacts.append("mfcc_over_smoothing")
        pitch = self._pitch_naturalness_analysis(waveform, sr)
        scores.append(pitch)
        if pitch > 0.6:
            artifacts.append("unnatural_pitch_contour")
        harmonic = self._harmonic_analysis(waveform, sr)
        scores.append(harmonic)
        if harmonic > 0.7:
            artifacts.append("harmonic_structure_anomaly")
        bandwidth = self._bandwidth_analysis(waveform, sr)
        scores.append(bandwidth)
        if bandwidth > 0.5:
            artifacts.append("bandwidth_clipping")
        return artifacts, scores, flatness, phase

    def _spectral_flatness_analysis(self, waveform, sr):
        flatness = librosa.feature.spectral_flatness(y=waveform)[0]
        return float(np.clip(float(np.mean(flatness)) * 3.0, 0, 1))

    def _phase_coherence_analysis(self, waveform, sr):
        stft = librosa.stft(waveform, n_fft=512, hop_length=160)
        phase = np.unwrap(np.angle(stft), axis=1)
        return float(np.clip(np.std(np.diff(phase, axis=1)) / 3.0, 0, 1))

    def _mfcc_temporal_analysis(self, waveform, sr):
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        delta_var = np.mean(np.var(librosa.feature.delta(mfcc), axis=1))
        return float(np.clip(1.0 - min(delta_var / 20.0, 1.0), 0, 1))

    def _pitch_naturalness_analysis(self, waveform, sr):
        f0, voiced, _ = librosa.pyin(waveform, fmin=62.5, fmax=500, sr=sr, frame_length=640, hop_length=160)
        f0_v = f0[voiced]
        if len(f0_v) < 5:
            return 0.5
        return float(np.clip(1.0 - min(float(np.std(f0_v)) / 40.0, 1.0), 0, 1))

    def _harmonic_analysis(self, waveform, sr):
        harmonic, _ = librosa.effects.hpss(waveform)
        ratio = float(np.sum(harmonic**2) / (np.sum(waveform**2) + 1e-8))
        return float(np.clip(ratio * 1.5 - 0.5, 0, 1))

    def _bandwidth_analysis(self, waveform, sr):
        fft = np.abs(np.fft.rfft(waveform, n=2048))
        freqs = np.fft.rfftfreq(2048, d=1 / sr)
        high = np.mean(fft[(freqs >= 4000) & (freqs <= 8000)] ** 2) if np.any((freqs >= 4000) & (freqs <= 8000)) else 0
        low = np.mean(fft[(freqs >= 300) & (freqs <= 4000)] ** 2) if np.any((freqs >= 300) & (freqs <= 4000)) else 1
        return float(np.clip(1.0 - min(high / (low + 1e-8) * 10, 1.0), 0, 1))

    def _generate_explanation(self, probability, artifacts, model_used):
        if probability > 0.75:
            severity = f"Strong synthetic speech indicators via {model_used}"
        elif probability > 0.5:
            severity = f"Moderate voice manipulation via {model_used}"
        elif probability > 0.3:
            severity = "Slight anomalies; mostly genuine"
        else:
            severity = "No significant synthetic artifacts"
        if artifacts:
            return f"{severity}. Artifacts: {', '.join(artifacts)}."
        return f"{severity}."


def preload_deepfake_model() -> bool:
    get_onnx_session()
    # Preload Indic encoder + head if available
    try:
        from core.indic_encoder import load_encoder, load_classifier_head
        enc, _ = load_encoder()
        head = load_classifier_head()
        if enc is not None and head is not None:
            print("[OK] IndicWav2Vec deepfake classifier loaded")
            return True
    except Exception as e:
        print(f"[INFO] Indic classifier not ready: {e}")
    # Try XLS-R-SLS
    try:
        from core.xlsr_sls_detector import load_xlsr_sls_model, get_load_status
        load_xlsr_sls_model()
        if get_load_status()["loaded"]:
            return True
    except Exception:
        pass
    model, _, _ = get_wav2vec2_model()
    return model is not None


detector = DeepfakeDetector()
