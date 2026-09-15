"""
VoiceShield AI — Attack Classifier
Classifies the type of voice attack beyond binary fake/real:
TTS synthesized, voice conversion (VC), replay attack, or genuine.

Attack taxonomy:
1. TTS (Text-to-Speech): Synthesized from scratch — WaveNet, VITS, FastSpeech
2. Voice Conversion (VC): Voice style transfer — RVC, VALL-E, FreeVC
3. Replay: Pre-recorded genuine voice replayed
4. Unknown Synthetic: Novel attack not fitting known categories
5. Genuine: Natural human speech

Each attack class has distinctive acoustic fingerprints.
"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import Dict


@dataclass
class AttackClassificationResult:
    predicted_class: str        # TTS_SYNTHESIZED, VOICE_CONVERSION, REPLAY_ATTACK, GENUINE, UNKNOWN_SYNTHETIC
    class_probabilities: Dict[str, float]
    confidence: float
    replay_probability: float
    voice_conversion_probability: float
    unknown_attack_score: float
    explanation: str


class AttackClassifier:
    """
    Multi-class voice attack classifier.

    Feature signatures per attack class:
    - TTS: very smooth spectral envelope, periodic harmonics, no breath noise
    - Voice Conversion: speaker-dependent artifacts, formant inconsistencies
    - Replay: compression artifacts, double recording noise floor, room reverb
    - Genuine: natural jitter, shimmer, aspiration, co-articulation noise
    - Unknown: synthetic but doesn't match TTS/VC patterns

    Detection approach:
    Feature-based heuristic classifier (production: replace with
    AASIST or Wav2Vec2 fine-tuned for attack type classification)
    """

    ATTACK_CLASSES = ["GENUINE", "TTS_SYNTHESIZED", "VOICE_CONVERSION", "REPLAY_ATTACK", "UNKNOWN_SYNTHETIC"]

    def classify(
        self,
        waveform: np.ndarray,
        sr: int,
        deepfake_probability: float,
        prosody_anomaly: float,
    ) -> AttackClassificationResult:
        """Classify the type of voice manipulation."""

        features = self._extract_attack_features(waveform, sr)

        # Score each attack class
        tts_score = self._score_tts(features, deepfake_probability, prosody_anomaly)
        vc_score = self._score_voice_conversion(features, deepfake_probability)
        replay_score = self._score_replay(features, waveform, sr)
        unknown_score = self._score_unknown(features, deepfake_probability, tts_score, vc_score, replay_score)
        genuine_score = max(0.0, 1.0 - max(tts_score, vc_score, replay_score, unknown_score))

        raw_scores = {
            "GENUINE": genuine_score,
            "TTS_SYNTHESIZED": tts_score,
            "VOICE_CONVERSION": vc_score,
            "REPLAY_ATTACK": replay_score,
            "UNKNOWN_SYNTHETIC": unknown_score,
        }

        # Softmax normalization
        total = sum(raw_scores.values()) + 1e-8
        probabilities = {k: float(v / total) for k, v in raw_scores.items()}

        predicted = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[predicted])

        return AttackClassificationResult(
            predicted_class=predicted,
            class_probabilities=probabilities,
            confidence=confidence,
            replay_probability=float(probabilities["REPLAY_ATTACK"]),
            voice_conversion_probability=float(probabilities["VOICE_CONVERSION"]),
            unknown_attack_score=float(probabilities["UNKNOWN_SYNTHETIC"]),
            explanation=self._explain(predicted, probabilities, confidence),
        )

    def _extract_attack_features(self, waveform: np.ndarray, sr: int) -> dict:
        """Extract attack-discriminative features."""
        # Noise floor estimation (silent regions)
        rms = librosa.feature.rms(y=waveform, frame_length=512, hop_length=160)[0]
        noise_floor = float(np.percentile(rms, 5))

        # Spectral flatness (TTS indicator)
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=waveform)[0]))

        # Spectral contrast (replay indicator — room reverb changes contrast)
        contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr, n_bands=6)
        contrast_mean = float(np.mean(contrast))
        contrast_std = float(np.std(contrast))

        # Formant estimation via LPC (voice conversion indicator)
        # Simplified: spectral envelope smoothness
        fft = np.abs(np.fft.rfft(waveform[:min(len(waveform), 4096)], n=4096))
        envelope_smoothness = float(np.std(np.diff(fft[:200])))

        # Double recording artifact: elevated and irregular noise floor
        noise_variability = float(np.std(rms[:20])) if len(rms) > 20 else 0.0

        # Cepstral peak prominence
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        mfcc_skewness = float(np.mean(np.abs(np.diff(mfcc[0]))))

        return {
            "noise_floor": noise_floor,
            "noise_variability": noise_variability,
            "spectral_flatness": flatness,
            "spectral_contrast_mean": contrast_mean,
            "spectral_contrast_std": contrast_std,
            "envelope_smoothness": envelope_smoothness,
            "mfcc_skewness": mfcc_skewness,
        }

    def _score_tts(self, features: dict, deepfake_prob: float, prosody_anomaly: float) -> float:
        """TTS signature: smooth spectrum, low flatness variation, clean noise floor."""
        score = 0.0
        score += deepfake_prob * 0.4
        score += prosody_anomaly * 0.3

        # Very low noise floor = clean studio-style TTS generation
        if features["noise_floor"] < 0.005:
            score += 0.2

        # High spectral flatness = TTS vocoder
        if features["spectral_flatness"] > 0.12:
            score += 0.1

        return float(np.clip(score, 0, 1))

    def _score_voice_conversion(self, features: dict, deepfake_prob: float) -> float:
        """VC signature: formant inconsistencies, spectral envelope artifacts."""
        score = 0.0
        score += deepfake_prob * 0.35

        # High envelope smoothness deviation = formant conversion artifact
        if features["envelope_smoothness"] > 50:
            score += 0.25

        # Moderate contrast variation (not as extreme as replay, not as flat as TTS)
        if 5 < features["spectral_contrast_std"] < 20:
            score += 0.15

        if features["mfcc_skewness"] > 15:
            score += 0.15

        return float(np.clip(score, 0, 1))

    def _score_replay(self, features: dict, waveform: np.ndarray, sr: int) -> float:
        """Replay signature: double-recorded noise, room reverb, compression artifacts."""
        score = 0.0

        # Elevated and variable noise floor (device → air → microphone)
        if features["noise_floor"] > 0.01:
            score += 0.3
        if features["noise_variability"] > 0.005:
            score += 0.2

        # Reverberation detection via tail energy
        reverb_score = self._detect_reverb(waveform, sr)
        score += reverb_score * 0.3

        # Low spectral contrast variation (flattened by replay)
        if features["spectral_contrast_std"] < 3:
            score += 0.2

        return float(np.clip(score, 0, 1))

    def _detect_reverb(self, waveform: np.ndarray, sr: int) -> float:
        """Estimate reverb energy ratio (C50 clearness metric approximation)."""
        # Early energy (0-50ms) vs total energy
        early_samples = int(0.05 * sr)
        total_energy = float(np.sum(waveform**2))
        if total_energy < 1e-8:
            return 0.0
        early_energy = float(np.sum(waveform[:early_samples]**2))
        late_energy = total_energy - early_energy

        # High late-energy ratio suggests reverb (replay characteristic)
        reverb_ratio = late_energy / (total_energy + 1e-8)
        return float(np.clip(reverb_ratio * 2, 0, 1))

    def _score_unknown(self, features, deepfake_prob, tts, vc, replay) -> float:
        """Unknown/unseen attack: synthetic but doesn't fit known categories well."""
        if deepfake_prob > 0.5:
            known_max = max(tts, vc, replay)
            if known_max < 0.4:
                # Synthetic but doesn't match known attack patterns
                return float(np.clip(deepfake_prob - known_max * 0.5, 0, 1))
        return 0.1

    def _explain(self, predicted: str, probs: dict, confidence: float) -> str:
        explanations = {
            "TTS_SYNTHESIZED": "Audio exhibits neural TTS synthesis signatures: smooth spectral envelope, clean noise floor, regular prosody. Likely generated by WaveNet, VITS, or similar TTS system.",
            "VOICE_CONVERSION": "Audio shows voice conversion artifacts: formant inconsistencies, spectral envelope deformation. Possible RVC/VALL-E style voice style transfer.",
            "REPLAY_ATTACK": "Audio shows replay attack indicators: elevated noise floor, potential reverb signature, compression artifacts suggesting pre-recorded playback.",
            "UNKNOWN_SYNTHETIC": "Audio appears synthetic but does not match known attack fingerprints. Possible novel or unseen voice cloning model.",
            "GENUINE": "No known attack signatures detected. Audio characteristics consistent with natural live speech.",
        }
        base = explanations.get(predicted, "Unknown classification.")
        return f"{base} (Confidence: {confidence:.0%})"


classifier = AttackClassifier()
