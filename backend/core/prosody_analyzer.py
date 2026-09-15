"""
VoiceShield AI — Prosody & Behavioral Analyzer
Analyzes speech rhythm, pitch contours, pauses, speech rate,
and micro-variations to differentiate natural human speech from
neural TTS outputs and voice-converted speech.

Key insight: TTS systems optimize for intelligibility/naturalness at utterance level
but fail to reproduce the micro-variations inherent in human speech production.
"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ProsodyResult:
    anomaly_probability: float    # 0=natural, 1=synthetic
    pitch_regularity: float       # 0=natural variation, 1=too regular (TTS)
    speech_rate_anomaly: float    # 0=natural, 1=unnatural rate
    pause_pattern_anomaly: float  # 0=natural pauses, 1=unnatural
    energy_contour_anomaly: float # 0=natural, 1=flat (TTS characteristic)
    jitter: float                 # F0 micro-variation (Hz)
    shimmer: float                # Amplitude micro-variation
    speaking_rate_wpm: float      # Estimated words per minute
    explanation: str


class ProsodyAnalyzer:
    """
    Prosodic feature analysis for voice cloning detection.

    Natural human speech characteristics:
    - F0 jitter: 0.2–2% (measured as RAP - Relative Average Perturbation)
    - Amplitude shimmer: 0.5–3 dB
    - Dynamic pause patterns (Poissonian distribution)
    - Non-monotonic energy contours
    - Natural co-articulation effects

    TTS/cloned voice characteristics:
    - F0 too regular (jitter < 0.1%)
    - Amplitude shimmer too uniform
    - Missing breath pauses or artificial pause placement
    - Unnaturally flat energy in certain frequency bands
    """

    # Normal speech ranges
    NATURAL_JITTER_MIN = 0.5    # Hz
    NATURAL_JITTER_MAX = 5.0    # Hz
    NATURAL_SHIMMER_MIN = 0.5   # dB
    NATURAL_RATE_MIN = 100      # wpm
    NATURAL_RATE_MAX = 200      # wpm

    def analyze(self, waveform: np.ndarray, sr: int) -> ProsodyResult:
        """Full prosodic analysis pipeline."""

        # --- Pitch (F0) analysis ---
        f0, voiced_flag, _ = librosa.pyin(
            waveform, fmin=60, fmax=400, sr=sr,
            frame_length=512, hop_length=160
        )
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)]

        # --- Amplitude envelope ---
        rms = librosa.feature.rms(y=waveform, frame_length=512, hop_length=160)[0]

        # --- Pause analysis ---
        pause_score = self._analyze_pauses(rms, sr)

        # --- Pitch regularity ---
        pitch_score = self._analyze_pitch_regularity(f0_voiced)

        # --- Jitter (F0 micro-variation) ---
        jitter = self._compute_jitter(f0_voiced)

        # --- Shimmer (amplitude micro-variation) ---
        shimmer = self._compute_shimmer(rms)

        # --- Energy contour ---
        energy_score = self._analyze_energy_contour(rms)

        # --- Speech rate estimation ---
        speaking_rate = self._estimate_speaking_rate(rms, sr)
        rate_score = self._analyze_speech_rate(speaking_rate)

        # Weighted ensemble
        scores = [pitch_score, pause_score, energy_score, rate_score]
        weights = [0.35, 0.25, 0.25, 0.15]
        anomaly_prob = float(np.average(scores, weights=weights))
        anomaly_prob = float(np.clip(anomaly_prob, 0, 1))

        # Jitter anomaly contribution (too low = TTS)
        if jitter < self.NATURAL_JITTER_MIN:
            anomaly_prob = min(1.0, anomaly_prob + 0.15)

        explanation = self._explain(anomaly_prob, jitter, shimmer, speaking_rate, pitch_score)

        return ProsodyResult(
            anomaly_probability=anomaly_prob,
            pitch_regularity=pitch_score,
            speech_rate_anomaly=rate_score,
            pause_pattern_anomaly=pause_score,
            energy_contour_anomaly=energy_score,
            jitter=float(jitter),
            shimmer=float(shimmer),
            speaking_rate_wpm=float(speaking_rate),
            explanation=explanation,
        )

    def _compute_jitter(self, f0_voiced: np.ndarray) -> float:
        """
        Compute F0 jitter (RAP - Relative Average Perturbation).
        Natural speech: 0.5–5 Hz jitter.
        TTS: < 0.3 Hz (too regular).
        """
        if len(f0_voiced) < 3:
            return 2.0  # assume natural if insufficient data
        diffs = np.abs(np.diff(f0_voiced))
        jitter = float(np.mean(diffs))
        return jitter

    def _compute_shimmer(self, rms: np.ndarray) -> float:
        """
        Compute amplitude shimmer (frame-to-frame energy variation).
        Natural speech: 0.5–3 dB shimmer.
        TTS: unnaturally flat amplitude.
        """
        if len(rms) < 3:
            return 1.0
        rms_db = librosa.amplitude_to_db(rms + 1e-8)
        shimmer = float(np.mean(np.abs(np.diff(rms_db))))
        return shimmer

    def _analyze_pitch_regularity(self, f0_voiced: np.ndarray) -> float:
        """
        TTS systems produce unnaturally monotone/regular pitch.
        Score → 1 means too regular (synthetic).
        """
        if len(f0_voiced) < 5:
            return 0.4

        f0_std = np.std(f0_voiced)
        f0_range = np.ptp(f0_voiced)

        # Natural speech: std > 20 Hz, range > 50 Hz
        regularity = 1.0 - min(f0_std / 40.0, 1.0)
        return float(np.clip(regularity, 0, 1))

    def _analyze_pauses(self, rms: np.ndarray, sr: int) -> float:
        """
        Natural speech has irregular pause patterns.
        TTS pauses are too uniform or missing co-articulation disfluencies.
        """
        rms_db = librosa.amplitude_to_db(rms + 1e-8)
        silence_mask = rms_db < -40  # silence frames

        if not np.any(silence_mask):
            # No pauses at all — suspicious for cloned speech (may be too fluent)
            return 0.4

        # Count pause segments
        transitions = np.diff(silence_mask.astype(int))
        pause_starts = np.where(transitions == 1)[0]
        pause_ends = np.where(transitions == -1)[0]

        if len(pause_starts) < 2:
            return 0.3

        # Pause durations
        min_len = min(len(pause_starts), len(pause_ends))
        pause_durations = pause_ends[:min_len] - pause_starts[:min_len]

        # Natural: high variance in pause durations (Poissonian)
        pause_cv = np.std(pause_durations) / (np.mean(pause_durations) + 1e-8)

        # Low coefficient of variation → too regular → suspicious
        regularity_score = float(np.clip(1.0 - min(pause_cv, 1.0), 0, 1))
        return regularity_score

    def _analyze_energy_contour(self, rms: np.ndarray) -> float:
        """
        TTS systems produce unnaturally flat energy contours.
        Natural speech has dynamic, non-monotonic energy variation.
        """
        rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
        energy_var = float(np.var(rms_smooth))

        # Low energy variance → flat contour → synthetic
        flatness_score = float(np.clip(1.0 - min(energy_var * 50, 1.0), 0, 1))
        return flatness_score

    def _estimate_speaking_rate(self, rms: np.ndarray, sr: int) -> float:
        """Estimate speaking rate in words per minute (using syllable nuclei detection)."""
        hop_length = 160
        frame_duration = hop_length / sr  # seconds per frame

        # Find energy peaks (syllable nuclei approximation)
        rms_smooth = np.convolve(rms, np.ones(10)/10, mode='same')
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(rms_smooth, height=np.max(rms_smooth) * 0.3, distance=10)

        total_duration = len(rms) * frame_duration
        if total_duration < 0.5:
            return 140.0  # default

        syllables_per_second = len(peaks) / total_duration
        # Average: ~1.5 syllables per word
        wpm = syllables_per_second * 60 / 1.5
        return float(np.clip(wpm, 50, 400))

    def _analyze_speech_rate(self, wpm: float) -> float:
        """Detect unnatural speech rate (too fast TTS or unnaturally slow)."""
        if self.NATURAL_RATE_MIN <= wpm <= self.NATURAL_RATE_MAX:
            # Normal range
            return 0.1
        elif wpm < 80 or wpm > 220:
            # Very abnormal
            return 0.7
        else:
            return 0.35

    def _explain(self, anomaly_prob, jitter, shimmer, wpm, pitch_score) -> str:
        parts = []
        if pitch_score > 0.6:
            parts.append(f"unusually regular pitch contour (F0 jitter: {jitter:.1f} Hz)")
        if jitter < self.NATURAL_JITTER_MIN:
            parts.append("abnormally low F0 micro-variation (TTS signature)")
        if shimmer < 0.5:
            parts.append("unnaturally flat amplitude envelope")
        if wpm > 200 or wpm < 80:
            parts.append(f"abnormal speaking rate ({wpm:.0f} wpm)")

        if anomaly_prob > 0.65:
            base = "Prosodic features strongly indicate synthetic/cloned speech."
        elif anomaly_prob > 0.4:
            base = "Moderate prosodic anomalies detected."
        else:
            base = "Prosodic patterns consistent with natural human speech."

        if parts:
            return f"{base} Evidence: {'; '.join(parts)}."
        return base


analyzer = ProsodyAnalyzer()
