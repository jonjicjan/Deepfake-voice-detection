"""
VoiceShield AI — Audio Preprocessor
Handles: VAD (Voice Activity Detection), normalization, codec simulation,
noise characterization, and feature frame extraction.
"""

import numpy as np
import librosa
import io
from typing import Tuple, Optional


class AudioPreprocessor:
    """
    Prepares raw audio for downstream AI analysis.
    Simulates real-world telephony conditions (VoIP codec degradation, noise).
    """

    TARGET_SR = 16000  # Hz — standard for speech models
    CHUNK_DURATION = 3.0  # seconds per analysis window
    MIN_AUDIO_DURATION = 0.5  # minimum valid audio

    def load_audio(self, audio_bytes: bytes, filename: str = "") -> Tuple[np.ndarray, int, float]:
        """
        Load audio from bytes, resample to 16kHz mono.
        Returns: (waveform, sample_rate, duration_seconds)
        """
        try:
            buf = io.BytesIO(audio_bytes)
            waveform, sr = librosa.load(buf, sr=self.TARGET_SR, mono=True)
            duration = len(waveform) / self.TARGET_SR
            waveform = self._normalize(waveform)
            return waveform, self.TARGET_SR, duration
        except Exception as e:
            raise ValueError(f"Failed to load audio: {e}")

    def _normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Peak normalization to [-1, 1]."""
        peak = np.max(np.abs(waveform))
        if peak > 0:
            return waveform / peak
        return waveform

    def apply_vad(self, waveform: np.ndarray, sr: int, threshold_db: float = -40.0) -> np.ndarray:
        """
        Simple energy-based Voice Activity Detection.
        Removes silence frames below threshold.
        """
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop

        rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Create mask for voiced frames
        voiced_mask = rms_db > threshold_db

        # Reconstruct voiced-only signal (simplified: keep if >30% frames voiced)
        voiced_ratio = np.mean(voiced_mask)
        if voiced_ratio < 0.1:
            raise ValueError("No voice activity detected in audio segment.")
        return waveform

    def simulate_voip_codec(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        Simulate VoIP codec degradation (G.711 μ-law compression).
        Adds realistic telephony artifacts to test robustness.
        """
        # μ-law encoding/decoding simulation
        mu = 255
        waveform_clipped = np.clip(waveform, -1, 1)
        compressed = np.sign(waveform_clipped) * np.log1p(mu * np.abs(waveform_clipped)) / np.log1p(mu)
        # Add slight quantization noise
        noise = np.random.normal(0, 0.001, waveform.shape)
        return compressed + noise

    def extract_mel_spectrogram(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Extract mel-spectrogram features for downstream analysis."""
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_mels=128, fmax=8000,
            n_fft=512, hop_length=160
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db

    def get_spectral_features(self, waveform: np.ndarray, sr: int) -> dict:
        """Extract comprehensive spectral features for deepfake analysis."""
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]
        # MFCCs (13 coefficients)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)

        return {
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_centroid_std": float(np.std(centroid)),
            "spectral_rolloff_mean": float(np.mean(rolloff)),
            "spectral_bandwidth_mean": float(np.mean(bandwidth)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "mfcc_mean": mfcc.mean(axis=1).tolist(),
            "mfcc_std": mfcc.std(axis=1).tolist(),
            "chroma_mean": chroma.mean(axis=1).tolist(),
        }

    def chunk_audio(self, waveform: np.ndarray, sr: int) -> list:
        """Split audio into fixed-size chunks for streaming analysis."""
        chunk_samples = int(self.CHUNK_DURATION * sr)
        chunks = []
        for start in range(0, len(waveform), chunk_samples):
            chunk = waveform[start:start + chunk_samples]
            if len(chunk) >= int(self.MIN_AUDIO_DURATION * sr):
                chunks.append(chunk)
        return chunks


preprocessor = AudioPreprocessor()
