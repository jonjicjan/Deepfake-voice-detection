"""
VoiceShield AI — Speaker Verifier
Speaker identity verification using SpeechBrain ECAPA-TDNN (192-dim embeddings)
with MFCC acoustic fallback when the neural model is unavailable.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np
import torch

from core.inference_utils import get_device

logger = logging.getLogger("voiceshield.speaker_verifier")

# Lazy-loaded SpeechBrain ECAPA-TDNN encoder
_ECAPA_CLASSIFIER = None
_ECAPA_FAILED = False
ECAPA_EMBEDDING_DIM = 192
MFCC_EMBEDDING_DIM = 128


def get_ecapa_classifier():
    """Load SpeechBrain ECAPA-TDNN speaker encoder (cached singleton)."""
    global _ECAPA_CLASSIFIER, _ECAPA_FAILED
    if _ECAPA_FAILED:
        return None
    if _ECAPA_CLASSIFIER is None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading ECAPA-TDNN speaker encoder (speechbrain/spkrec-ecapa-voxceleb)...")
            print("[INFO] Loading ECAPA-TDNN speaker model: speechbrain/spkrec-ecapa-voxceleb...")
            _ECAPA_CLASSIFIER = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": device},
                local_strategy=LocalStrategy.COPY,
            )
            print("[OK] ECAPA-TDNN Speaker Verifier model loaded successfully!")
        except Exception as e:
            logger.warning(f"Could not load ECAPA-TDNN model ({e}). Using MFCC fallback.")
            print(f"[WARN] ECAPA-TDNN model load failed: {e}. Falling back to MFCC acoustic engine.")
            _ECAPA_FAILED = True
            return None
    return _ECAPA_CLASSIFIER


@dataclass
class SpeakerVerificationResult:
    similarity_score: float      # 0=complete mismatch, 1=identical
    mismatch_probability: float  # 0=match, 1=mismatch (for risk engine)
    verified: bool
    confidence: str              # HIGH, MEDIUM, LOW
    explanation: str
    embedding_distance: float
    model_used: str = "MFCC-Acoustic"


class SpeakerVerifier:
    """
    Speaker identity verification via ECAPA-TDNN neural embeddings (192-dim)
    or MFCC acoustic fallback (128-dim) when the neural model is unavailable.
    """

    VERIFICATION_THRESHOLD = 0.75
    HIGH_CONFIDENCE_THRESHOLD = 0.90
    LOW_CONFIDENCE_THRESHOLD = 0.60

    def extract_embedding(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract speaker identity embedding from audio.
        Prefers ECAPA-TDNN (192-dim); falls back to MFCC statistics (128-dim).
        """
        ecapa = get_ecapa_classifier()
        if ecapa is not None:
            try:
                return self._extract_ecapa_embedding(waveform, sr, ecapa)
            except Exception as e:
                logger.error(f"ECAPA embedding extraction failed: {e}")
                print(f"[WARN] ECAPA embedding failed: {e}. Using MFCC fallback.")

        return self._extract_mfcc_embedding(waveform, sr)

    def _extract_ecapa_embedding(
        self, waveform: np.ndarray, sr: int, classifier
    ) -> np.ndarray:
        """Extract 192-dim L2-normalized ECAPA-TDNN speaker embedding."""
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000

        # SpeechBrain expects (batch, time) float32 tensor
        wav_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        device = get_device()
        if device.type == "cuda":
            wav_tensor = wav_tensor.to(device)
        with torch.inference_mode():
            embedding = classifier.encode_batch(wav_tensor)

        embedding = embedding.squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _extract_mfcc_embedding(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Extract 128-dim MFCC-based speaker embedding (acoustic fallback)."""
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
        features = features - np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        std[std < 1e-8] = 1.0
        features = features / std

        embedding = np.concatenate([
            np.mean(features, axis=1),
            np.std(features, axis=1),
        ])

        spec_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]

        spectral_stats = np.array([
            np.mean(spec_centroid), np.std(spec_centroid),
            np.mean(rolloff), np.std(rolloff),
            np.mean(zcr), np.std(zcr),
            np.mean(waveform ** 2),
            float(np.percentile(np.abs(waveform), 75)),
        ])

        full_embedding = np.concatenate([embedding, spectral_stats])
        norm = np.linalg.norm(full_embedding)
        if norm > 0:
            full_embedding = full_embedding / norm

        return full_embedding.astype(np.float32)

    def _resolve_embedding_method(self, reference_dim: int) -> str:
        """Pick embedding backend compatible with stored reference vector."""
        if reference_dim == ECAPA_EMBEDDING_DIM and get_ecapa_classifier() is not None:
            return "ecapa"
        if reference_dim == MFCC_EMBEDDING_DIM:
            return "mfcc"
        # Unknown dimension — prefer ECAPA for new extractions
        return "ecapa" if get_ecapa_classifier() is not None else "mfcc"

    def verify(
        self,
        test_embedding: np.ndarray,
        reference_embedding: np.ndarray,
    ) -> SpeakerVerificationResult:
        """Compare test embedding against enrolled reference."""
        if len(test_embedding) != len(reference_embedding):
            return SpeakerVerificationResult(
                similarity_score=0.0,
                mismatch_probability=1.0,
                verified=False,
                confidence="MISMATCH",
                explanation=(
                    f"Embedding dimension mismatch ({len(test_embedding)} vs "
                    f"{len(reference_embedding)}). Please re-enroll the speaker profile."
                ),
                embedding_distance=float("inf"),
                model_used="Dimension-Mismatch",
            )

        model_used = (
            "ECAPA-TDNN (VoxCeleb)"
            if len(test_embedding) == ECAPA_EMBEDDING_DIM
            else "MFCC-Acoustic"
        )

        similarity = float(
            np.dot(test_embedding, reference_embedding)
            / (np.linalg.norm(test_embedding) * np.linalg.norm(reference_embedding) + 1e-8)
        )
        similarity = float(np.clip(similarity, -1.0, 1.0))
        distance = float(np.linalg.norm(test_embedding - reference_embedding))
        mismatch_prob = float(np.clip(1.0 - (similarity + 1.0) / 2.0, 0, 1))
        verified = similarity >= self.VERIFICATION_THRESHOLD

        if similarity >= self.HIGH_CONFIDENCE_THRESHOLD:
            confidence = "HIGH"
        elif similarity >= self.VERIFICATION_THRESHOLD:
            confidence = "MEDIUM"
        elif similarity >= self.LOW_CONFIDENCE_THRESHOLD:
            confidence = "LOW"
        else:
            confidence = "MISMATCH"

        explanation = self._explain(similarity, verified, confidence, model_used)

        return SpeakerVerificationResult(
            similarity_score=similarity,
            mismatch_probability=mismatch_prob,
            verified=verified,
            confidence=confidence,
            explanation=explanation,
            embedding_distance=distance,
            model_used=model_used,
        )

    def verify_from_audio(
        self,
        test_waveform: np.ndarray,
        sr: int,
        reference_embedding: np.ndarray,
    ) -> SpeakerVerificationResult:
        """End-to-end verify: extract embedding from audio then compare."""
        ref_arr = np.array(reference_embedding, dtype=np.float32)
        method = self._resolve_embedding_method(len(ref_arr))

        if method == "ecapa":
            ecapa = get_ecapa_classifier()
            test_embedding = self._extract_ecapa_embedding(test_waveform, sr, ecapa)
        else:
            test_embedding = self._extract_mfcc_embedding(test_waveform, sr)

        return self.verify(test_embedding, ref_arr)

    def _explain(
        self, similarity: float, verified: bool, confidence: str, model_used: str
    ) -> str:
        prefix = f"[{model_used}] "
        if verified and confidence == "HIGH":
            return (
                prefix
                + f"High confidence speaker match (similarity: {similarity:.2f}). "
                "Caller identity consistent with enrolled profile."
            )
        if verified and confidence == "MEDIUM":
            return (
                prefix
                + f"Moderate speaker match (similarity: {similarity:.2f}). "
                "Channel or quality variation detected."
            )
        if not verified and confidence == "LOW":
            return (
                prefix
                + f"Weak match (similarity: {similarity:.2f}). "
                "Possible channel mismatch or voice alteration."
            )
        return (
            prefix
            + f"Speaker mismatch detected (similarity: {similarity:.2f}). "
            "Claimed identity does not match enrolled voice profile."
        )


def preload_speaker_model() -> bool:
    """Eager-load ECAPA-TDNN at startup. Returns True if loaded."""
    return get_ecapa_classifier() is not None


verifier = SpeakerVerifier()
