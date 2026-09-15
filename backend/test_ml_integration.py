"""Quick integration test for neural ML pipeline."""
import time
import numpy as np
import soundfile as sf

from core.deepfake_detector import detector, preload_deepfake_model
from core.speaker_verifier import verifier, preload_speaker_model

sr = 16000
t = np.linspace(0, 3, sr * 3, dtype=np.float32)
wave = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.2 * np.sin(2 * np.pi * 300 * t)
sf.write("test_audio.wav", wave, sr)
print("Created test_audio.wav")

t0 = time.time()
df_ok = preload_deepfake_model()
print(f"Deepfake model loaded: {df_ok} ({time.time() - t0:.1f}s)")

t0 = time.time()
sp_ok = preload_speaker_model()
print(f"Speaker model loaded: {sp_ok} ({time.time() - t0:.1f}s)")

t0 = time.time()
result = detector.analyze(wave, sr, {})
print(f"Deepfake inference ({time.time() - t0:.2f}s): prob={result.probability:.3f}")
print(f"  label={result.label} model={result.model_used}")

t0 = time.time()
emb = verifier.extract_embedding(wave, sr)
emb_type = "ECAPA-TDNN" if len(emb) == 192 else "MFCC"
print(f"Speaker embedding ({time.time() - t0:.2f}s): dim={len(emb)} ({emb_type})")

verify = verifier.verify(emb, emb)
print(f"Self-verify: similarity={verify.similarity_score:.3f} model={verify.model_used}")
