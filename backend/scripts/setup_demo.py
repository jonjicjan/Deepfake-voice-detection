"""Generate demo audio samples and enroll CEO speaker profile for SIH demo.

Custom recordings (recommended for judges):
  Place real .wav files in public/demo/custom/ — they override generated defaults:
    - genuine.wav       — your real voice (3+ seconds)
    - cloned.wav        — ElevenLabs / AI TTS clone
    - cloned_transfer.wav
    - replay.wav        — phone speaker replay of genuine
    - voice_conversion.wav
"""
import asyncio
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "public" / "demo"
CUSTOM_DIR = DEMO_DIR / "custom"
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

SR = 16000


def _natural_voice(duration=4.0, seed=42):
    """Speech-like signal with pitch jitter, breath noise, and syllable-like envelope."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(SR * duration), dtype=np.float32)
    f0 = 130 + 25 * np.sin(2 * np.pi * 0.7 * t) + rng.normal(0, 4, len(t))
    f0 = np.clip(f0, 80, 280)
    phase = np.cumsum(2 * np.pi * f0 / SR)
    wave = 0.30 * np.sin(phase)
    wave += 0.12 * np.sin(2 * phase)
    wave += 0.06 * np.sin(3 * phase)
    # syllable-rate amplitude modulation
    env = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.2 * t)) ** 2
    breath = 0.03 * rng.normal(0, 1, len(t))
    return ((wave * env) + breath).astype(np.float32)


def _synthetic_voice(duration=4.0):
    """Over-smooth TTS-like signal."""
    t = np.linspace(0, duration, int(SR * duration), dtype=np.float32)
    f0 = 155.0
    phase = 2 * np.pi * f0 * t
    return (0.42 * np.sin(phase) + 0.28 * np.sin(2 * phase)).astype(np.float32)


def _replay_voice(duration=4.0, seed=42):
    wave = _natural_voice(duration, seed)
    rng = np.random.default_rng(seed + 1)
    delay = int(0.035 * SR)
    reverb = np.zeros_like(wave)
    reverb[delay:] = wave[:-delay] * 0.35
    # μ-law style compression artifact
    compressed = np.sign(wave) * np.log1p(255 * np.abs(wave)) / np.log1p(255)
    return (compressed + reverb + 0.05 * rng.normal(0, 1, len(wave))).astype(np.float32)


def _voice_conversion(duration=4.0):
    natural = _natural_voice(duration, seed=99)
    synthetic = _synthetic_voice(duration)
    # formant-mismatch blend
    return (0.65 * synthetic + 0.35 * natural).astype(np.float32)


def _get_or_generate(name: str, generator) -> np.ndarray:
    custom = CUSTOM_DIR / name
    default = DEMO_DIR / name
    if custom.exists():
        wave, sr = sf.read(custom, dtype="float32")
        if sr != SR:
            import librosa
            wave = librosa.resample(wave, orig_sr=sr, target_sr=SR)
        print(f"  Using CUSTOM recording: {custom}")
        return wave.astype(np.float32)
    wave = generator()
    sf.write(default, wave, SR)
    print(f"  Generated default: {default}")
    return wave


async def enroll_ceo(waveform):
    from core.speaker_verifier import verifier
    from db.database import init_db, save_speaker_embedding

    await init_db()
    emb = verifier.extract_embedding(waveform, SR)
    await save_speaker_embedding("CEO-001", "Priya Sharma (CEO)", emb.tolist())
    print(f"Enrolled CEO-001 with {len(emb)}-dim ECAPA embedding")


def main():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)

    print("Setting up demo audio (custom/ overrides generated):")
    samples = {
        "genuine.wav": _get_or_generate("genuine.wav", lambda: _natural_voice(4.0, 42)),
        "cloned.wav": _get_or_generate("cloned.wav", lambda: _synthetic_voice(4.0)),
        "cloned_transfer.wav": _get_or_generate("cloned_transfer.wav", lambda: _synthetic_voice(4.5)),
        "replay.wav": _get_or_generate("replay.wav", lambda: _replay_voice(4.0)),
        "voice_conversion.wav": _get_or_generate("voice_conversion.wav", _voice_conversion),
    }

    asyncio.run(enroll_ceo(samples["genuine.wav"]))
    print("\nDemo setup complete.")
    print("For best SIH results, add real recordings to:", CUSTOM_DIR)


if __name__ == "__main__":
    main()
