"""Generate Indian-language/accent demo audio for SIH benchmark."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
INDIC_DIR = ROOT / "public" / "demo" / "indic"
SR = 16000

# Prosody profiles approximating different Indian language rhythm patterns
LANG_PROFILES = {
    "hi": {"f0_base": 145, "mod_rate": 2.8, "seed": 101, "label": "Hindi"},
    "ta": {"f0_base": 160, "mod_rate": 3.5, "seed": 202, "label": "Tamil"},
    "te": {"f0_base": 155, "mod_rate": 3.2, "seed": 303, "label": "Telugu"},
    "kn": {"f0_base": 150, "mod_rate": 3.0, "seed": 404, "label": "Kannada"},
    "en-IN": {"f0_base": 140, "mod_rate": 2.5, "seed": 505, "label": "English-IN"},
}


def _lang_voice(duration, profile, synthetic=False):
    rng = np.random.default_rng(profile["seed"])
    t = np.linspace(0, duration, int(SR * duration), dtype=np.float32)
    f0 = profile["f0_base"] + 18 * np.sin(2 * np.pi * profile["mod_rate"] * t)
    if synthetic:
        f0 = np.full_like(t, profile["f0_base"])  # flat F0 = TTS-like
    phase = np.cumsum(2 * np.pi * f0 / SR)
    wave = 0.32 * np.sin(phase) + 0.14 * np.sin(2 * phase)
    env = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * (profile["mod_rate"] + 0.5) * t)) ** 2
    breath = 0.025 * rng.normal(0, 1, len(t))
    return ((wave * env) + breath).astype(np.float32)


def main():
    INDIC_DIR.mkdir(parents=True, exist_ok=True)
    for code, profile in LANG_PROFILES.items():
        genuine = _lang_voice(4.0, profile, synthetic=False)
        cloned = _lang_voice(4.0, profile, synthetic=True)
        sf.write(INDIC_DIR / f"{code}_genuine.wav", genuine, SR)
        sf.write(INDIC_DIR / f"{code}_cloned.wav", cloned, SR)
        print(f"  {profile['label']}: {code}_genuine.wav, {code}_cloned.wav")
    print(f"\nIndic demo audio ready in {INDIC_DIR}")
    print("Add real IndicTTS/InDeepFake samples to public/demo/custom/indic/ to override.")


if __name__ == "__main__":
    main()
