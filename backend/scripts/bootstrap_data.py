"""Bootstrap /data folder from existing demo audio for pipeline testing.

This does NOT replace real ASVspoof/InDeepFake data — it only lets you
verify the training pipeline works before you add 100+ samples per language.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "public" / "demo"
DATA = ROOT / "data"

MAPPING = [
    ("genuine.wav", "en", "genuine"),
    ("cloned.wav", "en", "cloned"),
    ("cloned_transfer.wav", "en", "cloned"),
    ("replay.wav", "en", "cloned"),
    ("voice_conversion.wav", "en", "cloned"),
    ("indic/hi_genuine.wav", "hi", "genuine"),
    ("indic/hi_cloned.wav", "hi", "cloned"),
    ("indic/ta_genuine.wav", "ta", "genuine"),
    ("indic/ta_cloned.wav", "ta", "cloned"),
    ("indic/te_genuine.wav", "te", "genuine"),
    ("indic/te_cloned.wav", "te", "cloned"),
    ("indic/kn_genuine.wav", "kn", "genuine"),
    ("indic/kn_cloned.wav", "kn", "cloned"),
    ("indic/en-IN_genuine.wav", "en-IN", "genuine"),
    ("indic/en-IN_cloned.wav", "en-IN", "cloned"),
]


def main():
    copied = 0
    for src_rel, lang, label in MAPPING:
        src = DEMO / src_rel
        if not src.exists():
            print(f"[SKIP] missing {src}")
            continue
        dest_dir = DATA / lang / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        copied += 1
    print(f"[OK] Bootstrapped {copied} files into {DATA}")
    print("[NOTE] This is only 11-15 samples — NOT enough for production metrics.")
    print("       Add ASVspoof/InDeepFake to data/<lang>/<genuine|cloned>/ and re-run build_manifest.py")


if __name__ == "__main__":
    main()
