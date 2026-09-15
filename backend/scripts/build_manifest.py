"""Build manifest.csv from /data/<language>/<genuine|cloned>/*.wav layout.

Also supports ingesting ASVspoof/InDeepFake after manual download — place files
under data/ following the folder convention and re-run this script.

Usage:
  python scripts/build_manifest.py
  python scripts/build_manifest.py --data-dir ../data --output ../data/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

VALID_LABELS = {"genuine", "cloned", "bonafide", "spoof", "fake", "real"}
LABEL_MAP = {
    "genuine": 0,
    "bonafide": 0,
    "real": 0,
    "cloned": 1,
    "spoof": 1,
    "fake": 1,
}


def scan_data_dir(data_dir: Path) -> list[dict]:
    rows = []
    if not data_dir.exists():
        print(f"[WARN] Data directory not found: {data_dir}")
        return rows

    for lang_dir in sorted(data_dir.iterdir()):
        if not lang_dir.is_dir() or lang_dir.name.startswith("."):
            continue
        language = lang_dir.name
        for label_dir in sorted(lang_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label_name = label_dir.name.lower()
            if label_name not in VALID_LABELS:
                print(f"[SKIP] Unknown label folder: {label_dir}")
                continue
            label = LABEL_MAP[label_name]
            for wav in sorted(label_dir.glob("**/*.wav")):
                rows.append({
                    "filepath": str(wav.resolve()),
                    "label": label,
                    "language": language,
                    "label_name": "genuine" if label == 0 else "cloned",
                })
    return rows


def write_manifest(rows: list[dict], output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "language", "label_name"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]):
    from collections import Counter

    print(f"\nTotal samples: {len(rows)}")
    if not rows:
        return
    by_lang = Counter(r["language"] for r in rows)
    by_label = Counter(r["label_name"] for r in rows)
    print("By language:", dict(by_lang))
    print("By label:", dict(by_label))
    print("\nPer-language breakdown:")
    for lang in sorted(by_lang):
        lang_rows = [r for r in rows if r["language"] == lang]
        g = sum(1 for r in lang_rows if r["label"] == 0)
        c = sum(1 for r in lang_rows if r["label"] == 1)
        status = "OK" if g >= 100 and c >= 100 else "INSUFFICIENT (need 100+ each)"
        print(f"  {lang}: genuine={g}, cloned={c}  [{status}]")


def main():
    parser = argparse.ArgumentParser(description="Build manifest.csv from /data folder")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or (args.data_dir / "manifest.csv")
    rows = scan_data_dir(args.data_dir)
    write_manifest(rows, output)
    print(f"[OK] Wrote {len(rows)} entries to {output}")
    print_summary(rows)


if __name__ == "__main__":
    main()
