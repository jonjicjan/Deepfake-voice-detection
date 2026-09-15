"""Sweep decision threshold and recommend optimal operating point.

Usage:
  python scripts/tune_threshold.py
  python scripts/tune_threshold.py --predictions models/indic_classifier/test_predictions.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED = ROOT / "models" / "indic_classifier" / "test_predictions.npz"
OUT_JSON = ROOT / "models" / "indic_classifier" / "threshold_sweep.json"


def sweep(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
            "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
            "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
            "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
        })
    return rows


def recommend(rows):
    # Prefer threshold that maximizes F1 while keeping FP rate low
    best_f1 = max(rows, key=lambda r: r["f1"])
    low_fp = [r for r in rows if r["fp"] == 0]
    best_balanced = max(low_fp, key=lambda r: r["recall"]) if low_fp else best_f1
    return best_f1, best_balanced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    args = parser.parse_args()

    if not args.predictions.exists():
        print(f"[ERROR] Run train_classifier.py first. Missing: {args.predictions}")
        return

    data = np.load(args.predictions)
    y_true, y_prob = data["y_true"], data["y_prob"]

    print(f"Samples: {len(y_true)}  (cloned={y_true.sum()}, genuine={(y_true==0).sum()})")
    print(f"\n{'Thresh':>7} {'Prec':>6} {'Recall':>7} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    print("-" * 52)

    rows = sweep(y_true, y_prob)
    for r in rows:
        print(
            f"{r['threshold']:7.2f} {r['precision']:6.3f} {r['recall']:7.3f} "
            f"{r['f1']:6.3f} {r['tp']:4d} {r['fp']:4d} {r['fn']:4d} {r['tn']:4d}"
        )

    best_f1, best_balanced = recommend(rows)
    print("\n--- Recommendations ---")
    print(f"Max F1:      threshold={best_f1['threshold']}  F1={best_f1['f1']}  "
          f"(prec={best_f1['precision']}, recall={best_f1['recall']}, FP={best_f1['fp']}, FN={best_f1['fn']})")
    print(f"Low-FP bias: threshold={best_balanced['threshold']}  F1={best_balanced['f1']}  "
          f"(prec={best_balanced['precision']}, recall={best_balanced['recall']}, FP={best_balanced['fp']}, FN={best_balanced['fn']})")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"sweep": rows, "best_f1": best_f1, "best_low_fp": best_balanced}, f, indent=2)

    # Update calibration.json threshold
    cal_path = ROOT / "data" / "calibration.json"
    if cal_path.exists():
        with open(cal_path, encoding="utf-8") as f:
            cal = json.load(f)
        cal["deepfake_threshold"] = best_balanced["threshold"]
        cal["indic_classifier_threshold"] = best_balanced["threshold"]
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(cal, f, indent=2)
        print(f"\n[OK] Updated {cal_path} deepfake_threshold -> {best_balanced['threshold']}")


if __name__ == "__main__":
    main()
