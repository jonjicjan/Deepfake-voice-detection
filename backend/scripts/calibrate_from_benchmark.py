"""Auto-calibrate thresholds from benchmark results."""
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from core.evaluation import run_benchmark
from core.ml_config import save_calibration, CALIBRATION_PATH


def main():
    print("Running benchmark to calibrate thresholds...")
    report = run_benchmark()
    metrics = report.metrics
    results = report.results

    genuine = [r for r in results if r["expected_category"] == "GENUINE" and r["file_exists"]]
    fake = [r for r in results if r["expected_category"] != "GENUINE" and r["file_exists"]]

    if genuine and fake:
        g_max = max(r["deepfake_probability"] for r in genuine)
        f_min = min(r["deepfake_probability"] for r in fake)
        threshold = round((g_max + f_min) / 2, 3)
        threshold = float(max(0.35, min(0.65, threshold)))
    else:
        threshold = 0.5

    updates = {
        "deepfake_threshold": threshold,
        "risk_thresholds": {
            "low": 30.0,
            "high": 70.0 if metrics["f1_fake_detection"] >= 0.8 else 65.0,
            "critical": 85.0 if metrics["precision_fake_detection"] >= 0.9 else 80.0,
        },
        "_calibrated_from_benchmark": {
            "f1": metrics["f1_fake_detection"],
            "accuracy_pct": metrics["deepfake_detection_accuracy_pct"],
            "timestamp": report.timestamp,
        },
    }

    cfg = save_calibration(updates)
    print(f"Saved calibration to {CALIBRATION_PATH}")
    print(f"  deepfake_threshold = {cfg['deepfake_threshold']}")
    print(f"  risk_thresholds    = {cfg['risk_thresholds']}")
    print(f"  F1={metrics['f1_fake_detection']}  Accuracy={metrics['deepfake_detection_accuracy_pct']}%")


if __name__ == "__main__":
    main()
