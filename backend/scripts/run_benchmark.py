"""Quick benchmark runner for diagnostics."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.evaluation import run_benchmark, report_to_dict

if __name__ == "__main__":
    print("=== STEP 6: Full diagnostics (new pipeline) ===")
    d = report_to_dict(run_benchmark())
    m = d["metrics"]
    keys = [
        "f1_fake_detection", "precision_fake_detection", "recall_fake_detection",
        "deepfake_detection_accuracy_pct", "true_positives", "true_negatives",
        "false_positives", "false_negatives", "language_accuracy",
    ]
    print(json.dumps({k: m[k] for k in keys if k in m}, indent=2))
    print("Models:", json.dumps(d.get("models_loaded"), indent=2))
    for r in d.get("results", []):
        label = r['label'].encode('ascii', 'replace').decode('ascii')
        print(
            f"  {label[:28]:28s} exp={r['expected_category']:14s} "
            f"df={r['deepfake_probability']} ok={r['deepfake_binary_correct']} "
            f"model={str(r.get('models_used',''))[:20]}"
        )
