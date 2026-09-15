"""Export Wav2Vec2 deepfake model to ONNX for faster CPU/GPU inference."""
import sys
from pathlib import Path

import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"
OUT_PATH = BACKEND / "pretrained_models" / "wav2vec2_deepfake.onnx"


def main():
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model.eval()

    dummy = fe(
        torch.zeros(16000).numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_names = list(dummy.keys())
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch"}

    print(f"Exporting to {OUT_PATH}...")
    torch.onnx.export(
        model,
        tuple(dummy[k] for k in input_names),
        str(OUT_PATH),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    print("[OK] ONNX export complete. Set prefer_onnx=true in calibration.json")


if __name__ == "__main__":
    main()
