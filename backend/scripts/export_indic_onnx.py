"""Export Indic encoder + classifier head to ONNX for faster inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.indic_encoder import ClassifierHead, load_encoder, load_classifier_head, HEAD_PATH, CONFIG_PATH  # noqa: E402

OUT_DIR = ROOT / "models" / "indic_classifier"
ONNX_PATH = OUT_DIR / "classifier_head.onnx"


class EncoderClassifierWrapper(torch.nn.Module):
    """End-to-end: raw waveform -> fake probability."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, input_values):
        hidden = self.encoder(input_values).last_hidden_state
        emb = hidden.mean(dim=1)
        logits = self.head(emb)
        return torch.softmax(logits, dim=-1)


def main():
    import json
    import soundfile as sf

    if not HEAD_PATH.exists():
        print(f"[ERROR] Train classifier first. Missing {HEAD_PATH}")
        sys.exit(1)

    encoder, processor = load_encoder()
    head = load_classifier_head()
    if encoder is None or head is None:
        print("[ERROR] Encoder or head not loaded")
        sys.exit(1)

    device = torch.device("cpu")
    encoder.to(device).eval()
    head.to(device).eval()
    wrapper = EncoderClassifierWrapper(encoder, head).to(device).eval()

    # Test sample
    wav_path = ROOT.parent / "public" / "demo" / "cloned.wav"
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    inputs = processor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
    x = inputs["input_values"]

    with torch.inference_mode():
        pt_probs = wrapper(x)[0]
    pt_fake = float(pt_probs[1])

    # Export head only (encoder is large — export combined if disk allows)
    dummy = torch.randn(1, 16000)  # placeholder — export classifier on embeddings
    import json as _json
    with open(CONFIG_PATH) as f:
        cfg = _json.load(f)
    emb_dim = cfg["input_dim"]
    dummy_emb = torch.randn(1, emb_dim)

    import onnxruntime as ort

    torch.onnx.export(
        head,
        dummy_emb,
        str(ONNX_PATH),
        input_names=["embedding"],
        output_names=["logits"],
        dynamic_axes={"embedding": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"[OK] Exported classifier head ONNX -> {ONNX_PATH}")

    try:
        sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        emb_np = dummy_emb.numpy()
        onnx_logits = sess.run(None, {"embedding": emb_np})[0]
        onnx_probs = torch.softmax(torch.tensor(onnx_logits), dim=-1)[0]
        pt_logits = head(dummy_emb)
        pt_probs2 = torch.softmax(pt_logits, dim=-1)[0]
        max_diff = float((onnx_probs - pt_probs2).abs().max())
        print(f"ONNX vs PyTorch max prob diff: {max_diff:.6f}")
        print(f"PyTorch fake prob on cloned.wav: {pt_fake:.4f}")
        if max_diff < 1e-4:
            print("[OK] ONNX export verified — predictions match")
        else:
            print("[WARN] ONNX mismatch > 1e-4 — investigate before deploying")
    except ImportError:
        print("[WARN] onnxruntime not available for verification")


if __name__ == "__main__":
    main()
