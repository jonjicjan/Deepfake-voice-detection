"""Frozen IndicWav2Vec encoder for Indian-language deepfake feature extraction."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("voiceshield.indic_encoder")

PRIMARY_MODEL_ID = "ai4bharat/indicwav2vec-hindi"
FALLBACK_MODEL_ID = "facebook/wav2vec2-large-xlsr-53"  # multilingual, ungated
MODEL_ID = os.getenv("INDIC_ENCODER_MODEL", PRIMARY_MODEL_ID)
HEAD_PATH = Path(__file__).resolve().parents[1] / "models" / "indic_classifier" / "classifier_head.pt"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "models" / "indic_classifier" / "config.json"

_ENCODER = None
_PROCESSOR = None
_HEAD: nn.Module | None = None
_DEVICE = None
_LOAD_ERROR: str | None = None


class ClassifierHead(nn.Module):
    """Lightweight MLP on mean-pooled encoder embeddings."""

    def __init__(self, input_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def get_indic_status() -> dict:
    return {
        "encoder_id": MODEL_ID,
        "primary_model": PRIMARY_MODEL_ID,
        "fallback_model": FALLBACK_MODEL_ID,
        "using_fallback": MODEL_ID == FALLBACK_MODEL_ID,
        "hf_token_set": bool(os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")),
        "encoder_loaded": _ENCODER is not None,
        "head_loaded": _HEAD is not None,
        "head_path": str(HEAD_PATH),
        "head_exists": HEAD_PATH.exists(),
        "error": _LOAD_ERROR,
        "device": str(_DEVICE) if _DEVICE else None,
    }


def _load_from_hf(model_id: str):
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    kwargs = {"token": token} if token else {}
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, **kwargs)
    encoder = Wav2Vec2Model.from_pretrained(model_id, **kwargs)
    return encoder, processor


def load_encoder(force: bool = False):
    global _ENCODER, _PROCESSOR, _DEVICE, _LOAD_ERROR, MODEL_ID
    if _ENCODER is not None and not force:
        return _ENCODER, _PROCESSOR

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = [MODEL_ID]
    if MODEL_ID == PRIMARY_MODEL_ID and FALLBACK_MODEL_ID not in candidates:
        candidates.append(FALLBACK_MODEL_ID)

    last_err = None
    for model_id in candidates:
        try:
            print(f"[INFO] Loading frozen encoder: {model_id}...")
            _ENCODER, _PROCESSOR = _load_from_hf(model_id)
            MODEL_ID = model_id
            _ENCODER.to(_DEVICE)
            _ENCODER.eval()
            for p in _ENCODER.parameters():
                p.requires_grad = False
            _LOAD_ERROR = None
            if model_id == FALLBACK_MODEL_ID:
                print(f"[WARN] Using fallback {FALLBACK_MODEL_ID} — accept ai4bharat gate + set HF_TOKEN for IndicWav2Vec")
            else:
                print(f"[OK] Encoder loaded on {_DEVICE} (hidden={_ENCODER.config.hidden_size})")
            return _ENCODER, _PROCESSOR
        except Exception as e:
            last_err = e
            print(f"[FAIL] {model_id}: {e}")
            _ENCODER, _PROCESSOR = None, None

    _LOAD_ERROR = str(last_err)
    logger.exception("Encoder load failed")
    return None, None


def load_classifier_head(force: bool = False) -> ClassifierHead | None:
    global _HEAD
    if _HEAD is not None and not force:
        return _HEAD
    if not HEAD_PATH.exists():
        return None
    import json

    encoder, _ = load_encoder()
    if encoder is None:
        return None
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    head = ClassifierHead(cfg["input_dim"], cfg.get("hidden", 256))
    head.load_state_dict(torch.load(HEAD_PATH, map_location="cpu", weights_only=True))
    head.to(_DEVICE)
    head.eval()
    _HEAD = head
    return _HEAD


@torch.inference_mode()
def extract_embedding(waveform: np.ndarray, sr: int) -> torch.Tensor | None:
    """Mean-pooled hidden states -> (hidden_dim,) tensor."""
    import librosa

    encoder, processor = load_encoder()
    if encoder is None or processor is None:
        return None

    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(_DEVICE)
    outputs = encoder(input_values)
    hidden = outputs.last_hidden_state  # (1, T, H)
    emb = hidden.mean(dim=1).squeeze(0).cpu()
    return emb


@torch.inference_mode()
def predict_fake_probability(waveform: np.ndarray, sr: int) -> float | None:
    head = load_classifier_head()
    emb = extract_embedding(waveform, sr)
    if head is None or emb is None:
        return None
    logits = head(emb.unsqueeze(0).to(_DEVICE))
    probs = torch.softmax(logits, dim=-1)[0]
    return float(probs[1].item())  # index 1 = cloned/fake
