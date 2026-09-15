"""XLS-R + SLS deepfake detector (sukhdeveyash/XLS-R-SLS-Deepfake-Detection v1/epoch_2.pth)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("voiceshield.xlsr_sls")

HF_REPO = "sukhdeveyash/XLS-R-SLS-Deepfake-Detection"
CHECKPOINT_FILE = "v1/epoch_2.pth"
XLSR_BASE = Path(__file__).resolve().parents[1] / "pretrained_models" / "xlsr2_300m.pt"

_MODEL = None
_LOAD_ERROR: str | None = None
_DEVICE = None


def get_load_status() -> dict:
    return {
        "checkpoint_repo": HF_REPO,
        "checkpoint_file": CHECKPOINT_FILE,
        "xlsr_base_path": str(XLSR_BASE),
        "xlsr_base_exists": XLSR_BASE.exists(),
        "loaded": _MODEL is not None,
        "error": _LOAD_ERROR,
        "device": str(_DEVICE) if _DEVICE else None,
    }


def _strip_module_prefix(state_dict: dict) -> dict:
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_xlsr_sls_model(force: bool = False):
    global _MODEL, _LOAD_ERROR, _DEVICE
    if _MODEL is not None and not force:
        return _MODEL
    if _LOAD_ERROR and not force:
        return None

    try:
        import fairseq  # noqa: F401
    except ImportError as e:
        _LOAD_ERROR = (
            f"fairseq not installed ({e}). XLS-R-SLS requires fairseq + xlsr2_300m.pt. "
            "On Windows/Python 3.12, pip install fairseq typically fails — use Linux/WSL or Docker."
        )
        logger.warning(_LOAD_ERROR)
        return None

    if not XLSR_BASE.exists():
        _LOAD_ERROR = (
            f"Missing XLS-R base weights at {XLSR_BASE}. "
            "Download xlsr2_300m.pt from fairseq wav2vec/xlsr and place it there."
        )
        logger.warning(_LOAD_ERROR)
        return None

    try:
        from huggingface_hub import hf_hub_download
        from models.xlsr_sls.model import SLSDeepfakeModel

        ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=CHECKPOINT_FILE)
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SLSDeepfakeModel(device=_DEVICE, xlsr_path=str(XLSR_BASE))
        raw = torch.load(ckpt_path, map_location=_DEVICE, weights_only=False)
        state = raw if not isinstance(raw, dict) or "state_dict" not in raw else raw["state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state):
            state = _strip_module_prefix(state)
        model.load_state_dict(state, strict=False)
        model.to(_DEVICE)
        model.eval()
        _MODEL = model
        _LOAD_ERROR = None
        print(f"[OK] XLS-R-SLS loaded from {ckpt_path} on {_DEVICE}")
        return _MODEL
    except Exception as e:
        _LOAD_ERROR = str(e)
        logger.exception("XLS-R-SLS load failed")
        return None


def predict_fake_probability(waveform: np.ndarray, sr: int = 16000) -> float | None:
    """Return P(fake) in [0,1]. Expects mono float32 waveform."""
    model = load_xlsr_sls_model()
    if model is None:
        return None

    import librosa

    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    # Paper/repo uses fixed 64600 samples (~4.04s at 16kHz)
    target_len = 64600
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]

    x = torch.tensor(waveform, dtype=torch.float32, device=_DEVICE).unsqueeze(0).unsqueeze(-1)
    with torch.inference_mode():
        log_probs = model(x)
        probs = torch.exp(log_probs)[0]
    # index 1 = spoof/fake in ASVspoof convention (bonafide=0, spoof=1)
    return float(probs[1].item())
