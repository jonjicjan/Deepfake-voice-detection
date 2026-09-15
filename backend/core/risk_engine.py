"""
VoiceShield AI — Dynamic Risk Engine
Aggregates all detection and context signals into a single
calibrated risk score (0–100) with component transparency.

Formula (per handoff document §C):
  risk = w1*deepfake + w2*speaker_mismatch + w3*prosody_anomaly +
         w4*replay + w5*unknown_attack + w6*context_risk
  → scaled 0–100 → adjusted by transaction_risk_multiplier

Key design constraint (per handoff §D):
  Prevention rules live in PolicyEngine, NOT here.
  RiskEngine only computes numbers — it does not block anything.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from core.ml_config import load_calibration


@dataclass
class RiskScore:
    score: float                    # Final risk score 0–100
    level: str                      # LOW, MEDIUM, HIGH, CRITICAL
    component_weights: dict         # For explainability
    explanation: str
    confidence: float               # Overall confidence in score


class RiskEngine:
    """
    Weighted dynamic risk scorer.

    Weights calibrated based on:
    - Deepfake detection: primary signal, highest weight
    - Speaker mismatch: strong identity signal
    - Prosody anomaly: supporting evidence
    - Replay probability: distinct attack vector
    - Unknown attack: uncertainty penalty
    - Context risk: multiplier, not additive

    Score levels:
    - LOW (0–30): Likely genuine, proceed normally
    - MEDIUM (30–70): Suspicious, require step-up verification
    - HIGH (70–85): Strong impersonation evidence, block sensitive actions
    - CRITICAL (85–100): Near-certain attack, block + escalate + alert

    These thresholds are configurable via PolicyEngine.
    """

    # Component weights — must sum to 1.0
    WEIGHTS = {
        "deepfake": 0.38,
        "speaker_mismatch": 0.25,
        "prosody_anomaly": 0.17,
        "replay": 0.10,
        "unknown_attack": 0.10,
    }

    # Default thresholds (PolicyEngine can override)
    THRESHOLDS = {
        "low": 30.0,
        "high": 70.0,
        "critical": 85.0,
    }

    def __init__(self):
        self.reload_calibration()

    def reload_calibration(self):
        cfg = load_calibration()
        self.WEIGHTS = dict(cfg["risk_weights"])
        self.THRESHOLDS = dict(cfg["risk_thresholds"])

    def compute(
        self,
        deepfake_probability: float,
        speaker_mismatch: float,
        prosody_anomaly: float,
        replay_probability: float,
        unknown_attack_score: float,
        context_risk: float,
        transaction_risk_multiplier: float = 1.0,
        speaker_enrolled: bool = False,
    ) -> RiskScore:
        """
        Compute final impersonation risk score.

        All input scores: 0.0 (safe) to 1.0 (maximum risk)
        Output: 0 (no risk) to 100 (certain attack)
        """
        # If speaker not enrolled, speaker_mismatch is uncertain — penalize less
        if not speaker_enrolled:
            effective_speaker_mismatch = speaker_mismatch * 0.5
        else:
            effective_speaker_mismatch = speaker_mismatch

        # Weighted base score (0–1)
        base_score = (
            self.WEIGHTS["deepfake"] * deepfake_probability +
            self.WEIGHTS["speaker_mismatch"] * effective_speaker_mismatch +
            self.WEIGHTS["prosody_anomaly"] * prosody_anomaly +
            self.WEIGHTS["replay"] * replay_probability +
            self.WEIGHTS["unknown_attack"] * unknown_attack_score
        )

        # Context risk is additive (up to 20 points additional)
        context_addition = context_risk * 0.20

        # Scale to 0–100
        raw_score = (base_score + context_addition) * 100

        # Apply transaction multiplier (context-driven amplification)
        adjusted_score = raw_score * transaction_risk_multiplier

        # Hard cap at 100
        final_score = float(np.clip(adjusted_score, 0.0, 100.0))

        level = self._classify_level(final_score)
        confidence = self._compute_confidence(
            [deepfake_probability, speaker_mismatch, prosody_anomaly, replay_probability]
        )

        component_weights = {
            "deepfake_contribution": round(self.WEIGHTS["deepfake"] * deepfake_probability * 100, 1),
            "speaker_contribution": round(self.WEIGHTS["speaker_mismatch"] * effective_speaker_mismatch * 100, 1),
            "prosody_contribution": round(self.WEIGHTS["prosody_anomaly"] * prosody_anomaly * 100, 1),
            "replay_contribution": round(self.WEIGHTS["replay"] * replay_probability * 100, 1),
            "context_contribution": round(context_addition * 100, 1),
        }

        explanation = self._explain(final_score, level, component_weights)

        return RiskScore(
            score=round(final_score, 1),
            level=level,
            component_weights=component_weights,
            explanation=explanation,
            confidence=round(confidence, 2),
        )

    def _classify_level(self, score: float) -> str:
        if score >= self.THRESHOLDS["critical"]:
            return "CRITICAL"
        elif score >= self.THRESHOLDS["high"]:
            return "HIGH"
        elif score >= self.THRESHOLDS["low"]:
            return "MEDIUM"
        else:
            return "LOW"

    def _compute_confidence(self, scores: list) -> float:
        """
        Higher confidence when signals agree (low variance).
        Lower confidence when signals are contradictory.
        """
        scores_array = np.array(scores)
        agreement = 1.0 - np.std(scores_array)
        return float(np.clip(agreement, 0, 1))

    def _explain(self, score: float, level: str, components: dict) -> str:
        dominant = max(components, key=components.get)
        dominant_pct = components[dominant]

        level_messages = {
            "LOW": f"Risk score {score:.0f}/100 — Caller appears genuine. Proceed normally.",
            "MEDIUM": f"Risk score {score:.0f}/100 — Suspicious signals detected. Step-up verification required before sensitive actions.",
            "HIGH": f"Risk score {score:.0f}/100 — Strong impersonation indicators. Block sensitive actions and require secondary verification.",
            "CRITICAL": f"Risk score {score:.0f}/100 — Near-certain voice cloning attack. BLOCK ALL ACTIONS. Alert supervisor immediately.",
        }

        dominant_names = {
            "deepfake_contribution": "deepfake detection",
            "speaker_contribution": "speaker mismatch",
            "prosody_contribution": "prosody anomaly",
            "replay_contribution": "replay detection",
            "context_contribution": "contextual risk",
        }

        return (
            f"{level_messages[level]} "
            f"Primary signal: {dominant_names.get(dominant, dominant)} (+{dominant_pct:.1f} pts)."
        )


risk_engine = RiskEngine()
