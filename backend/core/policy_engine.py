"""
VoiceShield AI — Policy Engine
Configurable, JSON-driven prevention policy.

Key architectural principle (per handoff §D):
  Prevention rules are POLICY-DRIVEN, not hard-coded in the ML model.
  The PolicyEngine is the only component that decides what to block.
  ML components only compute probabilities.

Policy actions:
  LOW  (< 30)  → ALLOWED — proceed normally
  MEDIUM (30–70) → STEP_UP_VERIFICATION — OTP, MFA, callback
  HIGH (70–85) → BLOCKED — block sensitive action + alert
  CRITICAL (>85) → BLOCKED + ESCALATED — block + alert all channels + supervisor

Policies are fully configurable per organization and use-case scenario.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from db.schemas import RiskLevel, ActionTaken


@dataclass
class PolicyRule:
    name: str
    min_score: float
    max_score: float
    action: ActionTaken
    alert_channels: List[str]
    block_transaction: bool
    require_verification_methods: List[str]
    escalate_to_supervisor: bool = False
    log_incident: bool = True
    description: str = ""


@dataclass
class PolicyDecision:
    action: ActionTaken
    risk_level: RiskLevel
    block_transaction: bool
    secondary_verification_required: bool
    verification_methods: List[str]
    alert_channels: List[str]
    escalate: bool
    alert_message: str
    alert_title: str
    recommended_actions: List[str]


class PolicyEngine:
    """
    Configurable prevention policy engine.
    Policy rules are evaluated in order — first matching rule wins.
    """

    DEFAULT_RULES: List[PolicyRule] = field(default_factory=list)

    def __init__(self):
        self._rules = self._build_default_rules()
        self._config = {
            "low_threshold": 30.0,
            "high_threshold": 70.0,
            "critical_threshold": 85.0,
            "auto_block_on_high": True,
            "require_mfa_on_medium": True,
            "high_value_transaction_limit_inr": 50_000.0,
            "alert_channels": ["ui", "email"],
        }

    def _build_default_rules(self) -> List[PolicyRule]:
        return [
            PolicyRule(
                name="CRITICAL_BLOCK",
                min_score=85.0,
                max_score=100.0,
                action=ActionTaken.ESCALATED,
                alert_channels=["ui", "email", "sms", "supervisor"],
                block_transaction=True,
                require_verification_methods=["supervisor_approval", "callback_registered_device", "dual_mfa"],
                escalate_to_supervisor=True,
                description="Near-certain voice cloning attack — block all actions, escalate immediately",
            ),
            PolicyRule(
                name="HIGH_BLOCK",
                min_score=70.0,
                max_score=85.0,
                action=ActionTaken.BLOCKED,
                alert_channels=["ui", "email", "sms"],
                block_transaction=True,
                require_verification_methods=["callback_registered_device", "mfa_otp"],
                escalate_to_supervisor=False,
                description="Strong impersonation evidence — block sensitive action, require callback",
            ),
            PolicyRule(
                name="MEDIUM_STEPUP",
                min_score=30.0,
                max_score=70.0,
                action=ActionTaken.STEP_UP_VERIFICATION,
                alert_channels=["ui"],
                block_transaction=False,
                require_verification_methods=["mfa_otp"],
                escalate_to_supervisor=False,
                description="Suspicious signals — allow after additional verification",
            ),
            PolicyRule(
                name="LOW_ALLOW",
                min_score=0.0,
                max_score=30.0,
                action=ActionTaken.ALLOWED,
                alert_channels=[],
                block_transaction=False,
                require_verification_methods=[],
                escalate_to_supervisor=False,
                log_incident=False,
                description="Low risk — proceed normally",
            ),
        ]

    def evaluate(self, risk_score: float, risk_level: str) -> PolicyDecision:
        """Apply policy rules to a risk score and return prevention decision."""

        for rule in self._rules:
            if rule.min_score <= risk_score <= rule.max_score:
                return self._build_decision(rule, risk_score, risk_level)

        # Fallback to allow (shouldn't happen with default rules)
        return PolicyDecision(
            action=ActionTaken.ALLOWED,
            risk_level=RiskLevel.LOW,
            block_transaction=False,
            secondary_verification_required=False,
            verification_methods=[],
            alert_channels=[],
            escalate=False,
            alert_message="Risk assessment complete. No action required.",
            alert_title="Call Verified",
            recommended_actions=["Proceed normally"],
        )

    def _build_decision(self, rule: PolicyRule, score: float, level: str) -> PolicyDecision:
        secondary_required = len(rule.require_verification_methods) > 0

        alert_messages = {
            "CRITICAL_BLOCK": (
                "CRITICAL THREAT — Voice Cloning Attack Detected",
                f"Risk Score {score:.0f}/100 — AI analysis strongly indicates this caller is using a cloned or AI-generated voice. "
                "All sensitive actions have been blocked. Supervisor has been notified. "
                "Do not disclose information or approve requests until identity is verified through a registered device callback.",
            ),
            "HIGH_BLOCK": (
                "HIGH RISK — Voice Manipulation Suspected",
                f"Risk Score {score:.0f}/100 — Significant synthetic speech indicators detected. "
                "Sensitive transaction blocked. Verify caller identity via callback to registered number before proceeding.",
            ),
            "MEDIUM_STEPUP": (
                "CAUTION — Additional Verification Required",
                f"Risk Score {score:.0f}/100 — Suspicious voice patterns detected. "
                "Complete OTP verification before proceeding with sensitive actions.",
            ),
            "LOW_ALLOW": (
                "Voice Verified",
                f"Risk Score {score:.0f}/100 — No significant synthetic speech artifacts detected. Call appears genuine.",
            ),
        }

        title, message = alert_messages.get(rule.name, ("Risk Alert", f"Risk Score {score:.0f}/100"))

        recommended = []
        if rule.escalate_to_supervisor:
            recommended.append("Notify supervisor immediately")
        if "callback_registered_device" in rule.require_verification_methods:
            recommended.append("End call and callback via registered number")
        if "mfa_otp" in rule.require_verification_methods:
            recommended.append("Request OTP verification on registered mobile")
        if "dual_mfa" in rule.require_verification_methods:
            recommended.append("Require dual-factor authentication")
        if not recommended:
            recommended.append("Proceed with standard verification")

        return PolicyDecision(
            action=rule.action,
            risk_level=RiskLevel(level) if level in [r.value for r in RiskLevel] else RiskLevel.MEDIUM,
            block_transaction=rule.block_transaction,
            secondary_verification_required=secondary_required,
            verification_methods=rule.require_verification_methods,
            alert_channels=rule.alert_channels,
            escalate=rule.escalate_to_supervisor,
            alert_message=message,
            alert_title=title,
            recommended_actions=recommended,
        )

    def update_config(self, config: dict):
        """Hot-update policy configuration without restarting service."""
        self._config.update(config)
        if "high_value_transaction_limit" in config:
            self._config["high_value_transaction_limit_inr"] = config["high_value_transaction_limit"]
        for rule in self._rules:
            if rule.name == "CRITICAL_BLOCK":
                rule.min_score = config.get("critical_threshold", self._config.get("critical_threshold", 85.0))
            elif rule.name == "HIGH_BLOCK":
                rule.min_score = config.get("high_threshold", self._config.get("high_threshold", 70.0))
                rule.max_score = config.get("critical_threshold", self._config.get("critical_threshold", 85.0))
            elif rule.name == "MEDIUM_STEPUP":
                rule.min_score = config.get("low_threshold", self._config.get("low_threshold", 30.0))
                rule.max_score = config.get("high_threshold", self._config.get("high_threshold", 70.0))
            elif rule.name == "LOW_ALLOW":
                rule.max_score = config.get("low_threshold", self._config.get("low_threshold", 30.0))
        if "alert_channels" in config:
            ch = config["alert_channels"]
            for rule in self._rules:
                if rule.name == "CRITICAL_BLOCK":
                    rule.alert_channels = list(dict.fromkeys(ch + ["sms", "supervisor"]))
                elif rule.name == "HIGH_BLOCK":
                    rule.alert_channels = [c for c in ch if c in ("ui", "email", "sms")] or ["ui", "email"]
                elif rule.name == "MEDIUM_STEPUP":
                    rule.alert_channels = ["ui"] if "ui" in ch else []

    def get_config(self) -> dict:
        cfg = self._config.copy()
        cfg["high_value_transaction_limit"] = cfg.get(
            "high_value_transaction_limit_inr", cfg.get("high_value_transaction_limit", 50000.0)
        )
        return cfg


policy_engine = PolicyEngine()
