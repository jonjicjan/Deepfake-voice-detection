"""
VoiceShield AI — Context Engine
Enriches raw AI analysis scores with call/transaction/caller metadata
to produce a context-adjusted risk assessment.

Key principle: The same voice should not receive the same risk score in every scenario.
A 55% deepfake probability in a casual conversation ≠ same risk as in a ₹25L wire transfer.
"""

from dataclasses import dataclass
from typing import Optional
from db.schemas import CallerContext, TransactionContext
from core.fraud_registry import lookup_fraud


@dataclass
class ContextResult:
    context_risk: float          # 0=low context risk, 1=high context risk
    transaction_risk_multiplier: float  # multiplies base risk
    privileged_workflow: bool
    high_value_transaction: bool
    unknown_caller: bool
    fraud_history_flag: bool
    fraud_registry_match: bool
    fraud_registry_detail: str
    context_explanation: str
    recommended_verification: list


class ContextEngine:
    """
    Contextual risk enrichment engine.

    Context signals evaluated:
    1. Transaction amount and type
    2. Caller identity and known-contact status
    3. Historical fraud indicators
    4. Call origin (VoIP calls carry higher risk)
    5. Privileged workflow (financial approval, data access)
    6. Time-of-day and behavioral anomalies

    Policy principle: Context risk is additive to AI detection risk,
    never the sole determinant. A clean voice in a high-risk context
    still warrants step-up verification.
    """

    HIGH_VALUE_THRESHOLD_INR = 50_000     # ₹50,000
    CRITICAL_VALUE_THRESHOLD_INR = 500_000  # ₹5,00,000

    PRIVILEGED_TRANSACTION_TYPES = {
        "FUND_TRANSFER", "WIRE_TRANSFER", "APPROVAL", "DATA_ACCESS",
        "CREDENTIAL_CHANGE", "ADMIN_ACTION", "PAYMENT_RELEASE"
    }

    HIGH_RISK_CALL_ORIGINS = {"VOIP", "UNKNOWN", "INTERNATIONAL"}

    def evaluate(
        self,
        caller: Optional[CallerContext],
        transaction: Optional[TransactionContext],
    ) -> ContextResult:
        """Compute context risk score from metadata."""

        context_signals = []
        risk_score = 0.0
        multiplier = 1.0
        verifications = []

        # --- Caller context signals ---
        unknown_caller = True
        fraud_flag = False
        fraud_lookup = lookup_fraud(manual_flag=False)

        if caller:
            fraud_lookup = lookup_fraud(
                caller_id=caller.caller_id,
                phone_number=caller.phone_number,
                caller_name=caller.caller_name,
                manual_flag=caller.historical_fraud_flag,
            )
            if fraud_lookup.is_flagged:
                fraud_flag = True
                context_signals.append(fraud_lookup.explanation)
                risk_score += 0.35 + min(0.1, fraud_lookup.prior_incidents * 0.03)
                multiplier = max(multiplier, 1.5)
                verifications.append("Cross-check against fraud registry immediately")

            if not caller.is_known_contact:
                context_signals.append("unknown caller")
                risk_score += 0.2
                verifications.append("Verify caller identity through registered contact list")

            if caller.historical_fraud_flag and not fraud_flag:
                fraud_flag = True
                context_signals.append("caller flagged in fraud history")
                risk_score += 0.4
                multiplier = max(multiplier, 1.5)
                verifications.append("Cross-check against fraud registry immediately")

            if caller.call_origin in self.HIGH_RISK_CALL_ORIGINS:
                context_signals.append(f"high-risk call origin: {caller.call_origin}")
                risk_score += 0.15

            unknown_caller = not caller.is_known_contact

            if caller.claimed_identity and caller.caller_name:
                if caller.claimed_identity.lower() != caller.caller_name.lower():
                    context_signals.append("claimed identity does not match contact record")
                    risk_score += 0.25
                    verifications.append("Identity claim mismatch — require callback verification")

        # --- Transaction context signals ---
        high_value = False
        privileged = False

        if transaction:
            amount = transaction.amount or 0

            if amount >= self.CRITICAL_VALUE_THRESHOLD_INR:
                high_value = True
                context_signals.append(f"critical-value transaction: ₹{amount:,.0f}")
                risk_score += 0.35
                multiplier = max(multiplier, 1.8)
                verifications.extend([
                    "Dual-approval required for amounts above ₹5,00,000",
                    "Initiate callback to registered device",
                    "Notify compliance officer",
                ])
            elif amount >= self.HIGH_VALUE_THRESHOLD_INR:
                high_value = True
                context_signals.append(f"high-value transaction: ₹{amount:,.0f}")
                risk_score += 0.2
                multiplier = max(multiplier, 1.4)
                verifications.append("OTP verification required for high-value transactions")

            if transaction.transaction_type in self.PRIVILEGED_TRANSACTION_TYPES:
                privileged = True
                context_signals.append(f"privileged action: {transaction.transaction_type}")
                risk_score += 0.15
                verifications.append("Privileged action — log and require supervisor confirmation")

            if transaction.is_privileged_workflow:
                privileged = True
                risk_score += 0.1

        # Cap context risk at 1.0
        context_risk = float(min(risk_score, 1.0))

        explanation = self._explain(context_signals, context_risk)

        return ContextResult(
            context_risk=context_risk,
            transaction_risk_multiplier=multiplier,
            privileged_workflow=privileged,
            high_value_transaction=high_value,
            unknown_caller=unknown_caller,
            fraud_history_flag=fraud_flag,
            fraud_registry_match=fraud_lookup.is_flagged,
            fraud_registry_detail=fraud_lookup.explanation,
            context_explanation=explanation,
            recommended_verification=verifications if verifications else ["Standard call verification"],
        )

    def _explain(self, signals: list, risk: float) -> str:
        if not signals:
            return "No elevated contextual risk factors detected."

        if risk > 0.6:
            severity = "Critical contextual risk"
        elif risk > 0.3:
            severity = "Elevated contextual risk"
        else:
            severity = "Moderate contextual risk"

        return f"{severity}: {'; '.join(signals)}."


context_engine = ContextEngine()
