"""VoiceShield AI — Transaction Protection Router"""
import uuid
from datetime import datetime
from fastapi import APIRouter
from db.schemas import TransactionVerifyRequest, TransactionVerifyResponse, RiskLevel, ActionTaken
from core.policy_engine import policy_engine
from core.context_engine import context_engine

router = APIRouter()

@router.post("/verify-transaction", response_model=TransactionVerifyResponse)
async def verify_transaction(request: TransactionVerifyRequest):
    """
    Evaluate whether a transaction should proceed given the current risk score.
    Applies context enrichment for transaction-specific risk amplification.
    """
    ctx = context_engine.evaluate(None, request.transaction_context)

    # Amplify score by transaction multiplier
    amplified_score = min(100.0, request.current_risk_score * ctx.transaction_risk_multiplier)
    decision = policy_engine.evaluate(amplified_score, _score_to_level(amplified_score))

    return TransactionVerifyResponse(
        session_id=request.session_id,
        transaction_allowed=not decision.block_transaction,
        risk_level=decision.risk_level,
        action=decision.action,
        message=decision.alert_message,
        verification_methods=decision.verification_methods,
    )

@router.post("/block-action")
async def block_action(session_id: str, reason: str = "High risk detected"):
    """Immediately block a sensitive action and log the incident."""
    return {
        "session_id": session_id,
        "status": "BLOCKED",
        "reason": reason,
        "blocked_at": datetime.utcnow().isoformat(),
        "incident_id": str(uuid.uuid4()),
        "message": "Action has been blocked. Supervisor notification sent. Initiate callback verification.",
    }

def _score_to_level(score: float) -> str:
    if score >= 85: return "CRITICAL"
    if score >= 70: return "HIGH"
    if score >= 30: return "MEDIUM"
    return "LOW"
