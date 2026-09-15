"""VoiceShield AI — Risk, Transaction, and Alerts Routers"""
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from db.schemas import TransactionVerifyRequest, TransactionVerifyResponse, RiskLevel, ActionTaken, AlertResponse, PolicyConfig
from core.policy_engine import policy_engine
from db.database import get_incidents, log_incident

# --- Risk Router ---
router = APIRouter()

@router.get("/risk-score/{session_id}")
async def get_risk_score(session_id: str):
    """Get the latest risk score for an active session."""
    from db.database import DB_PATH
    import aiosqlite
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM incidents WHERE session_id = ? ORDER BY timestamp DESC LIMIT 1",
            (session_id,)
        )
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"No analysis found for session {session_id}")
    return dict(row)

@router.get("/policy")
async def get_policy():
    """Get current policy configuration."""
    return policy_engine.get_config()

@router.put("/policy")
async def update_policy(config: PolicyConfig):
    """Update policy thresholds (hot-update, no model retraining needed)."""
    policy_engine.update_config(config.model_dump())
    return {"status": "updated", "config": policy_engine.get_config()}
