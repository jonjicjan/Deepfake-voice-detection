"""VoiceShield AI — Alerts & Incidents Router"""
import uuid
from datetime import datetime
from fastapi import APIRouter
from db.database import get_incidents

router = APIRouter()

@router.get("/incidents")
async def list_incidents(limit: int = 50):
    """Retrieve recent security incidents (privacy-safe: feature metadata only, no raw audio)."""
    incidents = await get_incidents(limit)
    return {
        "incidents": incidents,
        "count": len(incidents),
        "privacy_note": "Incident log contains acoustic features only. No raw audio is stored.",
    }

@router.get("/alerts")
async def get_active_alerts():
    """Get current active alerts for the dashboard."""
    incidents = await get_incidents(10)
    high_risk = [i for i in incidents if i.get("risk_score", 0) >= 70]
    return {
        "active_alerts": len(high_risk),
        "recent_incidents": incidents[:5],
        "timestamp": datetime.utcnow().isoformat(),
    }

@router.post("/alerts/acknowledge/{incident_id}")
async def acknowledge_alert(incident_id: int):
    """Mark an alert as acknowledged by operator."""
    return {
        "incident_id": incident_id,
        "status": "acknowledged",
        "acknowledged_at": datetime.utcnow().isoformat(),
    }

@router.get("/stats")
async def get_stats():
    """Dashboard statistics overview."""
    incidents = await get_incidents(1000)
    total = len(incidents)
    high_risk = sum(1 for i in incidents if i.get("risk_score", 0) >= 70)
    blocked = sum(1 for i in incidents if i.get("transaction_blocked", 0))
    avg_score = sum(i.get("risk_score", 0) for i in incidents) / max(total, 1)
    return {
        "total_calls_analyzed": total,
        "high_risk_calls": high_risk,
        "transactions_blocked": blocked,
        "average_risk_score": round(avg_score, 1),
        "detection_rate": round(high_risk / max(total, 1) * 100, 1),
    }
