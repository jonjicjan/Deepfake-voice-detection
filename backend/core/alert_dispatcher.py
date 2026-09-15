"""Multi-channel alert dispatcher — simulates SMS, email, supervisor notifications."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from core.policy_engine import PolicyDecision


@dataclass
class ChannelNotification:
    channel: str
    status: str
    recipient: str
    message: str
    timestamp: str


def _mask_phone(phone: str) -> str:
    if not phone or len(phone) < 6:
        return "+91-XXXXX"
    return phone[:6] + "XXXXX"


def dispatch_alerts(
    policy: PolicyDecision,
    session_id: str,
    risk_score: float,
    caller_name: Optional[str] = None,
    phone_number: Optional[str] = None,
    supervisor_email: str = "security@company.com",
) -> List[ChannelNotification]:
    """
    Simulate multi-channel alert delivery (UI + SMS + email + supervisor).
    No external gateways required — logs delivery for demo/SIH.
    """
    notifications: List[ChannelNotification] = []
    now = datetime.utcnow().isoformat()
    name = caller_name or "Unknown caller"
    masked = _mask_phone(phone_number or "+91-98XXX-XXXXX")

    for channel in policy.alert_channels:
        if channel == "ui":
            notifications.append(ChannelNotification(
                channel="ui",
                status="delivered",
                recipient="Dashboard",
                message=policy.alert_title,
                timestamp=now,
            ))
        elif channel == "sms":
            notifications.append(ChannelNotification(
                channel="sms",
                status="sent",
                recipient=masked,
                message=f"[VoiceShield] ALERT {policy.risk_level.value}: Risk {risk_score:.0f}/100 for call from {name}. Action: {policy.action.value}. Session {session_id[:8]}.",
                timestamp=now,
            ))
        elif channel == "email":
            notifications.append(ChannelNotification(
                channel="email",
                status="sent",
                recipient=supervisor_email,
                message=f"Voice integrity alert — {policy.alert_title}. {policy.alert_message[:120]}...",
                timestamp=now,
            ))
        elif channel == "supervisor":
            notifications.append(ChannelNotification(
                channel="supervisor",
                status="escalated",
                recipient="Compliance Officer",
                message=f"ESCALATION: Voice cloning suspected on session {session_id[:8]}. Immediate review required.",
                timestamp=now,
            ))

    if policy.secondary_verification_required:
        notifications.append(ChannelNotification(
            channel="mfa",
            status="prompted",
            recipient=masked,
            message=f"OTP/callback verification required: {', '.join(policy.verification_methods)}",
            timestamp=now,
        ))

    return notifications
