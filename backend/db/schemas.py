"""
VoiceShield AI — Pydantic Schemas
All API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AttackType(str, Enum):
    TTS = "TTS_SYNTHESIZED"
    VOICE_CONVERSION = "VOICE_CONVERSION"
    REPLAY = "REPLAY_ATTACK"
    GENUINE = "GENUINE"
    UNKNOWN_SYNTHETIC = "UNKNOWN_SYNTHETIC"


class ActionTaken(str, Enum):
    ALLOWED = "ALLOWED"
    STEP_UP_VERIFICATION = "STEP_UP_VERIFICATION"
    BLOCKED = "BLOCKED"
    ESCALATED = "ESCALATED"


class TransactionContext(BaseModel):
    amount: Optional[float] = Field(None, description="Transaction amount in INR")
    transaction_type: Optional[str] = Field(None, description="FUND_TRANSFER, DATA_ACCESS, APPROVAL")
    recipient: Optional[str] = None
    is_privileged_workflow: bool = False


class CallerContext(BaseModel):
    caller_id: Optional[str] = None
    caller_name: Optional[str] = None
    phone_number: Optional[str] = None
    is_known_contact: bool = False
    historical_fraud_flag: bool = False
    call_origin: Optional[str] = Field(None, description="VOIP, MOBILE, LANDLINE, UNKNOWN")
    claimed_identity: Optional[str] = None


class AudioAnalysisRequest(BaseModel):
    session_id: Optional[str] = None
    caller_context: Optional[CallerContext] = None
    transaction_context: Optional[TransactionContext] = None
    language_hint: Optional[str] = Field(None, description="hi, ta, te, kn, en-IN")


class ComponentScores(BaseModel):
    deepfake_probability: float = Field(..., ge=0, le=1, description="0=genuine, 1=deepfake")
    speaker_mismatch: float = Field(..., ge=0, le=1, description="0=match, 1=mismatch")
    prosody_anomaly: float = Field(..., ge=0, le=1, description="0=natural, 1=synthetic")
    replay_probability: float = Field(..., ge=0, le=1)
    unknown_attack_score: float = Field(..., ge=0, le=1)
    context_risk: float = Field(..., ge=0, le=1)


class ChannelNotificationOut(BaseModel):
    channel: str
    status: str
    recipient: str
    message: str
    timestamp: str


class CrossSessionInfo(BaseModel):
    session_count: int
    rolling_avg_similarity: float
    current_similarity: float
    drift_score: float
    drift_detected: bool
    explanation: str


class AudioAnalysisResponse(BaseModel):
    session_id: str
    risk_score: float = Field(..., ge=0, le=100, description="Dynamic impersonation risk score")
    risk_level: RiskLevel
    attack_type: AttackType
    component_scores: ComponentScores
    action_recommended: ActionTaken
    alert_message: str
    explanation: str
    secondary_verification_required: bool
    transaction_blocked: bool
    processing_time_ms: float
    audio_duration_s: Optional[float] = None
    language_detected: Optional[str] = None
    models_used: Optional[str] = Field(None, description="Neural models used in analysis")
    fraud_registry_match: bool = False
    fraud_registry_detail: Optional[str] = None
    cross_session: Optional[CrossSessionInfo] = None
    channel_notifications: List[ChannelNotificationOut] = []
    verification_methods: List[str] = []


class SpeakerEnrollRequest(BaseModel):
    speaker_id: str
    speaker_name: str


class SpeakerVerifyResponse(BaseModel):
    speaker_id: str
    similarity_score: float
    verified: bool
    confidence: str


class TransactionVerifyRequest(BaseModel):
    session_id: str
    transaction_context: TransactionContext
    current_risk_score: float


class TransactionVerifyResponse(BaseModel):
    session_id: str
    transaction_allowed: bool
    risk_level: RiskLevel
    action: ActionTaken
    message: str
    verification_methods: List[str]


class IncidentRecord(BaseModel):
    id: int
    session_id: str
    timestamp: str
    caller_name: Optional[str]
    risk_score: float
    risk_level: str
    attack_type: Optional[str]
    action_taken: Optional[str]
    transaction_amount: Optional[float]
    transaction_blocked: bool
    alert_sent: bool


class AlertResponse(BaseModel):
    session_id: str
    alert_id: str
    severity: RiskLevel
    title: str
    message: str
    recommended_actions: List[str]
    channels_notified: List[str]
    timestamp: str


class PolicyConfig(BaseModel):
    low_threshold: float = Field(30.0, description="Risk score below this = LOW")
    high_threshold: float = Field(70.0, description="Risk score above this = HIGH")
    critical_threshold: float = Field(85.0, description="Risk score above this = CRITICAL")
    auto_block_on_high: bool = True
    require_mfa_on_medium: bool = True
    alert_channels: List[str] = ["ui", "email"]
    high_value_transaction_limit: float = Field(50000.0, description="INR amount triggering elevated checks")
