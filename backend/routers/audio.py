"""
VoiceShield AI — Audio Analysis Router
Main endpoints for audio ingestion and real-time analysis.

Endpoints:
  POST /api/analyze-audio  — Upload audio file for full analysis
  WS   /api/ws/stream      — WebSocket for real-time streaming analysis
"""

import uuid
import time
import asyncio
import json
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import Optional

from core.config import get_settings
from core.security import verify_ws_api_key

from core.preprocessor import preprocessor
from core.analysis_pipeline import run_parallel_layers
from core.deepfake_detector import detector, get_runtime_info
from core.speaker_verifier import verifier
from core.prosody_analyzer import analyzer
from core.attack_classifier import classifier
from core.context_engine import context_engine
from core.risk_engine import risk_engine
from core.policy_engine import policy_engine
from core.alert_dispatcher import dispatch_alerts
from core.cross_session import evaluate_cross_session, record_speaker_session
from db.database import log_incident, get_speaker_embedding
from db.schemas import (
    AudioAnalysisResponse, ComponentScores, AttackType,
    CallerContext, TransactionContext, CrossSessionInfo,
)

router = APIRouter()


@router.post("/analyze-audio", response_model=AudioAnalysisResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    caller_id: Optional[str] = Form(None),
    caller_name: Optional[str] = Form(None),
    phone_number: Optional[str] = Form(None),
    is_known_contact: bool = Form(False),
    historical_fraud_flag: bool = Form(False),
    call_origin: Optional[str] = Form("UNKNOWN"),
    claimed_identity: Optional[str] = Form(None),
    transaction_amount: Optional[float] = Form(None),
    transaction_type: Optional[str] = Form(None),
    is_privileged_workflow: bool = Form(False),
    language_hint: Optional[str] = Form(None),
):
    """
    Full voice integrity analysis pipeline.

    Processes uploaded audio through:
    1. Preprocessing (VAD, normalization)
    2. Deepfake detection (spectral/phase analysis)
    3. Speaker verification (if enrolled profile exists)
    4. Prosody analysis (pitch/rhythm/pause)
    5. Attack classification (TTS/VC/Replay/Unknown)
    6. Context enrichment (caller/transaction metadata)
    7. Risk scoring (0–100)
    8. Policy evaluation (ALLOW/BLOCK/ESCALATE)
    9. Incident logging (privacy-safe: features only)
    """
    start_time = time.time()

    if not session_id:
        session_id = str(uuid.uuid4())

    # --- 1. Load and preprocess audio ---
    settings = get_settings()
    audio_bytes = await file.read()
    if len(audio_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum upload size is {settings.max_upload_bytes // (1024 * 1024)} MB.",
        )
    if len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Audio file too small or empty.")

    allowed_types = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/flac", "audio/mp4", "audio/webm", "application/octet-stream"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail=f"Unsupported audio format: {file.content_type}")

    try:
        waveform, sr, duration = preprocessor.load_audio(audio_bytes, file.filename or "")
        waveform = preprocessor.apply_vad(waveform, sr)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # --- 2–6. Parallel ML layers (deepfake ∥ prosody, then attack + speaker) ---
    import numpy as np
    ref_arr = None
    if caller_id:
        ref_embedding = await get_speaker_embedding(caller_id)
        if ref_embedding:
            ref_arr = np.array(ref_embedding, dtype=np.float32)

    layers = await asyncio.to_thread(run_parallel_layers, waveform, sr, ref_arr)
    deepfake_result = layers.deepfake_result
    prosody_result = layers.prosody_result
    attack_result = layers.attack_result
    speaker_mismatch = layers.speaker_mismatch
    speaker_similarity = layers.speaker_similarity
    speaker_enrolled = layers.speaker_enrolled
    duration = layers.analysis_duration_s

    cross_session_info = None
    if caller_id and speaker_enrolled:
        cross_session_info = await evaluate_cross_session(caller_id, speaker_similarity)
        speaker_mismatch = float(min(1.0, speaker_mismatch + cross_session_info.adjusted_mismatch_boost))
        await record_speaker_session(caller_id, session_id, speaker_similarity)

    # --- 7. Context enrichment ---
    caller_ctx = CallerContext(
        caller_id=caller_id,
        caller_name=caller_name,
        phone_number=phone_number,
        is_known_contact=is_known_contact,
        historical_fraud_flag=historical_fraud_flag,
        call_origin=call_origin,
        claimed_identity=claimed_identity,
    )
    txn_ctx = None
    if transaction_amount is not None or transaction_type is not None:
        txn_ctx = TransactionContext(
            amount=transaction_amount,
            transaction_type=transaction_type,
            is_privileged_workflow=is_privileged_workflow,
        )

    context_result = context_engine.evaluate(caller_ctx, txn_ctx)

    # --- 8. Risk scoring ---
    risk_result = risk_engine.compute(
        deepfake_probability=deepfake_result.probability,
        speaker_mismatch=speaker_mismatch,
        prosody_anomaly=prosody_result.anomaly_probability,
        replay_probability=attack_result.replay_probability,
        unknown_attack_score=attack_result.unknown_attack_score,
        context_risk=context_result.context_risk,
        transaction_risk_multiplier=context_result.transaction_risk_multiplier,
        speaker_enrolled=speaker_enrolled,
    )

    # --- 9. Policy evaluation ---
    policy_decision = policy_engine.evaluate(risk_result.score, risk_result.level)

    channel_notifications = dispatch_alerts(
        policy_decision, session_id, risk_result.score, caller_name, phone_number
    )

    processing_ms = (time.time() - start_time) * 1000

    # --- 10. Privacy-safe incident logging ---
    await log_incident({
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "caller_id": caller_id,
        "caller_name": caller_name,
        "risk_score": risk_result.score,
        "risk_level": risk_result.level,
        "deepfake_score": deepfake_result.probability,
        "speaker_score": speaker_mismatch,
        "prosody_score": prosody_result.anomaly_probability,
        "context_score": context_result.context_risk,
        "attack_type": attack_result.predicted_class,
        "action_taken": policy_decision.action.value,
        "transaction_amount": transaction_amount,
        "transaction_blocked": int(policy_decision.block_transaction),
        "alert_sent": int(len(policy_decision.alert_channels) > 0),
        "notes": deepfake_result.explanation[:200] if deepfake_result.explanation else "",
    })

    # --- Build response ---
    component_scores = ComponentScores(
        deepfake_probability=round(deepfake_result.probability, 3),
        speaker_mismatch=round(speaker_mismatch, 3),
        prosody_anomaly=round(prosody_result.anomaly_probability, 3),
        replay_probability=round(attack_result.replay_probability, 3),
        unknown_attack_score=round(attack_result.unknown_attack_score, 3),
        context_risk=round(context_result.context_risk, 3),
    )

    # Map attack class string to enum
    attack_map = {
        "TTS_SYNTHESIZED": AttackType.TTS,
        "VOICE_CONVERSION": AttackType.VOICE_CONVERSION,
        "REPLAY_ATTACK": AttackType.REPLAY,
        "GENUINE": AttackType.GENUINE,
        "UNKNOWN_SYNTHETIC": AttackType.UNKNOWN_SYNTHETIC,
    }

    return AudioAnalysisResponse(
        session_id=session_id,
        risk_score=risk_result.score,
        risk_level=risk_result.level,
        attack_type=attack_map.get(attack_result.predicted_class, AttackType.UNKNOWN_SYNTHETIC),
        component_scores=component_scores,
        action_recommended=policy_decision.action,
        alert_message=policy_decision.alert_message,
        explanation=risk_result.explanation,
        secondary_verification_required=policy_decision.secondary_verification_required,
        transaction_blocked=policy_decision.block_transaction,
        processing_time_ms=round(processing_ms, 2),
        audio_duration_s=round(duration, 2),
        language_detected=language_hint,
        models_used=deepfake_result.model_used,
        fraud_registry_match=context_result.fraud_registry_match,
        fraud_registry_detail=context_result.fraud_registry_detail,
        cross_session=CrossSessionInfo(
            session_count=cross_session_info.session_count,
            rolling_avg_similarity=cross_session_info.rolling_avg_similarity,
            current_similarity=cross_session_info.current_similarity,
            drift_score=cross_session_info.drift_score,
            drift_detected=cross_session_info.drift_detected,
            explanation=cross_session_info.explanation,
        ) if cross_session_info else None,
        channel_notifications=[
            {"channel": n.channel, "status": n.status, "recipient": n.recipient, "message": n.message, "timestamp": n.timestamp}
            for n in channel_notifications
        ],
        verification_methods=policy_decision.verification_methods,
    )


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, api_key: Optional[str] = Query(None)):
    """
    WebSocket endpoint for real-time streaming audio analysis.
    Protocol:
      1. Client sends JSON config: {"type":"stream_config","caller_id":"...","caller_name":"..."}
      2. Client streams binary PCM int16 mono 16kHz chunks
      3. Server responds with risk_update every ~3 seconds of audio
    """
    if not verify_ws_api_key(api_key):
        await websocket.close(code=1008, reason="Unauthorized")
        return
    await websocket.accept()
    session_id = str(uuid.uuid4())
    chunk_buffer = bytearray()
    chunk_count = 0
    stream_config = {
        "caller_id": None,
        "caller_name": None,
        "call_origin": "LIVE_STREAM",
        "is_known_contact": False,
        "historical_fraud_flag": False,
        "transaction_amount": None,
        "transaction_type": None,
    }
    # ~3 seconds of int16 PCM at 16kHz mono
    PCM_CHUNK_BYTES = 16000 * 2 * 3

    async def process_buffer():
        nonlocal chunk_count
        if len(chunk_buffer) < PCM_CHUNK_BYTES:
            return

        audio_bytes = bytes(chunk_buffer[:PCM_CHUNK_BYTES])
        chunk_buffer[:] = chunk_buffer[PCM_CHUNK_BYTES:]
        chunk_count += 1

        try:
            import numpy as np
            waveform = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sr = 16000

            if len(waveform) / sr < 0.5:
                return

            ref_arr = None
            if stream_config.get("caller_id"):
                ref_embedding = await get_speaker_embedding(stream_config["caller_id"])
                if ref_embedding:
                    ref_arr = np.array(ref_embedding, dtype=np.float32)

            layers = await asyncio.to_thread(run_parallel_layers, waveform, sr, ref_arr)
            deepfake_result = layers.deepfake_result
            prosody_result = layers.prosody_result
            attack_result = layers.attack_result
            speaker_mismatch = layers.speaker_mismatch
            speaker_enrolled = layers.speaker_enrolled

            caller_ctx = CallerContext(
                caller_id=stream_config.get("caller_id"),
                caller_name=stream_config.get("caller_name"),
                is_known_contact=stream_config.get("is_known_contact", False),
                historical_fraud_flag=stream_config.get("historical_fraud_flag", False),
                call_origin=stream_config.get("call_origin", "LIVE_STREAM"),
            )
            txn_ctx = None
            if stream_config.get("transaction_amount") is not None:
                txn_ctx = TransactionContext(
                    amount=stream_config.get("transaction_amount"),
                    transaction_type=stream_config.get("transaction_type"),
                )
            context_result = context_engine.evaluate(caller_ctx, txn_ctx)

            risk_result = risk_engine.compute(
                deepfake_probability=deepfake_result.probability,
                speaker_mismatch=speaker_mismatch,
                prosody_anomaly=prosody_result.anomaly_probability,
                replay_probability=attack_result.replay_probability,
                unknown_attack_score=attack_result.unknown_attack_score,
                context_risk=context_result.context_risk,
                transaction_risk_multiplier=context_result.transaction_risk_multiplier,
                speaker_enrolled=speaker_enrolled,
            )
            policy_decision = policy_engine.evaluate(risk_result.score, risk_result.level)

            await websocket.send_json({
                "type": "risk_update",
                "session_id": session_id,
                "chunk": chunk_count,
                "risk_score": risk_result.score,
                "risk_level": risk_result.level,
                "attack_type": attack_result.predicted_class,
                "deepfake_probability": round(deepfake_result.probability, 3),
                "speaker_mismatch": round(speaker_mismatch, 3),
                "prosody_anomaly": round(prosody_result.anomaly_probability, 3),
                "replay_probability": round(attack_result.replay_probability, 3),
                "context_risk": round(context_result.context_risk, 3),
                "label": deepfake_result.label,
                "models_used": deepfake_result.model_used,
                "action_recommended": policy_decision.action.value,
                "transaction_blocked": policy_decision.block_transaction,
                "alert_message": policy_decision.alert_message,
                "explanation": risk_result.explanation,
                "component_scores": {
                    "deepfake_probability": round(deepfake_result.probability, 3),
                    "speaker_mismatch": round(speaker_mismatch, 3),
                    "prosody_anomaly": round(prosody_result.anomaly_probability, 3),
                    "replay_probability": round(attack_result.replay_probability, 3),
                    "unknown_attack_score": round(attack_result.unknown_attack_score, 3),
                    "context_risk": round(context_result.context_risk, 3),
                },
                "timestamp": datetime.utcnow().isoformat(),
            })
        except Exception as e:
            await websocket.send_json({
                "type": "analysis_error",
                "message": str(e),
                "chunk": chunk_count,
            })

    try:
        await websocket.send_json({
            "type": "session_start",
            "session_id": session_id,
            "message": "VoiceShield AI stream connected. Monitoring for voice cloning attacks.",
        })

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "stream_config":
                        stream_config.update({
                            k: payload[k] for k in stream_config if k in payload
                        })
                        await websocket.send_json({
                            "type": "config_ack",
                            "caller_id": stream_config.get("caller_id"),
                        })
                except json.JSONDecodeError:
                    pass
                continue

            if "bytes" in message and message["bytes"]:
                chunk_buffer.extend(message["bytes"])
                await process_buffer()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
