"""VoiceShield AI — Speaker Router"""
import uuid
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from core.preprocessor import preprocessor
from core.speaker_verifier import verifier
from db.database import save_speaker_embedding, get_speaker_embedding
from db.schemas import SpeakerVerifyResponse
import numpy as np

router = APIRouter()

@router.post("/enroll-speaker")
async def enroll_speaker(
    file: UploadFile = File(...),
    speaker_id: str = Form(...),
    speaker_name: str = Form(...),
):
    """Enroll a speaker's voice for future verification. Stores embedding only — no raw audio."""
    audio_bytes = await file.read()
    try:
        waveform, sr, duration = preprocessor.load_audio(audio_bytes)
        if duration < 3.0:
            raise HTTPException(status_code=400, detail="Please provide at least 3 seconds of audio for enrollment.")
        embedding = verifier.extract_embedding(waveform, sr)
        await save_speaker_embedding(speaker_id, speaker_name, embedding.tolist())
        return {
            "status": "enrolled",
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "embedding_dim": len(embedding),
            "audio_duration_s": round(duration, 2),
            "privacy_note": "Only acoustic embedding stored. Raw audio discarded.",
            "enrolled_at": datetime.utcnow().isoformat(),
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@router.post("/verify-speaker", response_model=SpeakerVerifyResponse)
async def verify_speaker(
    file: UploadFile = File(...),
    speaker_id: str = Form(...),
):
    """Verify if uploaded audio matches an enrolled speaker profile."""
    ref_embedding = await get_speaker_embedding(speaker_id)
    if not ref_embedding:
        raise HTTPException(status_code=404, detail=f"Speaker '{speaker_id}' not enrolled. Please enroll first.")
    audio_bytes = await file.read()
    try:
        waveform, sr, _ = preprocessor.load_audio(audio_bytes)
        result = verifier.verify_from_audio(waveform, sr, np.array(ref_embedding, dtype=np.float32))
        return SpeakerVerifyResponse(
            speaker_id=speaker_id,
            similarity_score=round(result.similarity_score, 4),
            verified=result.verified,
            confidence=result.confidence,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@router.get("/enrolled-speakers")
async def list_enrolled_speakers():
    """List all enrolled speaker profiles (metadata only)."""
    from db.database import DB_PATH
    import aiosqlite
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT speaker_id, speaker_name, enrolled_at FROM enrolled_speakers")
        rows = await cursor.fetchall()
    return {"enrolled_speakers": [dict(r) for r in rows], "count": len(rows)}
