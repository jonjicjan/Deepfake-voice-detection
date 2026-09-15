"""
VoiceShield AI — Backend Entry Point
Production voice integrity verification platform for Cyber Security operations.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from routers import audio, speaker, risk, transaction, alerts, evaluation
from db.database import init_db
from core.deepfake_detector import preload_deepfake_model
from core.speaker_verifier import preload_speaker_model
from dotenv import load_dotenv
load_dotenv()

from core.config import get_settings
from core.security import (
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
    APIKeyMiddleware,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_db()
    print("[INFO] Preloading neural ML models...")
    deepfake_ok = preload_deepfake_model()
    speaker_ok = preload_speaker_model()
    print(f"[OK] Deepfake detector: {'loaded' if deepfake_ok else 'DSP fallback'}")
    print(f"[OK] Speaker verifier: {'loaded' if speaker_ok else 'MFCC fallback'}")
    if settings.auth_enabled:
        print("[OK] API key authentication: ENABLED")
    else:
        print("[WARN] API key authentication: DISABLED — set VOICESHIELD_API_KEY for production")
    print("[OK] VoiceShield AI backend ready.")
    yield
    print("[OFF] VoiceShield AI backend shutting down.")


settings = get_settings()

app = FastAPI(
    title="VoiceShield AI",
    description=(
        "Multi-layer voice integrity verification framework combining deepfake detection, "
        "speaker verification, prosody analysis, and contextual risk scoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(audio.router, prefix="/api", tags=["Audio Analysis"])
app.include_router(speaker.router, prefix="/api", tags=["Speaker Verification"])
app.include_router(risk.router, prefix="/api", tags=["Risk Engine"])
app.include_router(transaction.router, prefix="/api", tags=["Transaction Protection"])
app.include_router(alerts.router, prefix="/api", tags=["Alerts & Incidents"])
app.include_router(evaluation.router, prefix="/api", tags=["System Diagnostics"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "system": "VoiceShield AI",
        "version": "1.0.0",
        "status": "operational",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "service": "VoiceShield AI Backend"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
