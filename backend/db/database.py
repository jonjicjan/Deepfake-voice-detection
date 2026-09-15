"""
VoiceShield AI — Database Layer
Lightweight SQLite-based incident and session store.
Privacy-first: stores feature metadata only, never raw audio.
"""

import aiosqlite
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "voiceshield.db"


async def init_db():
    """Initialize the SQLite database with required tables."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                caller_id TEXT,
                caller_name TEXT,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                deepfake_score REAL,
                speaker_score REAL,
                prosody_score REAL,
                context_score REAL,
                attack_type TEXT,
                action_taken TEXT,
                transaction_amount REAL,
                transaction_blocked INTEGER DEFAULT 0,
                alert_sent INTEGER DEFAULT 0,
                notes TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS enrolled_speakers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker_id TEXT UNIQUE NOT NULL,
                speaker_name TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                enrolled_at TEXT NOT NULL,
                last_verified TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS call_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                caller_id TEXT,
                status TEXT DEFAULT 'active',
                final_risk_score REAL,
                chunks_analyzed INTEGER DEFAULT 0
            )
        """)
        await db.commit()
    from core.cross_session import init_cross_session_table
    await init_cross_session_table()
    print("[OK] Database initialized at:", DB_PATH)


async def log_incident(incident_data: dict) -> int:
    """Log a voice security incident (feature-only, no raw audio)."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("""
            INSERT INTO incidents (
                session_id, timestamp, caller_id, caller_name,
                risk_score, risk_level, deepfake_score, speaker_score,
                prosody_score, context_score, attack_type, action_taken,
                transaction_amount, transaction_blocked, alert_sent, notes
            ) VALUES (
                :session_id, :timestamp, :caller_id, :caller_name,
                :risk_score, :risk_level, :deepfake_score, :speaker_score,
                :prosody_score, :context_score, :attack_type, :action_taken,
                :transaction_amount, :transaction_blocked, :alert_sent, :notes
            )
        """, incident_data)
        await db.commit()
        return cursor.lastrowid


async def get_incidents(limit: int = 50) -> list:
    """Retrieve recent incidents."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM incidents ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def save_speaker_embedding(speaker_id: str, speaker_name: str, embedding: list):
    """Save speaker enrollment embedding (privacy: embedding vector only, no audio)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO enrolled_speakers
            (speaker_id, speaker_name, embedding_json, enrolled_at)
            VALUES (?, ?, ?, ?)
        """, (speaker_id, speaker_name, json.dumps(embedding), datetime.utcnow().isoformat()))
        await db.commit()


async def get_speaker_embedding(speaker_id: str) -> list | None:
    """Retrieve a speaker's enrollment embedding."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT embedding_json FROM enrolled_speakers WHERE speaker_id = ?",
            (speaker_id,)
        )
        row = await cursor.fetchone()
        if row:
            return json.loads(row["embedding_json"])
        return None
