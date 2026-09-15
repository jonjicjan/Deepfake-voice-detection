"""Cross-session speaker consistency tracking across multiple calls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import aiosqlite

from db.database import DB_PATH

ROLLING_WINDOW = 5
DRIFT_THRESHOLD = 0.15


@dataclass
class CrossSessionResult:
    session_count: int
    rolling_avg_similarity: float
    current_similarity: float
    drift_score: float
    drift_detected: bool
    explanation: str
    adjusted_mismatch_boost: float


async def init_cross_session_table():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS speaker_session_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caller_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                recorded_at TEXT NOT NULL
            )
        """)
        await db.commit()


async def record_speaker_session(caller_id: str, session_id: str, similarity: float):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO speaker_session_history (caller_id, session_id, similarity_score, recorded_at) VALUES (?, ?, ?, ?)",
            (caller_id, session_id, similarity, datetime.utcnow().isoformat()),
        )
        await db.commit()


async def evaluate_cross_session(
    caller_id: str,
    current_similarity: float,
) -> CrossSessionResult:
    """Compare current call against rolling history of prior speaker-match scores."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT similarity_score FROM speaker_session_history
               WHERE caller_id = ? ORDER BY recorded_at DESC LIMIT ?""",
            (caller_id, ROLLING_WINDOW),
        )
        rows = await cursor.fetchall()

    prior = [r["similarity_score"] for r in rows]
    session_count = len(prior)

    if session_count == 0:
        return CrossSessionResult(
            session_count=0,
            rolling_avg_similarity=current_similarity,
            current_similarity=current_similarity,
            drift_score=0.0,
            drift_detected=False,
            explanation="First call in session history — baseline established.",
            adjusted_mismatch_boost=0.0,
        )

    rolling_avg = sum(prior) / session_count
    drift = float(max(0.0, rolling_avg - current_similarity))
    drift_detected = drift >= DRIFT_THRESHOLD

    if drift_detected:
        explanation = (
            f"Cross-session drift detected: current similarity {current_similarity:.2f} "
            f"vs rolling avg {rolling_avg:.2f} over {session_count} prior call(s)."
        )
        boost = min(0.4, drift * 1.5)
    else:
        explanation = (
            f"Speaker consistent across {session_count} prior call(s). "
            f"Rolling avg similarity: {rolling_avg:.2f}."
        )
        boost = 0.0

    return CrossSessionResult(
        session_count=session_count,
        rolling_avg_similarity=round(rolling_avg, 3),
        current_similarity=round(current_similarity, 3),
        drift_score=round(drift, 3),
        drift_detected=drift_detected,
        explanation=explanation,
        adjusted_mismatch_boost=boost,
    )
