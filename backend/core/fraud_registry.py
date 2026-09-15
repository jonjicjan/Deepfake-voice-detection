"""Fraud history registry lookup — satisfies PS 'historical fraud indicators' requirement."""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "fraud_registry.json"


@dataclass
class FraudLookupResult:
    is_flagged: bool
    match_type: Optional[str]
    match_value: Optional[str]
    reason: Optional[str]
    prior_incidents: int
    explanation: str


def lookup_fraud(
    caller_id: Optional[str] = None,
    phone_number: Optional[str] = None,
    caller_name: Optional[str] = None,
    manual_flag: bool = False,
) -> FraudLookupResult:
    """Check caller against local fraud watchlist."""
    if manual_flag:
        return FraudLookupResult(
            is_flagged=True,
            match_type="manual_flag",
            match_value="historical_fraud_flag",
            reason="Caller marked with historical fraud indicator",
            prior_incidents=1,
            explanation="Historical fraud flag set on this call.",
        )

    if not REGISTRY_PATH.exists():
        return FraudLookupResult(False, None, None, None, 0, "Fraud registry not loaded.")

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    if caller_id and caller_id in registry.get("flagged_caller_ids", []):
        entry = next((e for e in registry.get("entries", []) if e["id"] == caller_id), None)
        return FraudLookupResult(
            is_flagged=True,
            match_type="caller_id",
            match_value=caller_id,
            reason=entry["reason"] if entry else "Listed in fraud registry",
            prior_incidents=entry.get("incidents", 1) if entry else 1,
            explanation=f"Caller ID '{caller_id}' found in fraud watchlist.",
        )

    if phone_number and phone_number in registry.get("flagged_phone_numbers", []):
        entry = next((e for e in registry.get("entries", []) if e.get("phone") == phone_number), None)
        return FraudLookupResult(
            is_flagged=True,
            match_type="phone_number",
            match_value=phone_number,
            reason=entry["reason"] if entry else "Listed in fraud registry",
            prior_incidents=entry.get("incidents", 1) if entry else 1,
            explanation=f"Phone '{phone_number}' found in fraud watchlist.",
        )

    if caller_name and caller_name in registry.get("flagged_names", []):
        return FraudLookupResult(
            is_flagged=True,
            match_type="caller_name",
            match_value=caller_name,
            reason="Name associated with prior impersonation attempts",
            prior_incidents=1,
            explanation=f"Caller name '{caller_name}' matches fraud registry.",
        )

    return FraudLookupResult(
        is_flagged=False,
        match_type=None,
        match_value=None,
        reason=None,
        prior_incidents=0,
        explanation="No fraud registry match.",
    )
