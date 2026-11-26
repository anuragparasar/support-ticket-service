"""Confidence, safety flags and escalation. Computed from evidence, not from a model's self-report."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .config import (
    CONFIDENCE_WEIGHTS,
    CONFLICT_PENALTY,
    ESCALATING_FLAGS,
    FALLBACK_PENALTY,
    LOW_CONFIDENCE_THRESHOLD,
    NO_HIT_CONFIDENCE_CAP,
    POLICY_SENSITIVE_INTENTS,
    RETRIEVAL_SCORE_CAP,
    SAFETY_FLAGS,
    SUPPORTED_LANGUAGES,
)
from .schemas import Classification, Hit, Ticket
from .text import canonical
from .validator import ContentCheck

ACCOUNT_SPECIFIC = re.compile(
    r"\bmy (?:balance|withdraw\w*|deposit\w*|payment\w*|transaction\w*|funds?|money|transfer\w*|order\w*)\b")


def confidence(cls: Classification, hits: List[Hit], check: ContentCheck, tags: Dict[str, str],
               fallback_used: bool) -> Tuple[float, Dict]:
    retrieval = min(1.0, hits[0].score / RETRIEVAL_SCORE_CAP) if hits else 0.0
    w = CONFIDENCE_WEIGHTS
    raw = w["classifier"] * cls.certainty + w["retrieval"] * retrieval + w["grounding"] * check.grounding_score
    penalties: Dict[str, float] = {}
    if hits and cls.label != "other" and tags.get(hits[0].article_id) != cls.label:
        penalties["conflict: top article topic != intent"] = CONFLICT_PENALTY
    if fallback_used:
        penalties["fallback component used"] = FALLBACK_PENALTY
    value = raw - sum(penalties.values())
    if not hits:
        value = min(value, NO_HIT_CONFIDENCE_CAP)
        penalties["no relevant article (capped)"] = 0.0
    value = round(max(0.0, min(1.0, value)), 4)
    breakdown = {
        "classifier_certainty": round(cls.certainty, 4),
        "retrieval_strength": round(retrieval, 4),
        "grounding_score": round(check.grounding_score, 4),
        "penalties": penalties,
        "final": value,
    }
    return value, breakdown


def flags_and_escalation(ticket: Ticket, cls: Classification, hits: List[Hit], check: ContentCheck,
                         conf: float, extra_flags: List[str]) -> Tuple[List[str], bool, List[str]]:
    flags = set(extra_flags)
    reasons: List[str] = []
    if cls.advice_detected or cls.label == "trading_advice_request":
        flags.add("advice_request")
    if not check.grounded:
        flags.add("insufficient_grounding")
        reasons.append("reply could not be fully grounded in retrieved articles"
                       if hits else "no relevant help-centre article found")
    if cls.label in POLICY_SENSITIVE_INTENTS:
        flags.add("policy_sensitive")
        reasons.append(f"'{cls.label}' outcomes depend on account / compliance review")
    if conf < LOW_CONFIDENCE_THRESHOLD:
        flags.add("low_confidence")
        reasons.append(f"confidence {conf:.2f} below {LOW_CONFIDENCE_THRESHOLD}")
    if ACCOUNT_SPECIFIC.search(canonical(ticket.message)):
        flags.add("account_specific_request")
    if check.has_claim_issue:
        flags.add("unsupported_claim")
        reasons.append("draft contained a promise or account-specific claim")
    if ticket.input_errors:
        flags.add("invalid_input")
        reasons.append(f"input problems: {', '.join(ticket.input_errors)}")
    if ticket.language not in SUPPORTED_LANGUAGES:
        flags.add("unsupported_language")
        reasons.append(f"language '{ticket.language}' not supported")
    if cls.label == "other":
        reasons.append("intent not covered by supported categories")

    escalate = bool(flags & ESCALATING_FLAGS) or cls.label == "other"
    ordered = [f for f in SAFETY_FLAGS if f in flags]
    if not escalate:
        reasons = ["handled safely by policy refusal" if "advice_request" in flags
                   else "confident, grounded and not policy-sensitive"]
    return ordered, escalate, list(dict.fromkeys(reasons))
