"""CLASSIFY_INTENT: rule-based classifier (default), optional Gemini classifier, always-on advice guard."""
from __future__ import annotations

import re
from typing import Dict, List, Literal, Protocol, Tuple

from pydantic import BaseModel, Field

from .config import INTENTS, MIN_INTENT_SCORE, OTHER_CERTAINTY, STRONG_INTENT_SCORE
from .schemas import Classification
from .text import canonical


class IntentClassifier(Protocol):
    name: str

    def classify(self, text: str) -> Classification: ...


def _p(regex: str) -> re.Pattern:
    return re.compile(regex)


# (pattern, weight). Patterns run on canonical text (lowercased, "log in" -> "signin", "can't" -> "cannot").
# Strong, direction-specific words carry weight 3; shared words (card, balance, login) stay weak.
INTENT_PATTERNS: Dict[str, List[Tuple[re.Pattern, float]]] = {
    "deposit_issue": [
        (_p(r"\bdeposit\w*"), 3),
        (_p(r"\btopup\w*|\btopped up\b"), 3),
        (_p(r"\b(?:not|hasn't|haven't|isn't|didn't)\s+(?:\w+\s+)?(?:show|arriv|appear|reflect|credit)\w*.{0,40}\b(?:paid|pay\w*|transfer\w*|sent)\b"), 3),
        (_p(r"\b(?:paid|pay\w*|transfer\w*|sent)\b.{0,40}\b(?:not|hasn't|haven't|isn't|didn't)\s+(?:\w+\s+)?(?:show|arriv|appear|reflect|credit)\w*"), 3),
        (_p(r"\bshows? (?:zero|0|nothing)\b|\bnot credited\b"), 2),
        (_p(r"\bfund(?:s|ed|ing)?\b"), 1),
        (_p(r"\b(?:card|bank)\b"), 1),
        (_p(r"\bbalance\b"), 1),
        (_p(r"\bmoney\b"), 1),
    ],
    "password_reset": [
        (_p(r"\bforg[eo]t\w*\b.{0,20}\bpassword"), 3),
        (_p(r"\breset\w*"), 3),
        (_p(r"\bchange\w* (?:my )?password"), 2),
        (_p(r"\bpassword\w*"), 2),
        (_p(r"\bemail\w*"), 1),
    ],
    "withdrawal_issue": [
        (_p(r"\bwithdr[ae]w\w*"), 3),
        (_p(r"\bcashout\w*|\bpayout\w*"), 3),
        (_p(r"\b(?:declin|reject|refus|denied)\w*"), 1),
        (_p(r"\bpending\b"), 1),
        (_p(r"\bverif\w*"), 1),
    ],
    "account_lock": [
        (_p(r"\block(?:ed|out)?\b"), 3),
        (_p(r"\btoo many\b.{0,30}\b(?:attempt|tries|signin|password)\w*"), 3),
        (_p(r"\b(?:blocked|suspended|frozen|disabled)\b"), 2),
        (_p(r"\b(?:failed|wrong|incorrect)\b.{0,20}\b(?:attempt|password|signin)\w*"), 2),
        (_p(r"\bcannot (?:signin|access|get into)\b|\bget into my account\b"), 2),
        (_p(r"\bcool ?down\b"), 2),
        (_p(r"\battempts?\b"), 1),
        (_p(r"\bsignin\b"), 1),
    ],
    "trading_advice_request": [
        (_p(r"\bwhich (?:asset|stock|coin|crypto|currency|pair|market|share|commodit)\w*"), 3),
        (_p(r"\b(?:go|goes|going|will)\s+(?:up|down)\b|\bwill (?:rise|fall|increase|drop|pump)\b"), 3),
        (_p(r"\bshould i (?:buy|sell|invest|trade)\b"), 3),
        (_p(r"\b(?:predict|forecast)\w*"), 3),
        (_p(r"\b(?:make|making|earn|get|maximi[sz]e|so i can)\b.{0,15}\bprofit\w*|\bmake money\b|\bprofit guarantee\w*|\bguarantee\w* profit"), 3),
        (_p(r"\b(?:trading|trade|market|investment) (?:signal|tip)s?\b"), 3),
        (_p(r"\b(?:recommend\w*|best)\b.{0,25}\b(?:asset|stock|coin|crypto|investment)s?\b|\basset recommend\w*"), 3),
        (_p(r"\bprofit\w*"), 1),
        (_p(r"\b(?:market|invest\w*|trading|trade)\b"), 1),
        (_p(r"\btoday\b"), 1),
    ],
}

_LABEL_ORDER = {label: i for i, label in enumerate(INTENTS)}


def detect_advice(text: str) -> Tuple[bool, List[str]]:
    """Advice guard: strong trading-advice patterns only. Runs regardless of the active classifier."""
    canon = canonical(text)
    hits = [m.group(0) for pattern, weight in INTENT_PATTERNS["trading_advice_request"] if weight >= 3
            for m in [pattern.search(canon)] if m]
    return bool(hits), hits


class RuleClassifier:
    name = "rules"

    def classify(self, text: str) -> Classification:
        canon = canonical(text)
        scores: Dict[str, float] = {}
        matched: Dict[str, List[str]] = {}
        for label, patterns in INTENT_PATTERNS.items():
            total, terms = 0.0, []
            for pattern, weight in patterns:
                m = pattern.search(canon)
                if m:
                    total += weight
                    terms.append(m.group(0).strip())
            scores[label] = total
            if terms:
                matched[label] = terms

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], _LABEL_ORDER[kv[0]]))
        (top_label, top), (_, runner_up) = ranked[0], ranked[1]
        advice, _ = detect_advice(text)

        # Mixed ticket (real support issue + advice request): keep the actionable support intent;
        # the advice guard still flags it and the reply still carries a refusal.
        if advice and top_label == "trading_advice_request":
            support = [(lbl, s) for lbl, s in ranked if lbl != "trading_advice_request" and s >= MIN_INTENT_SCORE + 1]
            if support:
                (top_label, top), runner_up = support[0], scores["trading_advice_request"]

        if top < MIN_INTENT_SCORE:
            label, certainty = "other", OTHER_CERTAINTY
        else:
            label = top_label
            margin = (top / (top + runner_up) - 0.5) * 2  # 0 when tied, 1 when unopposed
            strength = min(1.0, top / STRONG_INTENT_SCORE)
            certainty = round(0.5 * margin + 0.5 * strength, 4)
        return Classification(
            label=label,
            certainty=certainty,
            scores=scores,
            matched_terms=matched,
            advice_detected=advice,
            component=self.name,
        )


class _LLMIntent(BaseModel):
    intent: Literal[
        "deposit_issue", "password_reset", "withdrawal_issue", "account_lock", "trading_advice_request", "other"
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


_CLASSIFY_PROMPT = """You classify customer support tickets for a digital platform with account, payments and trading help content.
Choose exactly one intent:
- deposit_issue: money paid in / deposit not showing or delayed
- password_reset: forgotten password, reset email or link problems
- withdrawal_issue: withdrawals / cash-outs declined, pending or delayed
- account_lock: account locked or blocked, e.g. after failed sign-in attempts
- trading_advice_request: asks for market predictions, asset recommendations or profit advice
- other: anything else

Return JSON with intent, confidence (0-1) and a one-sentence rationale.

Ticket: <<<{ticket}>>>"""


class GeminiClassifier:
    name = "gemini"

    def __init__(self, client):
        self.client = client

    def classify(self, text: str) -> Classification:
        data = self.client.generate_json(_CLASSIFY_PROMPT.format(ticket=text), _LLMIntent)
        parsed = _LLMIntent.model_validate(data)  # rejects labels outside the allowed set
        advice, _ = detect_advice(text)
        return Classification(
            label=parsed.intent,
            certainty=round(parsed.confidence, 4),
            scores={parsed.intent: parsed.confidence},
            matched_terms={},
            advice_detected=advice,
            component=self.name,
            rationale=parsed.rationale,
        )


class FallbackClassifier:
    """Runs the primary classifier; on any error uses the backup for that ticket and records why."""

    def __init__(self, primary: IntentClassifier, backup: IntentClassifier):
        self.primary, self.backup = primary, backup
        self.name = f"{primary.name}->{backup.name}"

    def classify(self, text: str) -> Classification:
        try:
            return self.primary.classify(text)
        except Exception as exc:
            result = self.backup.classify(text)
            result.fallback_reason = f"{self.primary.name} classifier failed ({type(exc).__name__}): {str(exc)[:160]}"
            return result
