"""Data contracts: input records, the required output schema, and internal stage results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import SAFETY_FLAGS

Intent = Literal[
    "deposit_issue",
    "password_reset",
    "withdrawal_issue",
    "account_lock",
    "trading_advice_request",
    "other",
]


# ---------- inputs ----------
class Ticket(BaseModel):
    id: str
    message: str
    language: str = "en"
    # Problems found while loading (malformed entry). Never written to results.json.
    input_errors: List[str] = Field(default_factory=list)


class Article(BaseModel):
    article_id: str = Field(min_length=1)
    title: str
    body: str = Field(min_length=1)


# ---------- required output ----------
class TicketResult(BaseModel):
    """Exactly the schema required per ticket. extra='forbid' keeps results.json clean."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(min_length=1)
    intent: Intent
    retrieved_articles: List[str] = Field(max_length=3)
    reply_draft: str = Field(min_length=1)
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_escalation: bool
    safety_flags: List[str]

    @field_validator("safety_flags")
    @classmethod
    def _known_flags(cls, flags: List[str]) -> List[str]:
        unknown = [f for f in flags if f not in SAFETY_FLAGS]
        if unknown:
            raise ValueError(f"unknown safety flags: {unknown}")
        return flags


RESULT_FIELDS = list(TicketResult.model_fields.keys())


# ---------- internal stage results ----------
@dataclass
class Classification:
    label: str
    certainty: float
    scores: Dict[str, float]
    matched_terms: Dict[str, List[str]]
    advice_detected: bool
    component: str
    fallback_reason: Optional[str] = None
    rationale: Optional[str] = None


@dataclass
class Hit:
    article_id: str
    score: float
    components: Dict[str, float]


@dataclass
class Draft:
    text: str
    cited_article_ids: List[str]
    component: str
    # Fixed sentences with no factual content (greeting, refusal framing); exempt from citation checks.
    boilerplate: List[str] = field(default_factory=list)
    fallback_reason: Optional[str] = None
