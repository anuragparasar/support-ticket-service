"""VALIDATE_OUTPUT: content checks on a draft (grounding, refusals, claims) and schema checks on a result."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set

from pydantic import ValidationError

from .config import MIN_SENTENCE_SUPPORT, SUBSTANTIVE_MIN_TOKENS
from .schemas import RESULT_FIELDS, Article, Draft, Hit, TicketResult
from .text import canonical, citations, split_sentences, strip_citations, tokens

REFUSAL_MARKER = re.compile(
    r"\b(?:cannot|can't|unable to|not able to|don't|do not|won't|will not)\b[^.]{0,40}\b(?:provide|give|offer|share|recommend|predict)\w*",
    re.I)
RECOMMENDATION = re.compile(
    r"\byou should (?:buy|sell|invest|trade)\b|\bi (?:would )?recommend (?:buying|selling|investing)\b|"
    r"\b(?:is|are) (?:likely|expected|going) to (?:go up|rise|increase|fall|drop)\b|\bbest (?:asset|stock|coin)s? to\b|"
    r"\bwill (?:go up|rise|increase|fall|drop)\b", re.I)
PROMISE = re.compile(
    r"\bwill be (?:approved|processed|credited|refunded|released|unlocked|resolved|reversed)\b|"
    r"\b(?:we|i) (?:can )?guarantee\b|\bguaranteed\b|\bwithin \d+\s*(?:minutes?|hours?|days?|business days?)\b|"
    r"\b(?:refund|payment|withdrawal|deposit)s? will (?:be|arrive|reach|go)\b|\bi promise\b", re.I)
ACCOUNT_FACT = re.compile(
    r"\byour (?:account|balance|withdrawal|deposit|payment|transaction)s? (?:is|was|has been|shows|appears)\b|"
    r"\bwe (?:can )?see (?:that )?your\b|\bi (?:can )?see (?:that )?your\b|\bour records show\b", re.I)


@dataclass
class ContentCheck:
    grounded: bool
    grounding_score: float
    hard_issues: List[str] = field(default_factory=list)   # must be repaired before release
    unsupported_sentences: List[str] = field(default_factory=list)
    sentence_support: List[Dict] = field(default_factory=list)

    @property
    def has_claim_issue(self) -> bool:
        return any(i.startswith("unsupported_claim") for i in self.hard_issues)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_MARKER.search(text)) and not RECOMMENDATION.search(text)


def check_content(draft: Draft, hits: List[Hit], articles: Dict[str, Article], advice_required: bool) -> ContentCheck:
    retrieved = {h.article_id for h in hits}
    issues: List[str] = []
    cited_all = set(citations(draft.text)) | set(draft.cited_article_ids)
    unknown = sorted(cited_all - retrieved)
    if unknown:
        issues.append(f"cites non-retrieved articles: {unknown}")

    boiler = {b.strip() for b in draft.boilerplate}
    support_rows, unsupported = [], []
    grounded_count = substantive = 0
    for sentence in split_sentences(draft.text):
        plain = strip_citations(sentence).replace(" .", ".").strip()
        if sentence.strip() in boiler or plain in boiler:
            continue
        cited = [c for c in citations(sentence) if c in retrieved]
        toks = tokens(plain)
        if not cited and len(toks) < SUBSTANTIVE_MIN_TOKENS:
            continue  # greeting / filler, no factual content
        substantive += 1
        source: Set[str] = set()
        for cid in cited:
            source |= set(tokens(f"{articles[cid].title} {articles[cid].body}"))
        support = (len([t for t in toks if t in source]) / len(toks)) if toks and cited else 0.0
        ok = bool(cited) and support >= MIN_SENTENCE_SUPPORT
        support_rows.append({"sentence": plain, "cited": cited, "support": round(support, 3), "ok": ok})
        if ok:
            grounded_count += 1
        else:
            unsupported.append(plain)

    text = canonical(draft.text)
    for name, pattern in (("promise", PROMISE), ("account fact", ACCOUNT_FACT)):
        m = pattern.search(text)
        if m:
            issues.append(f"unsupported_claim: {name} '{m.group(0)}'")
    if advice_required and not is_refusal(draft.text):
        issues.append("advice request not refused")

    grounding_score = grounded_count / substantive if substantive else 0.0
    grounded = grounded_count > 0 and not unsupported and not unknown
    return ContentCheck(grounded, round(grounding_score, 4), issues, unsupported, support_rows)


def validate_result(result: dict, kb_ids: Iterable[str]) -> List[str]:
    """Schema + cross-field checks on one final result. Returns a list of problems (empty = valid)."""
    errors: List[str] = []
    if not isinstance(result, dict):
        return ["result is not an object"]
    missing = [f for f in RESULT_FIELDS if f not in result]
    extra = [f for f in result if f not in RESULT_FIELDS]
    if missing:
        errors.append(f"missing fields: {missing}")
    if extra:
        errors.append(f"unexpected fields: {extra}")
    try:
        TicketResult.model_validate(result)
    except ValidationError as exc:
        errors.extend(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
    known = set(kb_ids)
    articles = result.get("retrieved_articles") or []
    if isinstance(articles, list):
        bad = [a for a in articles if a not in known]
        if bad:
            errors.append(f"unknown article ids: {bad}")
        if len(set(articles)) != len(articles):
            errors.append("duplicate article ids")
    flags = result.get("safety_flags") or []
    if result.get("intent") == "trading_advice_request" or "advice_request" in flags:
        if not is_refusal(str(result.get("reply_draft", ""))):
            errors.append("trading advice request is not refused")
        if "advice_request" not in flags:
            errors.append("advice request missing advice_request flag")
    if result.get("grounded") is False and "insufficient_grounding" not in flags:
        errors.append("ungrounded reply without insufficient_grounding flag")
    if "unsupported_claim" in flags and not result.get("needs_human_escalation"):
        errors.append("unsupported claim not escalated")
    return errors
