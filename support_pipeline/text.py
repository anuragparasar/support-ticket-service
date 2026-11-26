"""Shared, deterministic text normalisation used by the classifier, retriever and grounding checks."""
from __future__ import annotations

import re
from typing import List

STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be",
    "been", "before", "but", "by", "can", "could", "did", "do", "does", "doing", "for", "from", "get",
    "got", "had", "has", "have", "having", "hello", "hi", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "me", "my", "now", "of", "on", "or", "our", "please", "should", "so", "some", "still",
    "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "up",
    "us", "was", "we", "were", "what", "when", "where", "which", "while", "who", "why", "will", "with",
    "would", "you", "your", "not", "no", "dont", "don", "t", "s", "im", "ive", "tell", "thanks", "thank",
    "may", "might", "must", "usually", "two", "few", "hours", "hour", "ago", "today", "today's",
}

# Canonical forms for multi-word variants, applied before tokenising.
_CANONICAL = [
    (re.compile(r"[‘’`]"), "'"),
    (re.compile(r"\b(?:sign|log)[\s-]?(?:in|on)\b|\blogin\b|\blogon\b"), "signin"),
    (re.compile(r"\btop[\s-]?up\b"), "topup"),
    (re.compile(r"\bcash[\s-]?out\b"), "cashout"),
    (re.compile(r"\bcan ?not\b|\bcan't\b"), "cannot"),
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Post-stem aliases so word families collapse to one token.
TOKEN_ALIASES = {
    "withdrawal": "withdraw", "withdrew": "withdraw", "payout": "withdraw", "cashout": "withdraw",
    "verification": "verify", "verifi": "verify", "verif": "verify",
    "prediction": "predict", "recommendation": "recommend",
    "locked": "lock", "lockout": "lock",
    "pwd": "password",
}

# Query-side expansion (stemmed token -> extra tokens). Bridges paraphrases to KB vocabulary.
QUERY_EXPANSIONS = {
    "money": ["deposit", "balance", "payment"],
    "paid": ["payment", "deposit"],
    "pay": ["payment"],
    "fund": ["deposit", "balance"],
    "topup": ["deposit"],
    "credit": ["deposit", "balance"],
    "block": ["lock"],
    "frozen": ["lock"],
    "freez": ["lock"],
    "suspend": ["lock", "restriction"],
    "hack": ["security", "signin"],
    "wrong": ["fail"],
    "incorrect": ["fail"],
    "access": ["signin"],
    "buy": ["asset", "market", "recommend"],
    "sell": ["asset", "market", "recommend"],
    "invest": ["asset", "market", "recommend"],
    "coin": ["asset", "market"],
    "crypto": ["asset", "market"],
    "bitcoin": ["asset", "market"],
    "gold": ["asset", "market"],
    "stock": ["asset", "market"],
    "forex": ["asset", "market"],
    "forecast": ["predict", "market"],
    "tip": ["recommend"],
    "signal": ["recommend"],
    "reject": ["declin"],
    "denied": ["declin"],
    "refus": ["declin"],
    "kyc": ["verify"],
    "document": ["verify"],
    "spam": ["email"],
    "inbox": ["email"],
}

# Query-side phrase expansion on canonical text.
PHRASE_EXPANSIONS = [
    (re.compile(r"\b(?:go|goes|going|will)\s+(?:up|down)\b|\b(?:rise|fall|pump)\b"), ["predict", "market"]),
    (re.compile(r"\bget into\b|\bcannot (?:get|access|signin)\b"), ["signin", "lock"]),
    (re.compile(r"\b(?:not|hasn't|haven't|isn't|didn't)\s+(?:\w+\s+)?(?:show|arriv|appear|reflect|credit)\w*|\bshows? (?:zero|0|nothing)\b|\bmissing\b"),
     ["delay", "transaction"]),
]


def canonical(text: str) -> str:
    out = (text or "").lower()
    for pattern, repl in _CANONICAL:
        out = pattern.sub(repl, out)
    return out


def stem(word: str) -> str:
    """Tiny suffix stripper: deterministic and good enough for a small support vocabulary."""
    if len(word) <= 3:
        return word
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith("ss"):
                break
            word = word[: -len(suffix)] + repl
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokens(text: str) -> List[str]:
    """Canonicalise, tokenise, drop stopwords, stem, alias."""
    result = []
    for raw in _TOKEN_RE.findall(canonical(text)):
        if raw in STOPWORDS:
            continue
        tok = stem(raw)
        tok = TOKEN_ALIASES.get(tok, TOKEN_ALIASES.get(raw, tok))
        if tok and tok not in STOPWORDS:
            result.append(tok)
    return result


def query_tokens(text: str) -> List[str]:
    """Tokens plus synonym / phrase expansions (used for retrieval queries only)."""
    base = tokens(text)
    extra: List[str] = []
    for tok in base:
        extra.extend(QUERY_EXPANSIONS.get(tok, []))
    canon = canonical(text)
    for pattern, words in PHRASE_EXPANSIONS:
        if pattern.search(canon):
            extra.extend(words)
    return base + extra


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split((text or "").strip()) if s.strip()]


_CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-]+)\]")


def citations(text: str) -> List[str]:
    return _CITATION_RE.findall(text or "")


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text or "")
