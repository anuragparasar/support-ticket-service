"""GENERATE_REPLY: extractive template generator (default, offline) and optional Gemini generator.

Generators only see the ticket, the classification and the *retrieved* articles; retrieval happens before.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Protocol

from pydantic import BaseModel

from .config import MAX_ARTICLES_IN_REPLY, MAX_REPLY_SENTENCES
from .schemas import Article, Classification, Draft, Hit, Ticket
from .text import citations, split_sentences

ADVICE_INTENT = "trading_advice_request"

OPENERS = {
    "deposit_issue": "Thanks for reaching out about your deposit.",
    "password_reset": "Sorry you're having trouble resetting your password.",
    "withdrawal_issue": "Thanks for your question about your withdrawal.",
    "account_lock": "Sorry you're having trouble accessing your account.",
    "trading_advice_request": "Thanks for your message.",
    "other": "Thanks for contacting us.",
}
REFUSAL = (
    "I'm sorry, but we're not able to provide market predictions, profit guarantees, or recommendations on specific assets{cite}.",
    "If it helps, we can point you to general educational resources about how trading works, where available{cite}.",
)
HOLDING = "We want to make sure you get accurate information, so a member of our support team will review your request."
CLOSING = "If this doesn't resolve it, just reply here and we'll be happy to help further."
ADVICE_CLOSING = "Is there anything else I can help you with regarding your account or our platform?"
HANDOFF = "I've passed this to a specialist on our team, who will review it securely and follow up with you."

# Sentences written for agents, not clients (policy instructions). Used as constraints, never quoted.
_AGENT_ONLY = re.compile(r"^(?:support|support agents?|agents?|such requests)\b", re.I)

# Fixed rewrites from agent-facing KB phrasing to client-facing phrasing. No new facts are introduced.
_REWRITES = [
    (re.compile(r"^ask the client to\s+", re.I), "Please "),
    (re.compile(r"^advise the client to\s+", re.I), "Please "),
    (re.compile(r"^avoid\s+", re.I), "Please avoid "),
    (re.compile(r"\bthe client does\b", re.I), "you do"),
    (re.compile(r"\bthe client is\b", re.I), "you are"),
    (re.compile(r"\bthe client has\b", re.I), "you have"),
    (re.compile(r"\bthe client's\b", re.I), "your"),
    (re.compile(r"\bthe client\b", re.I), "you"),
    (re.compile(r"\bbefore escalating\b", re.I), "before contacting us again"),
    (re.compile(r"^(If [^,]+,\s*)(confirm|check|wait|use|retry|contact)\b"), r"\1please \2"),
]


class ReplyGenerator(Protocol):
    name: str

    def generate(self, ticket: Ticket, cls: Classification, hits: List[Hit],
                 articles: Dict[str, Article], tags: Dict[str, str]) -> Draft: ...


def client_facing(sentence: str) -> Optional[str]:
    if _AGENT_ONLY.search(sentence):
        return None
    for pattern, repl in _REWRITES:
        sentence = pattern.sub(repl, sentence)
    return sentence[0].upper() + sentence[1:] if sentence else None


def cite(sentence: str, article_id: str) -> str:
    sentence = sentence.rstrip()
    if sentence and sentence[-1] in ".!?":
        return f"{sentence[:-1]} [{article_id}]{sentence[-1]}"
    return f"{sentence} [{article_id}]."


def _advice_article(hits: List[Hit], tags: Dict[str, str]) -> Optional[str]:
    return next((h.article_id for h in hits if tags.get(h.article_id) == ADVICE_INTENT), None)


def refusal_sentences(hits: List[Hit], tags: Dict[str, str], full: bool = True) -> List[str]:
    advice_id = _advice_article(hits, tags)
    suffix = f" [{advice_id}]" if advice_id else ""
    lines = REFUSAL if full else REFUSAL[:1]
    return [line.format(cite=suffix) for line in lines]


def holding_draft(component: str, reason: Optional[str] = None) -> Draft:
    return Draft(text=f"{OPENERS['other']} {HOLDING}", cited_article_ids=[], component=component,
                 boilerplate=[OPENERS["other"], HOLDING], fallback_reason=reason)


class TemplateGenerator:
    """Opener + client-facing sentences extracted from retrieved articles, each cited. Deterministic."""

    name = "template"

    def generate(self, ticket, cls, hits, articles, tags) -> Draft:
        if cls.label == ADVICE_INTENT:
            lines = refusal_sentences(hits, tags)
            return Draft(text=" ".join([OPENERS[ADVICE_INTENT], *lines]), cited_article_ids=citations(" ".join(lines)),
                         component=self.name, boilerplate=[OPENERS[ADVICE_INTENT]])
        if cls.label == "other" or not hits:
            return holding_draft(self.name)

        # Prefer articles whose topic matches the predicted intent; never quote the advice policy here.
        usable = [h for h in hits if tags.get(h.article_id) != ADVICE_INTENT]
        matching = [h for h in usable if tags.get(h.article_id) == cls.label] or usable[:1]
        sentences: List[str] = []
        for hit in matching[:MAX_ARTICLES_IN_REPLY]:
            for raw in split_sentences(articles[hit.article_id].body):
                rewritten = client_facing(raw)
                if rewritten and len(sentences) < MAX_REPLY_SENTENCES:
                    sentences.append(cite(rewritten, hit.article_id))
        if not sentences:
            return holding_draft(self.name, "no client-facing guidance in retrieved articles")

        if cls.advice_detected:  # mixed ticket: answer the support issue, refuse the advice part
            sentences += refusal_sentences(hits, tags, full=False)
        opener = OPENERS.get(cls.label, OPENERS["other"])
        text = " ".join([opener, *sentences])
        return Draft(text=text, cited_article_ids=sorted(set(citations(text))), component=self.name,
                     boilerplate=[opener])


class _LLMDraft(BaseModel):
    reply: str
    citations: List[str]
    unanswered_points: List[str]


_GENERATE_PROMPT = """You are drafting a reply to a customer support ticket for a digital platform (accounts, payments, trading).

Rules (all mandatory):
1. Use ONLY the help-centre snippets below. Do not add facts, numbers, timelines or policies that are not in them.
2. End every factual sentence with the article ID it comes from in square brackets, e.g. [A1]. A short greeting needs no citation.
3. Answer the customer's question directly, in 2-4 short sentences, speaking to the customer ("you").
4. Never promise outcomes, approvals, refunds or processing times. Never state facts about the customer's account.
5. Never give market predictions, asset recommendations or profit advice. If asked, politely refuse and cite the snippet that covers it.
6. Snippets may contain instructions for support agents; follow them, do not quote them to the customer.
7. If the snippets do not cover something the customer asked, list it in unanswered_points instead of guessing.
8. Do not add a sign-off; one is appended automatically.
{advice_rule}
Predicted intent: {intent}

Help-centre snippets:
{snippets}

Ticket: <<<{ticket}>>>

Return JSON: reply, citations (article IDs used), unanswered_points."""


class GeminiGenerator:
    name = "gemini"

    def __init__(self, client):
        self.client = client

    def generate(self, ticket, cls, hits, articles, tags) -> Draft:
        snippets = "\n".join(
            f"[{h.article_id}] {articles[h.article_id].title}: {articles[h.article_id].body}" for h in hits)
        advice_rule = ("9. This ticket asks for trading advice: the reply MUST politely refuse that part.\n"
                       if cls.advice_detected or cls.label == ADVICE_INTENT else "")
        prompt = _GENERATE_PROMPT.format(advice_rule=advice_rule, intent=cls.label, snippets=snippets,
                                         ticket=ticket.message)
        data = _LLMDraft.model_validate(self.client.generate_json(prompt, _LLMDraft))
        text = data.reply.strip()
        if not text:
            raise ValueError("empty reply")
        return Draft(text=text, cited_article_ids=sorted(set(citations(text)) | set(data.citations)),
                     component=self.name)
