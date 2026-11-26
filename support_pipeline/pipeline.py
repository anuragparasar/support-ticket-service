"""Pipeline orchestration.

LOAD_DATA -> INDEX_KB -> CLASSIFY_INTENT -> RETRIEVE_CONTEXT -> GENERATE_REPLY -> VALIDATE_OUTPUT -> WRITE_RESULTS

Each stage is a plain function. Components (classifier / retriever / generator) are chosen by name in
`build_components`, so any of them can be swapped without touching the stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .classifier import FallbackClassifier, GeminiClassifier, IntentClassifier, RuleClassifier
from .data import load_articles, load_tickets
from .generator import (
    ADVICE_CLOSING,
    ADVICE_INTENT,
    CLOSING,
    HANDOFF,
    GeminiGenerator,
    ReplyGenerator,
    TemplateGenerator,
    holding_draft,
    refusal_sentences,
)
from .llm import get_gemini_client
from .report import write_results as _write_results
from .retriever import Retriever, build_retriever
from .schemas import Article, Classification, Draft, Hit, Ticket, TicketResult
from .scoring import confidence, flags_and_escalation
from .validator import ContentCheck, check_content, validate_result


@dataclass
class RunOptions:
    classifier: str = "rules"     # rules | gemini
    retriever: str = "lexical"    # lexical | hybrid
    generator: str = "template"   # template | gemini


@dataclass
class Components:
    classifier: IntentClassifier
    generator: ReplyGenerator
    template: TemplateGenerator
    notes: List[str] = field(default_factory=list)


@dataclass
class KBIndex:
    articles: Dict[str, Article]
    tags: Dict[str, str]          # article_id -> intent tag (from the rule classifier, no hardcoding)
    retriever: Retriever


def build_components(options: RunOptions) -> Components:
    """Selects implementations by name; anything unavailable degrades to the offline default."""
    notes: List[str] = []
    rules, template = RuleClassifier(), TemplateGenerator()
    classifier: IntentClassifier = rules
    generator: ReplyGenerator = template
    if "gemini" in (options.classifier, options.generator):
        client, reason = get_gemini_client()
        if client is None:
            notes.append(f"Gemini unavailable ({reason}); using offline components")
        else:
            if options.classifier == "gemini":
                classifier = FallbackClassifier(GeminiClassifier(client), rules)
            if options.generator == "gemini":
                generator = GeminiGenerator(client)
    return Components(classifier, generator, template, notes)


# ----------------------------------------------------------------------------- stages
def load_data(tickets_path: Path, kb_path: Path) -> Tuple[List[Ticket], List[Article], List[str]]:
    tickets = load_tickets(tickets_path)
    articles, warnings = load_articles(kb_path)
    return tickets, articles, warnings


def index_kb(articles: List[Article], retriever_name: str) -> KBIndex:
    tagger = RuleClassifier()
    tags = {a.article_id: tagger.classify(f"{a.title}. {a.body}").label for a in articles}
    return KBIndex({a.article_id: a for a in articles}, tags, build_retriever(retriever_name, articles))


def classify_intent(ticket: Ticket, classifier: IntentClassifier) -> Classification:
    return classifier.classify(ticket.message)


def retrieve_context(ticket: Ticket, index: KBIndex) -> List[Hit]:
    return index.retriever.retrieve(ticket.message)


def generate_reply(ticket: Ticket, cls: Classification, hits: List[Hit], index: KBIndex,
                   components: Components) -> Draft:
    generator = components.generator
    if generator is components.template or cls.label == "other" or not hits:
        return components.template.generate(ticket, cls, hits, index.articles, index.tags)
    try:
        return generator.generate(ticket, cls, hits, index.articles, index.tags)
    except Exception as exc:
        draft = components.template.generate(ticket, cls, hits, index.articles, index.tags)
        draft.fallback_reason = f"{generator.name} generator failed ({type(exc).__name__}): {str(exc)[:160]}"
        return draft


def validate_output(ticket: Ticket, cls: Classification, hits: List[Hit], draft: Draft, index: KBIndex,
                    components: Components) -> Tuple[TicketResult, Dict]:
    advice_required = cls.advice_detected or cls.label == ADVICE_INTENT
    check = check_content(draft, hits, index.articles, advice_required)
    repairs: List[str] = []

    # 1) A model draft with any problem is replaced by the deterministic template draft.
    if draft.component != components.template.name and (check.hard_issues or not check.grounded):
        repairs.append(f"{draft.component} draft rejected: {check.hard_issues or check.unsupported_sentences}")
        draft = components.template.generate(ticket, cls, hits, index.articles, index.tags)
        draft.fallback_reason = "model draft failed validation"
        check = check_content(draft, hits, index.articles, advice_required)

    # 2) Anything still unsafe becomes a holding reply (+ refusal if advice was requested).
    claim_flag: List[str] = []
    if check.hard_issues:
        repairs.append(f"draft replaced by holding reply: {check.hard_issues}")
        claim_flag = ["unsupported_claim"] if check.has_claim_issue else []
        draft = holding_draft(components.template.name, "draft failed content validation")
        if advice_required:
            refusal = refusal_sentences(hits, index.tags)
            draft.text = f"{draft.text} {' '.join(refusal)}"
        check = check_content(draft, hits, index.articles, advice_required)

    fallback_used = bool(cls.fallback_reason or draft.fallback_reason or index.retriever.fallback_reason or repairs)
    conf, breakdown = confidence(cls, hits, check, index.tags, fallback_used)
    flags, escalate, reasons = flags_and_escalation(ticket, cls, hits, check, conf, claim_flag)

    closing = HANDOFF if escalate else (ADVICE_CLOSING if cls.label == ADVICE_INTENT else CLOSING)
    result = TicketResult(
        ticket_id=ticket.id,
        intent=cls.label,
        retrieved_articles=[h.article_id for h in hits],
        reply_draft=f"{draft.text} {closing}",
        grounded=check.grounded,
        confidence=conf,
        needs_human_escalation=escalate,
        safety_flags=flags,
    )
    problems = validate_result(result.model_dump(), index.articles.keys())
    if problems:
        raise ValueError(f"result failed final validation: {problems}")

    debug = {
        "ticket_id": ticket.id,
        "message": ticket.message,
        "intent": {
            "label": cls.label,
            "certainty": cls.certainty,
            "component": cls.component,
            "scores": {k: v for k, v in cls.scores.items() if v},
            "matched_terms": cls.matched_terms,
            "advice_detected": cls.advice_detected,
            "rationale": cls.rationale,
            "fallback": cls.fallback_reason,
        },
        "retrieval": {
            "component": index.retriever.name,
            "fallback": index.retriever.fallback_reason,
            "hits": [{"article_id": h.article_id, "title": index.articles[h.article_id].title,
                      "topic": index.tags.get(h.article_id), "score": h.score, "components": h.components}
                     for h in hits],
        },
        "generation": {"component": draft.component, "fallback": draft.fallback_reason, "repairs": repairs},
        "grounding": {"grounded": check.grounded, "score": check.grounding_score,
                      "sentences": check.sentence_support},
        "confidence": breakdown,
        "escalation": {"escalate": escalate, "reasons": reasons},
        "safety_flags": flags,
    }
    return result, debug


def safe_error_result(ticket: Ticket, error: Exception) -> Tuple[TicketResult, Dict]:
    """Per-ticket last resort: a schema-valid, escalated result. Never raises."""
    draft = holding_draft("template")
    result = TicketResult(
        ticket_id=ticket.id, intent="other", retrieved_articles=[], reply_draft=f"{draft.text} {HANDOFF}",
        grounded=False, confidence=0.0, needs_human_escalation=True,
        safety_flags=["insufficient_grounding", "low_confidence", "processing_error"],
    )
    debug = {"ticket_id": ticket.id, "message": ticket.message,
             "error": f"{type(error).__name__}: {str(error)[:300]}",
             "escalation": {"escalate": True, "reasons": ["processing error; safe fallback result"]},
             "safety_flags": result.safety_flags}
    return result, debug


def process_ticket(ticket: Ticket, index: KBIndex, components: Components) -> Tuple[TicketResult, Dict]:
    try:
        cls = classify_intent(ticket, components.classifier)
        hits = retrieve_context(ticket, index)
        draft = generate_reply(ticket, cls, hits, index, components)
        return validate_output(ticket, cls, hits, draft, index, components)
    except Exception as exc:  # one bad ticket never stops the run
        return safe_error_result(ticket, exc)


def process_all(tickets: List[Ticket], articles: List[Article],
                options: Optional[RunOptions] = None) -> Tuple[List[TicketResult], List[Dict], Dict]:
    """In-memory entry point (used by tests): INDEX_KB + per-ticket stages."""
    options = options or RunOptions()
    components = build_components(options)
    index = index_kb(articles, options.retriever)
    pairs = [process_ticket(t, index, components) for t in tickets]
    results = [r for r, _ in pairs]
    debug = [d for _, d in pairs]
    meta = {
        "components": {"classifier": components.classifier.name, "retriever": index.retriever.name,
                       "generator": components.generator.name},
        "requested": vars(options),
        "notes": components.notes + ([index.retriever.fallback_reason] if index.retriever.fallback_reason else []),
        "article_topics": index.tags,
    }
    return results, debug, meta


def run_pipeline(tickets_path: Path, kb_path: Path, out_dir: Path,
                 options: Optional[RunOptions] = None) -> Dict:
    """Full run from disk to disk. Raises InputDataError only if inputs are missing / unreadable."""
    tickets, articles, warnings = load_data(tickets_path, kb_path)          # LOAD_DATA
    results, debug, meta = process_all(tickets, articles, options)           # INDEX_KB .. VALIDATE_OUTPUT
    meta["warnings"] = warnings
    summary = _write_results(results, debug, meta, out_dir)                  # WRITE_RESULTS
    return summary
