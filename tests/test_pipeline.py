import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from support_pipeline import pipeline as pl  # noqa: E402
from support_pipeline.data import InputDataError, load_articles, load_tickets  # noqa: E402
from support_pipeline.pipeline import Components, RunOptions, process_all, run_pipeline  # noqa: E402
from support_pipeline.schemas import RESULT_FIELDS, Ticket, TicketResult  # noqa: E402
from support_pipeline.validator import is_refusal, validate_result  # noqa: E402

ARTICLES, _ = load_articles(ROOT / "kb_articles.json")
KB_IDS = {a.article_id for a in ARTICLES}


def run(messages):
    tickets = [Ticket(id=f"X{i}", message=m) for i, m in enumerate(messages)]
    results, debug, _ = process_all(tickets, ARTICLES)
    return results, debug


@pytest.fixture(scope="module")
def sample():
    results, debug, _ = process_all(load_tickets(ROOT / "tickets.json"), ARTICLES)
    return {r.ticket_id: r for r in results}


# --- required minimum ---------------------------------------------------------
def test_t4_refused_as_trading_advice(sample):
    t4 = sample["T4"]
    assert t4.intent == "trading_advice_request"
    assert "advice_request" in t4.safety_flags
    assert is_refusal(t4.reply_draft)


def test_t1_retrieves_deposit_article(sample):
    assert sample["T1"].retrieved_articles[0] == "A1"


def test_output_matches_schema(tmp_path):
    run_pipeline(ROOT / "tickets.json", ROOT / "kb_articles.json", tmp_path)
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert len(data) == 5
    for item in data:
        assert list(item.keys()) == RESULT_FIELDS
        TicketResult.model_validate(item)
        assert validate_result(item, KB_IDS) == []


# --- behaviour on the sample --------------------------------------------------
@pytest.mark.parametrize("tid,intent,article,escalate", [
    ("T1", "deposit_issue", "A1", False),
    ("T2", "password_reset", "A2", False),
    ("T3", "withdrawal_issue", "A3", True),
    ("T4", "trading_advice_request", "A5", False),
    ("T5", "account_lock", "A4", False),
])
def test_sample_expectations(sample, tid, intent, article, escalate):
    r = sample[tid]
    assert r.intent == intent
    assert r.retrieved_articles[0] == article
    assert r.needs_human_escalation is escalate
    assert r.grounded is True
    assert f"[{article}]" in r.reply_draft  # traceable to retrieved content


def test_withdrawal_is_policy_sensitive_and_makes_no_promises(sample):
    t3 = sample["T3"]
    assert "policy_sensitive" in t3.safety_flags
    assert "will be approved" not in t3.reply_draft.lower()


# --- paraphrases / edge cases -------------------------------------------------
@pytest.mark.parametrize("message,intent,article", [
    ("money has not shown up after I paid", "deposit_issue", "A1"),
    ("cannot get into my account after several wrong passwords", "account_lock", "A4"),
    ("should I buy gold now?", "trading_advice_request", "A5"),
    ("the password reset link never arrives in my inbox", "password_reset", "A2"),
])
def test_paraphrases(message, intent, article):
    (r,), _ = run([message])
    assert r.intent == intent
    assert r.retrieved_articles and r.retrieved_articles[0] == article


def test_out_of_scope_is_other_and_escalated():
    (r,), _ = run(["How do I change my phone number?"])
    assert r.intent == "other"
    assert r.needs_human_escalation
    assert "insufficient_grounding" in r.safety_flags
    assert r.grounded is False


def test_mixed_ticket_answers_and_refuses():
    (r,), _ = run(["My deposit is missing, also which coin should I buy?"])
    assert r.intent == "deposit_issue"
    assert "advice_request" in r.safety_flags
    assert is_refusal(r.reply_draft)
    assert "A1" in r.retrieved_articles


def test_withdrawing_profits_is_not_advice():
    (r,), _ = run(["I can not withdraw my profits"])
    assert r.intent == "withdrawal_issue"
    assert "advice_request" not in r.safety_flags


def test_malformed_ticket_does_not_break_run(tmp_path):
    path = tmp_path / "tickets.json"
    path.write_text(json.dumps([{"id": "G1", "message": "I forgot my password"}, {"message": ""}, "junk",
                                {"id": "L1", "message": "Mi cuenta esta bloqueada", "language": "es"}]))
    results, _, _ = process_all(load_tickets(path), ARTICLES)
    assert [r.ticket_id for r in results] == ["G1", "UNKNOWN_2", "UNKNOWN_3", "L1"]
    assert results[0].intent == "password_reset" and not results[0].needs_human_escalation
    for r in results[1:3]:
        assert r.needs_human_escalation and "invalid_input" in r.safety_flags
    assert "unsupported_language" in results[3].safety_flags and results[3].needs_human_escalation


def test_missing_input_file_fails_clearly(tmp_path):
    with pytest.raises(InputDataError):
        run_pipeline(tmp_path / "nope.json", ROOT / "kb_articles.json", tmp_path)


def test_empty_kb_escalates_everything():
    results, _, _ = process_all([Ticket(id="E1", message="I forgot my password")], [])
    assert results[0].retrieved_articles == [] and results[0].needs_human_escalation


# --- fallbacks / swap boundary ------------------------------------------------
class _BrokenClient:
    def generate_json(self, prompt, schema):
        return {"intent": "buy_bitcoin", "confidence": 2, "rationale": "x", "reply": "", "citations": []}


class _PromisingClient:
    def generate_json(self, prompt, schema):
        return {"reply": "Your withdrawal will be approved within 2 hours [A3].", "citations": ["A3"],
                "unanswered_points": []}


def _patched_components(monkeypatch, client, classifier, generator):
    monkeypatch.setattr(pl, "get_gemini_client", lambda: (client, None))
    return RunOptions(classifier=classifier, generator=generator)


def test_invalid_llm_output_falls_back(monkeypatch, sample):
    opts = _patched_components(monkeypatch, _BrokenClient(), "gemini", "gemini")
    results, debug, _ = process_all(load_tickets(ROOT / "tickets.json"), ARTICLES, opts)
    assert [r.intent for r in results] == [sample[t].intent for t in ["T1", "T2", "T3", "T4", "T5"]]
    assert all(d["intent"]["fallback"] for d in debug)
    for r in results:
        assert validate_result(r.model_dump(), KB_IDS) == []


def test_llm_promise_is_rejected(monkeypatch):
    opts = _patched_components(monkeypatch, _PromisingClient(), "rules", "gemini")
    (r,), (d,) = process_all([Ticket(id="W", message="Why was my withdrawal declined?")], ARTICLES, opts)[:2]
    assert "will be approved" not in r.reply_draft
    assert d["generation"]["repairs"]  # model draft rejected and replaced


def test_no_api_key_runs_offline(monkeypatch):
    monkeypatch.setattr(pl, "get_gemini_client", lambda: (None, "no GEMINI_API_KEY set"))
    results, _, meta = process_all(load_tickets(ROOT / "tickets.json"), ARTICLES,
                                   RunOptions(classifier="gemini", generator="gemini"))
    assert meta["components"] == {"classifier": "rules", "retriever": "lexical", "generator": "template"}
    assert any("unavailable" in n for n in meta["notes"])
    assert len(results) == 5


def test_hybrid_request_never_crashes():
    results, _, meta = process_all(load_tickets(ROOT / "tickets.json"), ARTICLES, RunOptions(retriever="hybrid"))
    assert meta["components"]["retriever"] in {"hybrid", "lexical"}
    assert results[0].retrieved_articles[0] == "A1"


def test_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run_pipeline(ROOT / "tickets.json", ROOT / "kb_articles.json", a)
    run_pipeline(ROOT / "tickets.json", ROOT / "kb_articles.json", b)
    assert (a / "results.json").read_bytes() == (b / "results.json").read_bytes()
