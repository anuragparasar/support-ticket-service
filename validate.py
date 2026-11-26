"""End-to-end validation of the pipeline outputs.

    python validate.py            # exits 0 if every check passes, 1 otherwise

Checks: files exist, JSON valid, one result per ticket, schema, valid article IDs,
trading-advice requests refused (T4 on the sample data), reproducible outputs.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from support_pipeline.pipeline import RunOptions, run_pipeline
from support_pipeline.schemas import RESULT_FIELDS
from support_pipeline.validator import is_refusal, validate_result

ROOT = Path(__file__).resolve().parent
TICKETS, KB, RESULTS = ROOT / "tickets.json", ROOT / "kb_articles.json", ROOT / "results.json"
REQUIRED_FILES = [TICKETS, KB, RESULTS, ROOT / "debug_report.md", ROOT / "debug_report.json", ROOT / "README.md"]

failures: list = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(label)
    return condition


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, str(exc)


def main() -> int:
    if not RESULTS.exists():
        print("results.json not found - running `python main.py` first")
        run_pipeline(TICKETS, KB, ROOT)

    # 1. files exist
    for path in REQUIRED_FILES:
        check(path.exists(), f"file exists: {path.name}")

    # 2. JSON valid
    tickets, err_t = load(TICKETS)
    kb, err_k = load(KB)
    results, err_r = load(RESULTS)
    check(err_t is None, "tickets.json is valid JSON", err_t or "")
    check(err_k is None, "kb_articles.json is valid JSON", err_k or "")
    check(err_r is None and isinstance(results, list), "results.json is a valid JSON list", err_r or "")
    _, err_d = load(ROOT / "debug_report.json")
    check(err_d is None, "debug_report.json is valid JSON", err_d or "")
    if failures:
        return finish()

    kb_ids = {a["article_id"] for a in kb if isinstance(a, dict) and "article_id" in a}
    ticket_ids = [str(t.get("id")) for t in tickets if isinstance(t, dict)]

    # 3. one result per ticket, same order
    result_ids = [r.get("ticket_id") for r in results]
    check(len(results) == len(tickets), "one result per ticket", f"{len(results)} results for {len(tickets)} tickets")
    check(result_ids == ticket_ids, "result ticket_ids match tickets (in order)", f"{result_ids} vs {ticket_ids}")

    # 4. schema + article IDs + per-result rules
    for r in results:
        problems = validate_result(r, kb_ids)
        check(not problems, f"{r.get('ticket_id')}: schema and content rules", "; ".join(problems))
        check(list(r.keys()) == RESULT_FIELDS, f"{r.get('ticket_id')}: exact field set / order")
        arts = r.get("retrieved_articles", [])
        check(all(a in kb_ids for a in arts) and len(arts) <= 3,
              f"{r.get('ticket_id')}: retrieved_articles valid ({arts})")
        if not arts:
            check(r.get("needs_human_escalation") is True, f"{r.get('ticket_id')}: no article -> escalated")

    # 5. trading advice safely refused (generic, plus the sample's T4)
    advice = [r for r in results if r.get("intent") == "trading_advice_request" or "advice_request" in r.get("safety_flags", [])]
    for r in advice:
        check(is_refusal(r["reply_draft"]), f"{r['ticket_id']}: advice request refused")
    by_id = {r.get("ticket_id"): r for r in results}
    if "T4" in by_id:
        t4 = by_id["T4"]
        check(t4["intent"] == "trading_advice_request" and "advice_request" in t4["safety_flags"]
              and is_refusal(t4["reply_draft"]), "T4 is classified as trading advice and safely refused")

    # 6. reproducibility: two fresh offline runs must be byte-identical (and match results.json if it was offline)
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        run_pipeline(TICKETS, KB, Path(a), RunOptions())
        run_pipeline(TICKETS, KB, Path(b), RunOptions())
        first = (Path(a) / "results.json").read_bytes()
        second = (Path(b) / "results.json").read_bytes()
        check(first == second, "reproducible: two runs produce identical results.json")
        debug, _ = load(ROOT / "debug_report.json")
        offline = debug and debug["run"]["components"] == {"classifier": "rules", "retriever": "lexical", "generator": "template"}
        if offline:
            check(first == RESULTS.read_bytes(), "reproducible: committed results.json matches a fresh run")
        else:
            print("[SKIP] results.json was produced with optional components; byte comparison skipped")
    return finish()


def finish() -> int:
    print()
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} check(s) failed")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
