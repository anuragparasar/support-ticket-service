# Support Ticket Triage & Grounded Reply Service

A small, runnable Python pipeline that reads support tickets and a local help-centre KB, then for each ticket:

1. **classifies the intent** (one of 6 allowed labels),
2. **retrieves the 1–3 most relevant KB articles** (deterministic TF-IDF),
3. **drafts a reply grounded in those articles** (every factual sentence cites `[A#]`), with a **confidence** score,
   **escalation** signal and **safety flags**.

Runs fully offline with no secrets. An LLM (Gemini) and a semantic retriever (MiniLM) are optional plug-ins,
and every optional piece falls back to the offline path. Design notes: [SPEC.md](SPEC.md).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python main.py                    # writes results.json, debug_report.md, debug_report.json
python validate.py                # end-to-end checks, exit code 0 = pass
pytest -q                         # 24 unit / behaviour tests
```

Requires Python 3.10+ (developed on 3.13).

## Pipeline

```
LOAD_DATA -> INDEX_KB -> CLASSIFY_INTENT -> RETRIEVE_CONTEXT -> GENERATE_REPLY -> VALIDATE_OUTPUT -> WRITE_RESULTS
```

Each stage is a named function in [`support_pipeline/pipeline.py`](support_pipeline/pipeline.py).

| Stage | Module | Default (offline) | Optional |
|---|---|---|---|
| LOAD_DATA | `data.py` | Pydantic-validated tickets + articles; malformed tickets kept and flagged | — |
| INDEX_KB | `retriever.py` | Word TF-IDF (stemmed + synonym-expanded) and char n-gram TF-IDF; each article tagged with a topic by the rule classifier | MiniLM embeddings |
| CLASSIFY_INTENT | `classifier.py` | Weighted keyword/phrase rules, certainty from score margin + strength; always-on trading-advice guard | Gemini, constrained to the 6 labels |
| RETRIEVE_CONTEXT | `retriever.py` | Ticket text only → fused score `0.6·word + 0.4·char`, threshold, ties by ID | `0.5·word + 0.3·char + 0.2·semantic` |
| GENERATE_REPLY | `generator.py` | Extractive: client-facing sentences from retrieved articles, each cited; fixed refusal for advice | Gemini with snippets-only prompt |
| VALIDATE_OUTPUT | `validator.py`, `scoring.py` | Citations ⊆ retrieved, per-sentence lexical support, refusal check, promise / account-fact detection, schema; confidence + flags + escalation | — |
| WRITE_RESULTS | `report.py` | Atomic writes of `results.json` + debug reports | — |

Retrieval never sees the predicted intent, and the generator only sees what retrieval returned.

## Output

`results.json`: one object per ticket in the required schema:

```json
{
  "ticket_id": "T4",
  "intent": "trading_advice_request",
  "retrieved_articles": ["A5"],
  "reply_draft": "Thanks for your message. I'm sorry, but we're not able to provide market predictions, profit guarantees, or recommendations on specific assets [A5]. ...",
  "grounded": true,
  "confidence": 0.8502,
  "needs_human_escalation": false,
  "safety_flags": ["advice_request"]
}
```

`debug_report.md` / `.json`: for each ticket, the predicted intent with matched terms, the retrieved article titles
with score breakdown, per-sentence grounding, the confidence formula with penalties, **why it was or wasn't escalated**,
which component ran, and any fallbacks.

### Sample results

| Ticket | Intent | Article | Escalated | Flags |
|---|---|---|---|---|
| T1 | deposit_issue | A1 | no | account_specific_request |
| T2 | password_reset | A2 | no | — |
| T3 | withdrawal_issue | A3 | **yes** | policy_sensitive, account_specific_request |
| T4 | trading_advice_request | A5 | no (refused) | advice_request |
| T5 | account_lock | A4 | no | — |

## Confidence, flags and escalation

```
confidence = 0.4·classifier_certainty + 0.4·retrieval_strength + 0.2·grounding_score − penalties
```

The score is computed from evidence, not taken from a model's self-report. Penalties apply for a conflict between the
top article's topic and the intent (−0.15) and for any fallback used (−0.05). With no relevant article,
confidence is capped at 0.3.

| Flag | When |
|---|---|
| `advice_request` | Prediction / recommendation / profit request (reply is a refusal) |
| `insufficient_grounding` | No relevant article, or a sentence not supported by its cited article |
| `policy_sensitive` | Withdrawal outcomes (compliance / account review) |
| `low_confidence` | confidence < 0.5 |
| `account_specific_request` | Asks about own balance / withdrawal / transaction (informational) |
| `unsupported_claim` | Draft promised an outcome or stated account facts |
| `invalid_input`, `unsupported_language`, `processing_error` | Edge cases |

**Escalate** if any of `insufficient_grounding`, `policy_sensitive`, `low_confidence`, `unsupported_claim`,
`invalid_input`, `unsupported_language`, `processing_error`, or the intent is `other`.
A pure advice request is refused but not escalated.

## Optional components

```bash
# Gemini classifier and/or generator (key in .env as GEMINI_API_KEY; model via GEMINI_MODEL)
pip install -r requirements-llm.txt
python main.py --classifier gemini --generator gemini

# MiniLM semantic layer for retrieval
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-semantic.txt
python main.py --retriever hybrid
```

LLM safety: the prompt contains only the retrieved snippets and requires a citation for every factual sentence.
Gemini output is held to the **same validator** as the template. A model draft that cites a non-retrieved article,
lacks support, promises outcomes, states account facts or fails to refuse advice is discarded, and the deterministic
template draft is used instead. In testing, the validator caught a Gemini draft that said "your payment was…" and
replaced it.

## Fallbacks

| Problem | Behaviour |
|---|---|
| No `google-genai` / no API key | Offline rules + template, with a note in the console and debug report |
| Gemini timeout, quota (429), bad JSON, label outside the set | One retry, then the offline component for that ticket |
| No `sentence-transformers` / model | Lexical retrieval |
| No article above threshold / empty KB | Empty `retrieved_articles`, holding reply, escalated |
| Malformed ticket, empty message, non-English | Schema-valid escalated result with a flag; other tickets unaffected |
| Unexpected error in one ticket | Safe escalated result with `processing_error` |
| Missing / invalid `tickets.json` or `kb_articles.json` | Clear error, exit code 2 (the only hard failure) |

## Validation

`python validate.py` checks:

- the files exist and the JSON is valid;
- there is one result per ticket, in input order;
- the exact schema and field set are respected;
- article IDs are valid (at most 3);
- trading-advice requests are refused, with T4 checked explicitly;
- the output is reproducible: two fresh runs are byte-identical and match `results.json`.

`pytest` covers:

- the required checks (T4 refused, T1 → A1, schema);
- every sample expectation;
- paraphrases that must still reach the right article;
- an out-of-scope ticket → `other` + escalation;
- a mixed deposit + advice ticket;
- "withdraw my profits" not being treated as advice;
- malformed and non-English input;
- a missing input file and an empty KB;
- a fake LLM returning invalid output or promises;
- a missing API key;
- the hybrid retriever;
- determinism.

## Swapping components

The classifier, retriever and generator are small `Protocol`s, chosen by name in `pipeline.build_components` and
`retriever.build_retriever`. Adding, say, a vector-DB retriever or another hosted LLM means adding one class and one
branch; the stages don't change.

## Project layout

```
main.py  validate.py  SPEC.md  README.md
requirements.txt  requirements-llm.txt  requirements-semantic.txt
tickets.json  kb_articles.json                    # inputs
results.json  debug_report.md  debug_report.json  # generated
support_pipeline/  config.py schemas.py data.py text.py classifier.py retriever.py
                   generator.py validator.py scoring.py llm.py pipeline.py report.py
tests/test_pipeline.py
```

## Limitations

- The rule classifier and synonym map are tuned to the KB's domain. New topics need patterns, or the Gemini classifier.
- TF-IDF on a 5-article KB is noisy. Thresholds were calibrated on the sample data plus paraphrases.
- Template replies are extractive. They're less fluent than LLM replies, but traceable and deterministic.
- The grounding check uses lexical overlap, not semantic entailment.
- English only; other languages are escalated.
