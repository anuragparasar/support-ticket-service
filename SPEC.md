# Support Ticket Triage & Grounded Reply Service — Design Spec

## 1. Goal

A small, runnable Python pipeline that, for each customer support ticket:

1. classifies the ticket **intent** into a fixed label set,
2. **retrieves** the 1–3 most relevant help-centre articles from a local KB,
3. **generates a grounded reply draft** with a **confidence** score, **escalation** signal and **safety flags**.

The emphasis is a clean, testable pipeline that separates retrieval, prompting and validation, and
behaves safely on edge cases — not model quality.

**Non-goals:** web API, database, real account lookups, multilingual replies, model training.

## 2. Guiding principles

| Principle | Consequence |
|---|---|
| **Runs from a clean checkout, without secrets** | Default path is fully offline (rules + TF-IDF + templates). LLM and embeddings are opt-in. |
| **Everything has a fallback** | Every optional component degrades to an offline one; one bad ticket never stops the run. |
| **Deterministic by default** | Same inputs → byte-identical `results.json`. No timestamps or randomness in results. |
| **Grounded or escalated** | A reply is either traceable to retrieved articles or flagged `insufficient_grounding` and escalated. |
| **Safety over helpfulness** | Trading advice is always refused; no outcome promises; no invented account facts. |
| **Swappable components** | Classifier, retriever and generator are interfaces selected by name. |
| **Nothing tied to sample IDs** | No logic depends on ticket IDs or article IDs; the evaluator may swap tickets. |

## 3. Inputs and outputs

### Inputs (read from disk)

- `tickets.json` — list of `{id, message, language}`.
- `kb_articles.json` — list of `{article_id, title, body}`.

Paths default to the repository root and can be overridden with `--tickets` / `--kb`.

### Outputs

| File | Content |
|---|---|
| `results.json` | One object per input ticket, in input order, exactly the required schema. |
| `debug_report.md` | Per ticket: predicted intent + matched terms, retrieved article titles + scores, confidence breakdown, escalation reasons, components used, fallbacks. |
| `debug_report.json` | Same information, machine-readable. |

### Required result schema

```json
{
  "ticket_id": "string",
  "intent": "deposit_issue | password_reset | withdrawal_issue | account_lock | trading_advice_request | other",
  "retrieved_articles": ["article_id"],
  "reply_draft": "string",
  "grounded": true,
  "confidence": 0.0,
  "needs_human_escalation": false,
  "safety_flags": ["string"]
}
```

`results.json` contains only these 8 fields. All diagnostics go to the debug report.

## 4. Pipeline

```
LOAD_DATA -> INDEX_KB -> CLASSIFY_INTENT -> RETRIEVE_CONTEXT -> GENERATE_REPLY -> VALIDATE_OUTPUT -> WRITE_RESULTS
```

Each stage is a named function in `support_pipeline/pipeline.py`. Stages 3–6 run per ticket inside a
`try/except`; an unexpected error produces a safe escalated result for that ticket only.

### 4.1 LOAD_DATA

- Parse both files into Pydantic models (`Ticket`, `Article`).
- **Missing file or invalid JSON → clear error message, non-zero exit.** Nothing to process; fabricating results would be worse than failing.
- Empty ticket list → `results.json` is `[]`.
- Malformed ticket entry (missing `message`, wrong types, empty text) → kept, and produces a schema-valid result: `intent=other`, escalated, flag `invalid_input`. Missing `id` → placeholder `UNKNOWN_<index>`.
- Very long messages are truncated for processing (limit in config).
- Non-`en` language → processed, flag `unsupported_language`, escalated.

### 4.2 INDEX_KB

- Shared text normalisation (`text.py`): lowercase, strip punctuation, stopword removal, light suffix stemmer, synonym map (e.g. `login→sign-in`, `money→deposit/balance`, `buy/invest/coin/gold→asset/trading`).
- Build two TF-IDF indexes over `title + body`:
  - **word-level** TF-IDF on normalised tokens (exact terms),
  - **character n-gram** TF-IDF (`char_wb`, 3–5) for morphological variants (`deposited`/`deposit`, `locked`/`lock`).
- Optional **embedding index** (MiniLM `all-MiniLM-L6-v2`, normalised vectors) when `--retriever hybrid`.
- **Article intent tagging:** run the rule classifier over each article's title + body once. Used only for the
  consistency check in confidence scoring — no hardcoded article-ID mapping.

### 4.3 CLASSIFY_INTENT

Interface: `classify(text) -> Classification(label, certainty, scores, matched_terms, component, fallback_reason)`

**Default — `RuleClassifier`:** weighted phrase/keyword patterns per intent on normalised text.

| Intent | Strong signals (weight 3) | Supporting (1–2) |
|---|---|---|
| `deposit_issue` | deposit, top up, funded | card, bank transfer, balance, payment not showing |
| `password_reset` | forgot password, reset email, reset link | password, email not received |
| `withdrawal_issue` | withdraw, withdrawal, cash out, payout | declined, rejected, pending, verification |
| `account_lock` | locked, lockout, blocked account, too many attempts | login attempts, can't sign in, cooldown |
| `trading_advice_request` | which asset, go up/down, should I buy/sell, predict, profit, signal, tip | market, invest, today, trade |

- Score = sum of matched weights. Highest wins; ties broken by fixed label order (deterministic).
- Top score below `MIN_INTENT_SCORE` (≈2) → `other`.
- `certainty` combines **margin** `top / (top + runner_up)` and **strength** (a single weak hit lowers it). Range 0–1.
- Shared terms (`balance`, `card`, `login`) stay weak; direction words (deposit vs withdraw, forgot/reset vs locked/attempts) carry the strong weights.
- `verification` alone does not imply `withdrawal_issue`.

**Advice guard (always on, independent of classifier):** detects prediction / recommendation / profit-seeking language.
If it fires, flag `advice_request` and the reply must contain a refusal. A mixed ticket ("deposit missing, also which coin should I buy?")
keeps the actionable support intent (if it has at least one strong signal) but still gets the flag and the refusal sentence.

**Optional — `GeminiClassifier` (`--classifier gemini`):** `google-genai`, temperature 0, response schema with `intent` as an enum of the 6 labels,
returns `{intent, confidence, rationale}`. Invalid label, bad JSON, timeout or any error → `RuleClassifier` for that ticket.

### 4.4 RETRIEVE_CONTEXT

Interface: `retrieve(query_text, k) -> list[Hit(article_id, score, components)]`

- **Query = ticket text only.** Retrieval does not use the predicted intent (kept independent).
- **Default — `LexicalRetriever`:** fused score `0.6·word + 0.4·char` (cosine similarities).
- **Optional — `HybridRetriever` (`--retriever hybrid`):** `0.5·word + 0.3·char + 0.2·semantic`.
- **Weighted score fusion, not reciprocal rank fusion** — confidence and the "no relevant article" check need absolute relevance, which ranks lose (especially with 5 documents).
- Return top `k=3` with fused score ≥ `MIN_RETRIEVAL_SCORE`; keep at least the top hit if it passes; ties broken by `article_id`.
- Nothing above threshold → empty list → `grounded=false`, `insufficient_grounding`, escalation.
- Scores rounded to 6 decimals for reproducibility.

### 4.5 GENERATE_REPLY

Interface: `generate(ticket, classification, hits, articles) -> Draft(text, cited_article_ids, component, fallback_reason)`

**Default — `TemplateGenerator` (offline, deterministic):**
1. Short acknowledgement sentence chosen by intent (generic, no facts).
2. Guidance sentences **extracted from the retrieved articles** (sentence split, ranked by overlap with the ticket), rewritten into client-facing form only by fixed prefixes, each ending with a citation like `[A1]`.
3. Fixed closing line (no promises).
- `trading_advice_request` → polite refusal grounded in the no-advice article if retrieved; refusal is emitted regardless of retrieval.
- No articles → safe holding reply ("a specialist will review your request") — no claims.
- Answers are composed from retrieved content, never hardcoded per ticket.

**Optional — `GeminiGenerator` (`--generator gemini`):**
- Prompt contains **only** the ticket and the retrieved snippets (numbered by article ID) and instructs the model to:
  answer only from the snippets, cite every factual sentence with `[article_id]`, say when something is not covered,
  never promise outcomes or timelines, never state account-specific facts, refuse trading advice.
- Temperature 0, JSON response `{reply, citations, unanswered_points}`.
- Any error or invalid output → `TemplateGenerator` for that ticket.

### 4.6 VALIDATE_OUTPUT

**Schema (Pydantic `TicketResult`):** required fields present, `intent` is a `Literal` of allowed labels,
`confidence` ∈ [0, 1], `retrieved_articles` ⊆ known KB IDs and length ≤ 3, `safety_flags` from the known set.

**Content rules:**

| Check | On failure |
|---|---|
| Advice request (intent or advice guard) → reply contains refusal and no prediction/recommendation terms | Replace with refusal template |
| Every cited ID is in `retrieved_articles` | `grounded=false`, `insufficient_grounding` |
| Each substantive sentence has a citation and token overlap with its cited article ≥ threshold | `grounded=false`, `insufficient_grounding` |
| No outcome promises (`will be approved`, `guarantee`, `within N hours`, `refund will`) | flag `unsupported_claim`, escalate |
| No invented account facts (`your account shows`, `your balance is`, `we can see`) | flag `unsupported_claim`, escalate |

A draft that cannot be repaired is replaced by the safe holding reply and escalated. Fallback results go through the same validator.

### 4.7 WRITE_RESULTS

- `results.json`: list in input order, fields in schema order, 2-space indent, UTF-8.
- Atomic write: write to `*.tmp`, then `os.replace`.
- Debug reports written the same way.
- Console summary: `N tickets | X escalated | Y fallbacks | components: rules/lexical/template`.

## 5. Confidence and escalation

```
confidence = 0.4 · classifier_certainty
           + 0.4 · retrieval_strength      (top fused score, normalised by config cap)
           + 0.2 · grounding_score         (share of grounded sentences)
           − penalties
```

Penalties (clamped to [0, 1] after):
- **conflict** — top article's intent tag ≠ predicted intent (−0.15),
- **no relevant article** (confidence capped at 0.3),
- **fallback used** for the ticket (−0.05).

**Safety flags**

| Flag | Raised when |
|---|---|
| `advice_request` | Advice guard fires (trading / prediction / profit request) |
| `insufficient_grounding` | No article retrieved or grounding check fails |
| `policy_sensitive` | Intent is policy-sensitive: `withdrawal_issue` (compliance / review outcomes) |
| `low_confidence` | `confidence < 0.5` |
| `account_specific_request` | Ticket asks about own balance / status / transaction ("my balance", "my withdrawal") |
| `unsupported_claim` | Draft contains promises or account facts |
| `invalid_input`, `unsupported_language`, `processing_error` | Input / runtime edge cases |

**`needs_human_escalation = true`** when any of: `low_confidence`, `insufficient_grounding`, `policy_sensitive`,
`unsupported_claim`, `intent == other`, `invalid_input`, `unsupported_language`, `processing_error`.

Not escalated on their own: `advice_request` (the refusal fully handles it) and `account_specific_request` (informational).

Escalation reasons are listed per ticket in the debug report.

**Expected on the sample data**

| Ticket | Intent | Top article | Escalate | Flags |
|---|---|---|---|---|
| T1 | deposit_issue | A1 | No | `account_specific_request` |
| T2 | password_reset | A2 | No | — |
| T3 | withdrawal_issue | A3 | Yes | `policy_sensitive`, `account_specific_request` |
| T4 | trading_advice_request | A5 | No | `advice_request` |
| T5 | account_lock | A4 | No | — |

## 6. Fallback matrix

| Component | Failure | Fallback |
|---|---|---|
| `google-genai` / `python-dotenv` | not installed | rules classifier, template generator |
| `GEMINI_API_KEY` | missing | same offline path, no error |
| Gemini call | timeout / rate limit / network | 1 retry, then offline component for that ticket |
| Gemini output | invalid JSON / label / ungrounded | offline component for that ticket |
| `sentence-transformers` / torch / model | missing or load error | lexical retrieval, re-weighted |
| Retrieval | empty KB / no hit above threshold | empty list, escalate |
| Ticket entry | malformed / empty / non-English | escalated result with flag |
| Validator | content check fails | safe holding reply or refusal, escalate |
| Any stage | unexpected exception | per-ticket safe result, `processing_error`, escalate |
| Output write | crash mid-write | atomic temp-file write |
| Input files | missing / invalid JSON | **hard fail**, clear message, non-zero exit |

Every fallback is recorded in the debug report with its reason.

## 7. Project layout

```
support-ticket-service/
  main.py                    # python main.py [--classifier rules|gemini] [--retriever lexical|hybrid]
                             #                [--generator template|gemini] [--tickets] [--kb] [--out-dir]
  validate.py                # end-to-end validation script
  SPEC.md  README.md
  requirements.txt           # core (offline): pydantic, scikit-learn, numpy, pytest
  requirements-llm.txt       # optional: google-genai, python-dotenv
  requirements-semantic.txt  # optional: sentence-transformers (+ CPU torch)
  .env (not committed)  .gitignore
  tickets.json  kb_articles.json            # inputs
  results.json  debug_report.md  debug_report.json   # generated
  support_pipeline/
    config.py      # thresholds, weights, model names
    schemas.py     # Ticket, Article, TicketResult, Classification, Hit, Draft
    data.py        # LOAD_DATA
    text.py        # normalisation, stemming, synonyms, sentence split
    classifier.py  # RuleClassifier, GeminiClassifier, advice guard
    retriever.py   # LexicalRetriever, HybridRetriever
    generator.py   # TemplateGenerator, GeminiGenerator
    validator.py   # schema + content rules
    scoring.py     # confidence, flags, escalation
    llm.py         # guarded google-genai client
    pipeline.py    # stage functions + orchestration + component factory
    report.py      # results + debug report writers
  tests/
    test_pipeline.py
```

## 8. Validation and tests

**`python validate.py`** (exits non-zero on any failure):
1. Input and output files exist; all JSON parses.
2. Exactly one result per ticket, matching IDs, input order.
3. Every result matches the schema (Pydantic + exact field set).
4. All `retrieved_articles` are known KB IDs, 1–3 entries (or empty with escalation).
5. Any ticket classified `trading_advice_request` is refused, flagged `advice_request`, contains no recommendation terms (T4 on sample data — checked by intent, not ID).
6. Reproducibility: runs the pipeline twice in-process (default offline components) and compares output byte-for-byte.

**`pytest`**:
- T4-style ticket → `trading_advice_request`, refused.
- T1-style ticket → retrieves the deposit article.
- Output JSON matches the schema.
- All 5 samples get expected intents.
- Paraphrased tickets reach the right article ("money hasn't shown up after I paid" → deposit article).
- Out-of-scope ticket → `other`, escalated.
- Mixed deposit + advice ticket → `deposit_issue` with `advice_request` and refusal sentence.
- Malformed ticket → escalated result, other tickets unaffected.
- Fake Gemini returning bad JSON / invalid label → fallback used, result still valid.
- Missing optional packages / key → offline run still passes validation.
- Determinism: two runs identical.

## 9. Swap boundary

`pipeline.build_components(classifier, retriever, generator)` returns objects implementing three small
`Protocol`s. Adding a new model means one class and one registry entry; no pipeline changes.

| Slot | Now | Later |
|---|---|---|
| Classifier | `RuleClassifier` | `GeminiClassifier` (included), fine-tuned model |
| Retriever | `LexicalRetriever` | `HybridRetriever` (included), vector DB |
| Generator | `TemplateGenerator` | `GeminiGenerator` (included), other hosted LLM |

## 10. Run commands

```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python main.py
python validate.py
pytest -q
```

Optional:
```
pip install -r requirements-llm.txt            # then: python main.py --generator gemini --classifier gemini
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-semantic.txt       # then: python main.py --retriever hybrid
```

## 11. Known limitations

- Rule classifier and synonym map are tuned to the domain of the provided KB; new topics need new patterns (or the Gemini classifier).
- TF-IDF statistics on a 5-article corpus are noisy; thresholds are calibrated on the sample data plus a few paraphrases.
- Template replies are extractive and less fluent than LLM replies, by design (traceable and deterministic).
- Grounding check is lexical overlap, not semantic entailment.
- English only; other languages are escalated.
