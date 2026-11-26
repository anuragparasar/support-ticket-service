"""Tunable constants. Everything that affects results lives here so it is easy to review."""

INTENTS = (
    "deposit_issue",
    "password_reset",
    "withdrawal_issue",
    "account_lock",
    "trading_advice_request",
    "other",
)

SAFETY_FLAGS = (
    "advice_request",
    "insufficient_grounding",
    "policy_sensitive",
    "low_confidence",
    "account_specific_request",
    "unsupported_claim",
    "invalid_input",
    "unsupported_language",
    "processing_error",
)

# Flags that force needs_human_escalation=True on their own.
ESCALATING_FLAGS = {
    "insufficient_grounding",
    "policy_sensitive",
    "low_confidence",
    "unsupported_claim",
    "invalid_input",
    "unsupported_language",
    "processing_error",
}

POLICY_SENSITIVE_INTENTS = {"withdrawal_issue"}
SUPPORTED_LANGUAGES = {"en"}
MAX_MESSAGE_CHARS = 2000

# --- classification ---
MIN_INTENT_SCORE = 2.0      # below this the ticket is "other"
STRONG_INTENT_SCORE = 6.0   # score at which classifier strength saturates
OTHER_CERTAINTY = 0.3       # certainty assigned to an "other" prediction

# --- retrieval ---
TOP_K = 3
MIN_RETRIEVAL_SCORE = 0.15  # absolute floor on the fused score
RELATIVE_SCORE_RATIO = 0.5  # secondary hits must reach this fraction of the top score
LEXICAL_WEIGHTS = {"word": 0.6, "char": 0.4}
HYBRID_WEIGHTS = {"word": 0.5, "char": 0.3, "semantic": 0.2}
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- generation ---
MAX_REPLY_SENTENCES = 3
MAX_ARTICLES_IN_REPLY = 2

# --- grounding / confidence ---
MIN_SENTENCE_SUPPORT = 0.5      # share of a sentence's content tokens found in its cited articles
SUBSTANTIVE_MIN_TOKENS = 4      # shorter uncited sentences (greetings) are not treated as claims
RETRIEVAL_SCORE_CAP = 0.55     # fused score treated as "fully strong" retrieval
CONFIDENCE_WEIGHTS = {"classifier": 0.4, "retrieval": 0.4, "grounding": 0.2}
CONFLICT_PENALTY = 0.15
FALLBACK_PENALTY = 0.05
NO_HIT_CONFIDENCE_CAP = 0.3
LOW_CONFIDENCE_THRESHOLD = 0.5

# --- LLM (optional) ---
GEMINI_MODEL_DEFAULT = "gemini-3.5-flash"
LLM_TIMEOUT_MS = 20_000
LLM_RETRIES = 1
