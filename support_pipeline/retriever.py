"""INDEX_KB + RETRIEVE_CONTEXT: word + char TF-IDF (default), optional MiniLM semantic layer, weighted fusion."""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    EMBEDDING_MODEL,
    HYBRID_WEIGHTS,
    LEXICAL_WEIGHTS,
    MIN_RETRIEVAL_SCORE,
    RELATIVE_SCORE_RATIO,
    TOP_K,
)
from .schemas import Article, Hit
from .text import canonical, query_tokens, tokens


class Retriever(Protocol):
    name: str
    fallback_reason: Optional[str]

    def retrieve(self, query: str, k: int = TOP_K) -> List[Hit]: ...


def _pretokenised(token_list: List[str]) -> List[str]:
    return token_list


def _doc_text(article: Article) -> str:
    return f"{article.title}. {article.body}"


def _select(article_ids: List[str], fused: np.ndarray, parts: Dict[str, np.ndarray], k: int) -> List[Hit]:
    """Deterministic top-k: score desc, then article_id; absolute floor plus relative-to-top cut."""
    order = sorted(range(len(article_ids)), key=lambda i: (-round(float(fused[i]), 6), article_ids[i]))
    hits: List[Hit] = []
    top_score = float(fused[order[0]]) if order else 0.0
    for i in order[:k]:
        score = round(float(fused[i]), 6)
        if score < MIN_RETRIEVAL_SCORE or score < top_score * RELATIVE_SCORE_RATIO:
            break
        hits.append(Hit(article_ids[i], score, {name: round(float(v[i]), 6) for name, v in parts.items()}))
    return hits


class LexicalRetriever:
    """Word TF-IDF (stemmed tokens + query expansion) fused with char n-gram TF-IDF (morphology)."""

    name = "lexical"

    def __init__(self, articles: List[Article]):
        self.fallback_reason: Optional[str] = None
        self.article_ids = [a.article_id for a in articles]
        docs = [_doc_text(a) for a in articles]
        # Word index takes pre-tokenised input so documents and (expanded) queries share one tokenizer.
        self._word = TfidfVectorizer(analyzer=_pretokenised, sublinear_tf=True, norm="l2")
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), preprocessor=canonical,
                                     sublinear_tf=True, norm="l2")
        if docs:
            self._word_matrix = self._word.fit_transform([tokens(d) for d in docs])
            self._char_matrix = self._char.fit_transform(docs)

    def lexical_scores(self, query: str) -> Dict[str, np.ndarray]:
        word_q = self._word.transform([query_tokens(query)])  # query side gets synonym expansion
        char_q = self._char.transform([query])
        return {
            "word": (self._word_matrix @ word_q.T).toarray().ravel(),
            "char": (self._char_matrix @ char_q.T).toarray().ravel(),
        }

    def retrieve(self, query: str, k: int = TOP_K) -> List[Hit]:
        if not self.article_ids or not query.strip():
            return []
        parts = self.lexical_scores(query)
        fused = sum(LEXICAL_WEIGHTS[name] * parts[name] for name in LEXICAL_WEIGHTS)
        return _select(self.article_ids, fused, parts, k)


class HybridRetriever(LexicalRetriever):
    """Adds MiniLM cosine similarity. Construction raises if the model can't load (caller falls back)."""

    name = "hybrid"

    def __init__(self, articles: List[Article]):
        super().__init__(articles)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        from sentence_transformers import SentenceTransformer  # ImportError -> fallback

        try:  # prefer the local cache (offline), then allow a download
            self._model = SentenceTransformer(EMBEDDING_MODEL, device="cpu", local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        docs = [_doc_text(a) for a in articles]
        self._doc_emb = self._model.encode(docs, normalize_embeddings=True, show_progress_bar=False) if docs else None

    def retrieve(self, query: str, k: int = TOP_K) -> List[Hit]:
        if not self.article_ids or not query.strip():
            return []
        parts = self.lexical_scores(query)
        try:
            q = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            parts["semantic"] = np.clip(np.round(self._doc_emb @ q, 6), 0.0, 1.0)
            weights = HYBRID_WEIGHTS
            self.fallback_reason = None
        except Exception as exc:  # embedding failure at query time -> lexical scores only
            self.fallback_reason = f"semantic scoring failed ({type(exc).__name__}); lexical only"
            weights = LEXICAL_WEIGHTS
        fused = sum(weights[name] * parts[name] for name in weights)
        return _select(self.article_ids, fused, parts, k)


def build_retriever(name: str, articles: List[Article]) -> Retriever:
    if name == "hybrid":
        try:
            return HybridRetriever(articles)
        except Exception as exc:
            retriever = LexicalRetriever(articles)
            retriever.fallback_reason = f"hybrid unavailable ({type(exc).__name__}: {str(exc)[:120]}); using lexical"
            return retriever
    return LexicalRetriever(articles)
