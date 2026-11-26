"""LOAD_DATA: read tickets and KB articles from disk into validated models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from pydantic import ValidationError

from .config import MAX_MESSAGE_CHARS
from .schemas import Article, Ticket


class InputDataError(Exception):
    """Raised when an input file is missing or unreadable. The only hard failure in the pipeline."""


def _read_json_list(path: Path, what: str) -> list:
    if not path.exists():
        raise InputDataError(f"{what} file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InputDataError(f"{what} file is not valid JSON ({path}): {exc}") from exc
    if not isinstance(data, list):
        raise InputDataError(f"{what} file must contain a JSON list: {path}")
    return data


def load_tickets(path: Path) -> List[Ticket]:
    """Malformed entries are kept (with input_errors) so every input still gets a result."""
    tickets: List[Ticket] = []
    seen_ids = set()
    for index, raw in enumerate(_read_json_list(path, "Tickets"), start=1):
        errors: List[str] = []
        if not isinstance(raw, dict):
            raw, errors = {}, ["entry is not an object"]
        ticket_id = raw.get("id")
        if not isinstance(ticket_id, (str, int)) or str(ticket_id).strip() == "":
            ticket_id = f"UNKNOWN_{index}"
            errors.append("missing id")
        ticket_id = str(ticket_id).strip()
        if ticket_id in seen_ids:
            errors.append("duplicate id")
            ticket_id = f"{ticket_id}_dup{index}"
        seen_ids.add(ticket_id)

        message = raw.get("message")
        if not isinstance(message, str) or not message.strip():
            errors.append("missing or empty message")
            message = ""
        elif len(message) > MAX_MESSAGE_CHARS:
            errors.append(f"message truncated to {MAX_MESSAGE_CHARS} chars")
            message = message[:MAX_MESSAGE_CHARS]
        language = raw.get("language")
        language = language.strip().lower() if isinstance(language, str) and language.strip() else "en"

        tickets.append(Ticket(id=ticket_id, message=message.strip(), language=language, input_errors=errors))
    return tickets


def load_articles(path: Path) -> Tuple[List[Article], List[str]]:
    """Returns valid articles plus warnings for skipped ones (duplicates / malformed)."""
    articles: List[Article] = []
    warnings: List[str] = []
    seen = set()
    for index, raw in enumerate(_read_json_list(path, "KB articles"), start=1):
        try:
            article = Article.model_validate(raw)
        except ValidationError as exc:
            warnings.append(f"KB entry {index} skipped: {exc.errors()[0]['msg']}")
            continue
        if article.article_id in seen:
            warnings.append(f"KB entry {index} skipped: duplicate article_id {article.article_id}")
            continue
        seen.add(article.article_id)
        articles.append(article)
    return articles, warnings
