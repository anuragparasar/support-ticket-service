"""Optional Gemini client. Every import is guarded: if anything is missing the caller gets (None, reason)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from .config import GEMINI_MODEL_DEFAULT, LLM_RETRIES, LLM_TIMEOUT_MS

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


class GeminiClient:
    def __init__(self, api_key: str, model: str):
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=LLM_TIMEOUT_MS))

    def generate_json(self, prompt: str, schema: Any) -> dict:
        """Temperature-0 call constrained to a JSON schema. Retries once, then raises."""
        config = self._types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema,
        )
        last_error: Optional[Exception] = None
        for _ in range(1 + LLM_RETRIES):
            try:
                response = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
                if not response.text:
                    raise ValueError("empty response")
                data = json.loads(response.text)
                if not isinstance(data, dict):
                    raise ValueError("response is not a JSON object")
                return data
            except Exception as exc:  # network, quota, parsing: all retried once
                last_error = exc
        raise RuntimeError(f"Gemini call failed: {type(last_error).__name__}: {last_error}")


def get_gemini_client() -> Tuple[Optional[GeminiClient], Optional[str]]:
    """Returns (client, None) or (None, reason it is unavailable)."""
    _load_env()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None, "no GEMINI_API_KEY set"
    try:
        client = GeminiClient(api_key, os.getenv("GEMINI_MODEL", GEMINI_MODEL_DEFAULT))
    except ImportError:
        return None, "google-genai not installed (pip install -r requirements-llm.txt)"
    except Exception as exc:
        return None, f"Gemini client init failed: {type(exc).__name__}"
    return client, None
