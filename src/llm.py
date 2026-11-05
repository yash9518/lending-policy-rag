"""Claude client setup. The LLM is optional: without credentials the app still
runs retrieval and the rule engine, it just does not generate answers."""
from __future__ import annotations

import os

from src import config


def build_client():
    """Return (client, model, status_message). ``client`` is None when disabled."""
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        return None, config.LLM_MODEL, "LLM disabled: set ANTHROPIC_API_KEY to enable the agent."
    try:
        import anthropic
    except ImportError:
        return None, config.LLM_MODEL, "LLM disabled: `pip install anthropic` to enable the agent."
    return anthropic.Anthropic(max_retries=3), config.LLM_MODEL, f"LLM enabled: {config.LLM_MODEL}"


def response_text(response) -> str:
    """Concatenate the text blocks of a Messages API response."""
    return "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
