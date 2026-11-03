import os

DEFAULT_LLM_MODEL = "claude-haiku-5-5"


def build_llm():
    """Return (llm, status_message).

    The LLM is optional. If ANTHROPIC_API_KEY is not set, or the
    langchain-anthropic package is not installed, this returns (None, reason)
    and the app falls back to showing retrieved clauses only.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None, "LLM disabled: set ANTHROPIC_API_KEY to enable generated answers."

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None, "LLM disabled: install langchain-anthropic to enable generated answers."

    model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    llm = ChatAnthropic(model=model, temperature=0, max_tokens=600)
    return llm, f"LLM enabled: {model}"


def message_text(response) -> str:
    """Extract plain text from a chat model response."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts).strip()
