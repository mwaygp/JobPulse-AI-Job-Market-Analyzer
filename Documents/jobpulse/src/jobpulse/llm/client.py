"""Thin wrapper around the Anthropic SDK. Returns None when no key/package is available,
so every LLM feature has an offline fallback instead of crashing."""
from __future__ import annotations

import os

from jobpulse.config import CLAUDE_MODEL


def get_client():
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic()


def complete(client, system: str, user: str, max_tokens: int = 1000) -> str:
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
