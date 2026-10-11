"""Pick an LLM provider from configuration.

To add a provider (Anthropic, Gemini, Ollama native, ...), implement LLMClient
and register a factory function here. The agent code does not change.
"""

from __future__ import annotations

import os
from typing import Callable

from contractsentinel.llm.base import LLMClient, LLMConfigError


def _make_openai() -> LLMClient:
    from contractsentinel.llm.openai_client import OpenAIClient

    return OpenAIClient()


PROVIDERS: dict[str, Callable[[], LLMClient]] = {
    "openai": _make_openai,
}


def register_provider(name: str, factory: Callable[[], LLMClient]) -> None:
    PROVIDERS[name] = factory


def get_llm_client(provider: str | None = None) -> LLMClient:
    """Build a client for ``provider`` or the LLM_PROVIDER env var (default: openai)."""
    name = (provider or os.getenv("LLM_PROVIDER") or "openai").lower()
    try:
        factory = PROVIDERS[name]
    except KeyError:
        raise LLMConfigError(
            f"Unknown LLM provider {name!r}. Available: {', '.join(sorted(PROVIDERS))}."
        ) from None
    return factory()
