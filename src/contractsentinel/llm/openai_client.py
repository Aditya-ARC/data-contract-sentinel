"""OpenAI client (Chat Completions with tool calling).

Also works with any OpenAI-compatible endpoint by setting OPENAI_BASE_URL, for
example a local Ollama server. Not every compatible server supports tools well,
so test the agent against it before relying on it.
"""

from __future__ import annotations

import json
import os
from typing import Any

from contractsentinel.llm.base import (
    LLMClient,
    LLMConfigError,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)

# Override with OPENAI_MODEL. Check OpenAI's models page for what is current.
DEFAULT_MODEL = "gpt-5-mini"


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or ""})
        elif m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.raw_arguments or json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": m.role, "content": m.content or ""})
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def parse_tool_call(raw: Any) -> ToolCall:
    text = raw.function.arguments or ""
    try:
        parsed = json.loads(text) if text else {}
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be a JSON object")
        return ToolCall(id=raw.id, name=raw.function.name, arguments=parsed, raw_arguments=text)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        return ToolCall(
            id=raw.id,
            name=raw.function.name,
            arguments={},
            raw_arguments=text,
            parse_error=str(exc),
        )


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_output_tokens: int | None = None,
        client: Any | None = None,
    ) -> None:
        """Pass ``client`` to inject a stub (used by tests)."""
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        self.max_output_tokens = max_output_tokens

        if client is not None:
            self._client = client
            return

        base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            if base_url:
                key = "not-needed"  # local servers ignore the key but the SDK requires one
            else:
                raise LLMConfigError(
                    "OPENAI_API_KEY is not set. Add it to your environment or .env file."
                )

        from openai import OpenAI

        self._client = OpenAI(api_key=key, base_url=base_url)

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        if self.max_output_tokens:
            kwargs["max_completion_tokens"] = self.max_output_tokens

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=message.content,
            tool_calls=[parse_tool_call(tc) for tc in (message.tool_calls or [])],
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            model=getattr(response, "model", self.model),
        )
