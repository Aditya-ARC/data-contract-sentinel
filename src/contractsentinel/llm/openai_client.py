"""OpenAI client with tool calling.

Two APIs are supported:
  * ``responses``: OpenAI's current API. Newer reasoning models need it for tool
    calling. This is the default when talking to OpenAI itself.
  * ``chat``: Chat Completions. The default when OPENAI_BASE_URL is set, because
    most OpenAI-compatible servers (Ollama, Gemini, GitHub Models) speak this one.

Override with OPENAI_API=responses|chat. Not every compatible server supports tools
well, so test the agent against it before relying on it.
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
DEFAULT_MODEL = "gpt-6-luna"


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


def to_responses_input(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert neutral messages to Responses API input items (stateless, no stored ids)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {"type": "function_call_output", "call_id": m.tool_call_id, "output": m.content or ""}
            )
        elif m.role == "assistant":
            if m.content:
                out.append({"role": "assistant", "content": m.content})
            for tc in m.tool_calls:
                out.append(
                    {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.raw_arguments or json.dumps(tc.arguments),
                    }
                )
        else:
            out.append({"role": m.role, "content": m.content or ""})
    return out


def to_responses_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    # strict=False: our schemas have optional fields, which strict mode would reject.
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
            "strict": False,
        }
        for t in tools
    ]


def _parse_arguments(call_id: str, name: str, text: str) -> ToolCall:
    try:
        parsed = json.loads(text) if text else {}
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be a JSON object")
        return ToolCall(id=call_id, name=name, arguments=parsed, raw_arguments=text)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        return ToolCall(id=call_id, name=name, arguments={}, raw_arguments=text, parse_error=str(exc))


def parse_tool_call(raw: Any) -> ToolCall:
    return _parse_arguments(raw.id, raw.function.name, raw.function.arguments or "")


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_output_tokens: int | None = None,
        client: Any | None = None,
        api: str | None = None,
    ) -> None:
        """Pass ``client`` to inject a stub (used by tests)."""
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        self.max_output_tokens = max_output_tokens

        base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.api = (api or os.getenv("OPENAI_API") or ("chat" if base_url else "responses")).lower()
        if self.api not in ("chat", "responses"):
            raise LLMConfigError(f"OPENAI_API must be 'chat' or 'responses', got '{self.api}'.")

        if client is not None:
            self._client = client
            return

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            if base_url:
                key = "not-needed"  # local servers ignore the key but the SDK requires one
            else:
                raise LLMConfigError(
                    "OPENAI_API_KEY is not set. Add it to your environment or .env file."
                )

        # CI passes unset variables as empty strings, and the SDK reads this variable itself.
        # An empty value would be taken as a real (invalid) address, so drop it.
        if not os.environ.get("OPENAI_BASE_URL"):
            os.environ.pop("OPENAI_BASE_URL", None)

        from openai import OpenAI

        self._client = OpenAI(api_key=key, base_url=base_url)

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        if self.api == "responses":
            return self._chat_responses(messages, tools)
        return self._chat_completions(messages, tools)

    def _chat_responses(self, messages: list[Message], tools: list[ToolSpec] | None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": to_responses_input(messages),
            "store": False,
        }
        if tools:
            kwargs["tools"] = to_responses_tools(tools)
        if self.max_output_tokens:
            kwargs["max_output_tokens"] = self.max_output_tokens

        response = self._client.responses.create(**kwargs)
        texts: list[str] = []
        calls: list[ToolCall] = []
        for item in response.output or []:
            kind = getattr(item, "type", None)
            if kind == "function_call":
                calls.append(_parse_arguments(item.call_id, item.name, item.arguments or ""))
            elif kind == "message":
                for part in getattr(item, "content", None) or []:
                    if getattr(part, "type", None) == "output_text":
                        texts.append(part.text)
        usage = getattr(response, "usage", None)
        return LLMResponse(
            content="".join(texts) or None,
            tool_calls=calls,
            usage=Usage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            ),
            model=getattr(response, "model", self.model),
        )

    def _chat_completions(self, messages: list[Message], tools: list[ToolSpec] | None) -> LLMResponse:
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
