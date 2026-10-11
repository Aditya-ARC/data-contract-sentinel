"""Provider-neutral LLM types.

The investigation agent only ever sees these types. Each provider client
translates them to and from its own API, so swapping models is a config change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


class LLMConfigError(RuntimeError):
    """Raised when a provider is misconfigured (missing key, unknown provider)."""


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call. ``parameters`` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    # Set when the model produced arguments that were not valid JSON.
    parse_error: str | None = None


@dataclass
class Message:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool" messages

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None = None, tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    usage: Usage
    model: str

    def to_message(self) -> Message:
        return Message.assistant(self.content, self.tool_calls)


class LLMClient(ABC):
    provider: str
    model: str

    @abstractmethod
    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        """Send a conversation and return the model's next turn."""
