"""A scripted LLM client for tests and offline development.

It replays a fixed list of responses and records every request, so the agent
loop can be tested without network access or API spend.
"""

from __future__ import annotations

from contractsentinel.llm.base import LLMClient, LLMResponse, Message, ToolCall, ToolSpec, Usage


class ScriptedLLMClient(LLMClient):
    provider = "scripted"

    def __init__(self, responses: list[LLMResponse], model: str = "scripted") -> None:
        self.model = model
        self._responses = list(responses)
        self.requests: list[tuple[list[Message], list[ToolSpec] | None]] = []

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        self.requests.append((list(messages), tools))
        if not self._responses:
            raise RuntimeError("ScriptedLLMClient ran out of scripted responses")
        return self._responses.pop(0)


def text_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[], usage=Usage(), model="scripted")


def tool_response(name: str, arguments: dict, call_id: str = "call_1") -> LLMResponse:
    call = ToolCall(id=call_id, name=name, arguments=arguments)
    return LLMResponse(content=None, tool_calls=[call], usage=Usage(), model="scripted")
