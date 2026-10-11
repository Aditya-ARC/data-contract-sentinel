from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from contractsentinel.llm import (
    LLMClient,
    LLMConfigError,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
    get_llm_client,
    register_provider,
)
from contractsentinel.llm.factory import PROVIDERS
from contractsentinel.llm.openai_client import (
    DEFAULT_MODEL,
    OpenAIClient,
    parse_tool_call,
    to_openai_messages,
    to_openai_tools,
)
from contractsentinel.llm.scripted import ScriptedLLMClient, text_response, tool_response

LLM_ENV = ["LLM_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL"]

READ_TOOL = ToolSpec(
    name="read_file",
    description="Read a file from the repository.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)


@pytest.fixture()
def clean_env(monkeypatch):
    for key in LLM_ENV:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def raw_tool_call(call_id="call_1", name="read_file", arguments='{"path": "a.py"}'):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def stub_openai(message, usage=(11, 7), model="stub-model"):
    """A stand-in for openai.OpenAI that records the request."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]),
            model=model,
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls


# ---- factory --------------------------------------------------------------------------


def test_factory_defaults_to_openai(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    client = get_llm_client()
    assert isinstance(client, OpenAIClient) and client.provider == "openai"
    assert client.model == DEFAULT_MODEL


def test_model_comes_from_the_environment(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    clean_env.setenv("OPENAI_MODEL", "my-model")
    assert get_llm_client("openai").model == "my-model"


def test_missing_api_key_is_a_clear_error(clean_env):
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        get_llm_client()


def test_base_url_allows_keyless_local_servers(clean_env):
    clean_env.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert isinstance(get_llm_client(), OpenAIClient)


def test_unknown_provider_lists_the_available_ones(clean_env):
    clean_env.setenv("LLM_PROVIDER", "mystery")
    with pytest.raises(LLMConfigError, match="Available: openai"):
        get_llm_client()


def test_providers_can_be_registered(clean_env):
    scripted = ScriptedLLMClient([])
    register_provider("scripted-test", lambda: scripted)
    try:
        assert get_llm_client("scripted-test") is scripted
        assert isinstance(scripted, LLMClient)
    finally:
        PROVIDERS.pop("scripted-test", None)


# ---- translation ----------------------------------------------------------------------


def test_messages_round_trip_a_tool_call_to_openai_format():
    call = ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})
    messages = [
        Message.system("sys"),
        Message.user("go"),
        Message.assistant(tool_calls=[call]),
        Message.tool_result("c1", "file contents"),
        Message.assistant("done"),
    ]
    out = to_openai_messages(messages)

    assert out[0] == {"role": "system", "content": "sys"}
    assert out[2]["role"] == "assistant" and out[2]["content"] is None
    sent = out[2]["tool_calls"][0]
    assert sent["id"] == "c1" and sent["type"] == "function"
    assert json.loads(sent["function"]["arguments"]) == {"path": "a.py"}
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "file contents"}
    assert out[4] == {"role": "assistant", "content": "done"}


def test_tools_are_translated_to_function_schemas():
    (tool,) = to_openai_tools([READ_TOOL])
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "read_file"
    assert tool["function"]["parameters"]["required"] == ["path"]


def test_tool_call_arguments_are_parsed():
    call = parse_tool_call(raw_tool_call())
    assert call.arguments == {"path": "a.py"} and call.parse_error is None


@pytest.mark.parametrize("bad", ["{not json", "[1, 2]", '"text"'])
def test_bad_tool_arguments_are_flagged_not_raised(bad):
    call = parse_tool_call(raw_tool_call(arguments=bad))
    assert call.arguments == {} and call.parse_error
    assert call.raw_arguments == bad


def test_empty_tool_arguments_mean_no_arguments():
    assert parse_tool_call(raw_tool_call(arguments="")).arguments == {}


# ---- OpenAIClient.chat ----------------------------------------------------------------


def test_chat_returns_text_and_usage():
    client, calls = stub_openai(SimpleNamespace(content="hello", tool_calls=None))
    llm = OpenAIClient(model="m", client=client, api="chat")

    response = llm.chat([Message.user("hi")])

    assert response.content == "hello" and response.tool_calls == []
    assert response.usage == Usage(input_tokens=11, output_tokens=7)
    assert response.model == "stub-model"
    assert calls[0]["model"] == "m"
    assert "tools" not in calls[0] and "max_completion_tokens" not in calls[0]


def test_chat_sends_tools_and_parses_tool_calls():
    message = SimpleNamespace(content=None, tool_calls=[raw_tool_call()])
    client, calls = stub_openai(message)
    llm = OpenAIClient(model="m", client=client, max_output_tokens=500, api="chat")

    response = llm.chat([Message.user("go")], tools=[READ_TOOL])

    assert calls[0]["tools"][0]["function"]["name"] == "read_file"
    assert calls[0]["max_completion_tokens"] == 500
    assert response.tool_calls[0].name == "read_file"
    assert response.to_message().tool_calls == response.tool_calls


# ---- scripted client ------------------------------------------------------------------


def test_scripted_client_replays_and_records():
    llm = ScriptedLLMClient([tool_response("read_file", {"path": "x"}), text_response("done")])
    first = llm.chat([Message.user("a")], tools=[READ_TOOL])
    second = llm.chat([Message.user("b")])

    assert first.tool_calls[0].arguments == {"path": "x"}
    assert second.content == "done"
    assert len(llm.requests) == 2 and llm.requests[0][1] == [READ_TOOL]


def test_scripted_client_fails_loudly_when_exhausted():
    llm = ScriptedLLMClient([])
    with pytest.raises(RuntimeError, match="ran out"):
        llm.chat([Message.user("a")])


def test_response_type_is_importable():
    assert LLMResponse(content="x", tool_calls=[], usage=Usage(), model="m").content == "x"


def test_empty_base_url_env_is_ignored(clean_env):
    """CI passes unset variables as empty strings; the SDK must not see one as an address."""
    import os

    clean_env.setenv("OPENAI_BASE_URL", "")
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    client = get_llm_client("openai")
    assert "OPENAI_BASE_URL" not in os.environ
    assert "api.openai.com" in str(client._client.base_url)


# ---- Responses API --------------------------------------------------------------------


def stub_responses(output, usage=(21, 9), model="stub-resp"):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            output=output,
            usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]),
            model=model,
        )

    return SimpleNamespace(responses=SimpleNamespace(create=create)), calls


def test_api_defaults_to_responses_for_openai_and_chat_for_custom_urls(clean_env):
    assert OpenAIClient(model="m", client=object()).api == "responses"
    clean_env.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert OpenAIClient(model="m", client=object()).api == "chat"
    clean_env.setenv("OPENAI_API", "responses")
    assert OpenAIClient(model="m", client=object()).api == "responses"


def test_invalid_api_setting_is_a_config_error(clean_env):
    clean_env.setenv("OPENAI_API", "soap")
    with pytest.raises(LLMConfigError):
        OpenAIClient(model="m", client=object())


def test_responses_input_maps_every_message_kind():
    from contractsentinel.llm.base import ToolCall
    from contractsentinel.llm.openai_client import to_responses_input, to_responses_tools

    call = ToolCall("c1", "read_file", {"path": "a"}, raw_arguments='{"path": "a"}')
    items = to_responses_input(
        [
            Message.system("s"),
            Message.user("u"),
            Message.assistant("thinking", [call]),
            Message.tool_result("c1", "file text"),
            Message.assistant("done"),
        ]
    )
    assert items == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "thinking"},
        {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": '{"path": "a"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "file text"},
        {"role": "assistant", "content": "done"},
    ]
    assert all("id" not in i for i in items)  # stored item ids would require reasoning items
    tool = to_responses_tools([READ_TOOL])[0]
    assert tool["name"] == "read_file" and tool["strict"] is False and "function" not in tool


def test_responses_text_and_usage():
    out = [SimpleNamespace(type="reasoning"), SimpleNamespace(
        type="message", content=[SimpleNamespace(type="output_text", text="hel"), SimpleNamespace(type="output_text", text="lo")]
    )]
    client, calls = stub_responses(out)
    response = OpenAIClient(model="m", client=client, api="responses").chat([Message.user("hi")])
    assert response.content == "hello" and response.tool_calls == []
    assert response.usage == Usage(21, 9) and response.model == "stub-resp"
    assert calls[0]["store"] is False and "tools" not in calls[0]


def test_responses_tool_calls_and_bad_arguments():
    out = [
        SimpleNamespace(type="function_call", call_id="c1", name="read_file", arguments='{"path": "a.py"}'),
        SimpleNamespace(type="function_call", call_id="c2", name="read_file", arguments="{oops"),
    ]
    client, calls = stub_responses(out)
    llm = OpenAIClient(model="m", client=client, api="responses", max_output_tokens=300)
    response = llm.chat([Message.user("go")], tools=[READ_TOOL])
    assert calls[0]["tools"][0]["name"] == "read_file" and calls[0]["max_output_tokens"] == 300
    assert response.content is None
    assert response.tool_calls[0].arguments == {"path": "a.py"}
    assert response.tool_calls[1].parse_error
