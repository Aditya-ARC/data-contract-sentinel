from __future__ import annotations

import json

from contractsentinel.agent.loop import REMOVED_NOTE, AgentConfig, investigate, trim_history
from contractsentinel.agent.schema import SUBMIT_TOOL_NAME
from contractsentinel.agent.tools import ToolBox
from contractsentinel.llm.base import LLMResponse, Message, ToolCall, Usage
from contractsentinel.llm.scripted import ScriptedLLMClient, text_response, tool_response

REPORT = {
    "root_cause": "source_defect",
    "confidence": "high",
    "summary": "Bad statuses from the source.",
    "evidence": ["violations show status typos"],
    "impact": "7 rows quarantined",
    "recommended_actions": ["Contact the vendor"],
}


def box(incident_env, contract):
    return ToolBox(incident_env.repo, incident_env.batch_dir, contract)


def run(incident_env, contract, responses, **cfg):
    llm = ScriptedLLMClient(responses)
    return llm, investigate(llm, box(incident_env, contract), "brief", AgentConfig(**cfg))


def with_usage(resp: LLMResponse, i: int, o: int) -> LLMResponse:
    resp.usage = Usage(i, o)
    return resp


def test_submits_after_using_tools(incident_env, contract):
    llm, out = run(
        incident_env,
        contract,
        [tool_response("get_violations", {}), tool_response(SUBMIT_TOOL_NAME, REPORT, "c2")],
    )
    assert out.stop_reason == "submitted" and out.steps == 2
    assert out.analysis.root_cause == "source_defect"
    # the second request contains the tool result from the first
    second = llm.requests[1][0]
    assert second[-1].role == "tool" and "rule_id" in second[-1].content


def test_tools_offered_include_submit(incident_env, contract):
    llm, _ = run(incident_env, contract, [tool_response(SUBMIT_TOOL_NAME, REPORT)])
    assert SUBMIT_TOOL_NAME in [t.name for t in llm.requests[0][1]]


def test_rejected_report_is_fed_back_and_can_be_corrected(incident_env, contract):
    bad = dict(REPORT, root_cause="gremlins")
    llm, out = run(
        incident_env,
        contract,
        [tool_response(SUBMIT_TOOL_NAME, bad), tool_response(SUBMIT_TOOL_NAME, REPORT, "c2")],
    )
    assert out.stop_reason == "submitted" and out.steps == 2
    feedback = llm.requests[1][0][-1]
    assert feedback.role == "tool" and "report rejected" in feedback.content


def test_invalid_json_arguments_are_reported_to_the_model(incident_env, contract):
    broken = LLMResponse(
        None,
        [ToolCall("c1", "get_violations", {}, raw_arguments="{oops", parse_error="Expecting value")],
        Usage(),
        "scripted",
    )
    llm, out = run(incident_env, contract, [broken, tool_response(SUBMIT_TOOL_NAME, REPORT)])
    assert out.stop_reason == "submitted"
    assert "not valid JSON" in llm.requests[1][0][-1].content


def test_unknown_tool_does_not_crash(incident_env, contract):
    llm, out = run(
        incident_env, contract, [tool_response("rm_rf", {}), tool_response(SUBMIT_TOOL_NAME, REPORT, "c2")]
    )
    assert out.stop_reason == "submitted"
    assert "unknown tool" in llm.requests[1][0][-1].content


def test_text_only_reply_is_nudged_once_then_stops(incident_env, contract):
    llm, out = run(incident_env, contract, [text_response("hmm"), text_response("still thinking")])
    assert out.stop_reason == "no_tool_call" and out.analysis is None
    assert SUBMIT_TOOL_NAME in llm.requests[1][0][-1].content


def test_nudge_can_succeed(incident_env, contract):
    _, out = run(incident_env, contract, [text_response("hmm"), tool_response(SUBMIT_TOOL_NAME, REPORT)])
    assert out.stop_reason == "submitted"


def test_step_limit(incident_env, contract):
    responses = [tool_response("get_run_summary", {}, f"c{i}") for i in range(5)]
    _, out = run(incident_env, contract, responses, max_steps=3)
    assert out.stop_reason == "max_steps" and out.steps == 3 and out.analysis is None


def test_token_budget(incident_env, contract):
    responses = [with_usage(tool_response("get_run_summary", {}, f"c{i}"), 600, 500) for i in range(5)]
    _, out = run(incident_env, contract, responses, max_total_tokens=1000)
    assert out.stop_reason == "token_budget" and out.steps == 1
    assert out.usage.input_tokens == 600 and out.usage.output_tokens == 500


def test_usage_is_summed(incident_env, contract):
    responses = [
        with_usage(tool_response("get_run_summary", {}), 10, 5),
        with_usage(tool_response(SUBMIT_TOOL_NAME, REPORT, "c2"), 20, 7),
    ]
    _, out = run(incident_env, contract, responses)
    assert (out.usage.input_tokens, out.usage.output_tokens) == (30, 12)


def test_llm_errors_are_captured(incident_env, contract):
    _, out = run(incident_env, contract, [])  # the scripted client raises when empty
    assert out.stop_reason == "llm_error" and "RuntimeError" in out.error and out.steps == 0


# ---- history trimming -----------------------------------------------------------------


def test_trim_history_removes_oldest_tool_output_first_and_keeps_latest_two():
    msgs = [Message.system("s"), Message.user("u")]
    for i in range(5):
        msgs.append(Message.assistant(None, [ToolCall(f"c{i}", "t", {})]))
        msgs.append(Message.tool_result(f"c{i}", "x" * 1000))
    trimmed = trim_history(msgs, 2500)
    tool_contents = [m.content for m in trimmed if m.role == "tool"]
    assert tool_contents[0] == REMOVED_NOTE
    assert tool_contents[-1] == "x" * 1000 and tool_contents[-2] == "x" * 1000
    assert msgs[3].content == "x" * 1000  # the input list is not mutated


def test_trim_history_noop_when_small():
    msgs = [Message.system("s"), Message.tool_result("c", "small")]
    assert trim_history(msgs, 1000)[1].content == "small"


def test_tool_arguments_are_json_serializable_in_scripted_flow(incident_env, contract):
    _, out = run(incident_env, contract, [tool_response(SUBMIT_TOOL_NAME, REPORT)])
    json.dumps(out.analysis.evidence)
