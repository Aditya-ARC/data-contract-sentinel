from __future__ import annotations

import json
import os

import pytest

from contractsentinel.agent.mask import mask_value, mask_violations
from contractsentinel.agent.tools import MAX_OUTPUT_CHARS, ToolBox, clip


@pytest.fixture()
def toolbox(incident_env, contract):
    return ToolBox(incident_env.repo, incident_env.batch_dir, contract)


# ---- masking --------------------------------------------------------------------------


def test_mask_keeps_shape_and_hides_content():
    assert mask_value("C12345") == "A99999"
    assert mask_value("jane.doe-7@x.io") == "aaaa.aaa-9@a.aa"
    assert mask_value("") == ""


def test_only_sensitive_columns_are_masked(contract):
    doc = {
        "violations": [
            {"column": "customer_id", "samples": [{"row_number": 2, "value": "X123", "raw_value": "x123"}]},
            {"column": "status", "samples": [{"row_number": 3, "value": "shiped"}]},
        ]
    }
    masked = mask_violations(doc, contract)
    assert masked["violations"][0]["samples"][0] == {"row_number": 2, "value": "A999", "raw_value": "a999"}
    assert masked["violations"][1]["samples"][0]["value"] == "shiped"
    assert doc["violations"][0]["samples"][0]["value"] == "X123"  # input is not mutated


def test_get_violations_returns_masked_samples(incident_env, contract, toolbox):
    path = incident_env.batch_dir / "violations.json"
    doc = json.loads(path.read_text())
    doc["violations"].append(
        {
            "rule_id": "orders.customer_id.pattern",
            "column": "customer_id",
            "rule": "pattern",
            "description": "x",
            "severity": "error",
            "scope": "row",
            "count": 1,
            "samples": [{"row_number": 9, "value": "C98765"}],
        }
    )
    path.write_text(json.dumps(doc))
    out = toolbox.call("get_violations", {})
    assert "A99999" in out and "C98765" not in out


def test_run_summary_is_compact_and_complete(toolbox):
    data = json.loads(toolbox.call("get_run_summary", {}))
    assert data["status"] == "partially_loaded"
    assert data["counts"]["rows_quarantined"] == 7
    assert "source_sha256" not in data


# ---- read_file ------------------------------------------------------------------------


def test_read_file_returns_numbered_lines(toolbox):
    out = toolbox.call("read_file", {"path": "src/demo/transform.py"})
    assert out.startswith("src/demo/transform.py lines 1-")
    assert "1: def normalize_date(value):" in out


def test_read_file_honors_line_range(toolbox):
    out = toolbox.call("read_file", {"path": "src/demo/transform.py", "start_line": 2, "end_line": 2})
    assert "lines 2-2" in out and "1:" not in out and out.splitlines()[1].startswith("2:")


def test_read_file_caps_the_window(incident_env, contract):
    big = incident_env.repo / "docs" / "big.md"
    big.parent.mkdir()
    big.write_text("\n".join(f"line {i}" for i in range(1, 500)))
    out = ToolBox(incident_env.repo, incident_env.batch_dir, contract).call(
        "read_file", {"path": "docs/big.md", "start_line": 1, "end_line": 400}
    )
    assert "lines 1-120 of 499" in out


@pytest.mark.parametrize(
    "path",
    [
        "../etc/passwd",
        "/etc/passwd",
        "src/../../etc/passwd",
        ".env",
        "samples/orders_2026-10-02.csv",
        "runs/x/violations.json",
        ".git/config",
        "",
    ],
)
def test_read_file_refuses_unsafe_or_unlisted_paths(toolbox, path):
    out = toolbox.call("read_file", {"path": path})
    assert out.startswith("error:")


def test_read_file_refuses_symlinks_that_escape(incident_env, contract, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    os.symlink(secret, incident_env.repo / "src" / "link.txt")
    out = ToolBox(incident_env.repo, incident_env.batch_dir, contract).call(
        "read_file", {"path": "src/link.txt"}
    )
    assert out.startswith("error:") and "top secret" not in out


def test_read_file_missing_file_and_bad_arguments(toolbox):
    assert "not found" in toolbox.call("read_file", {"path": "src/nope.py"})
    assert toolbox.call("read_file", {}).startswith("error:")
    assert toolbox.call("read_file", {"path": "src/demo/transform.py", "start_line": "1"}).startswith("error:")


# ---- git ------------------------------------------------------------------------------


def test_git_log_lists_commits_and_filters_by_path(toolbox, incident_env):
    out = toolbox.call("git_log", {})
    assert "Simplify date handling" in out and "Initial code" in out
    assert incident_env.suspect in out
    only_contract = toolbox.call("git_log", {"path": "contracts/orders.yaml"})
    assert "Initial code" in only_contract and "Simplify date handling" not in only_contract


def test_git_log_limits_count(toolbox):
    out = toolbox.call("git_log", {"max_count": 1})
    assert out.count("\n") == 0 and "Simplify date handling" in out


def test_git_show_returns_the_diff(toolbox, incident_env):
    out = toolbox.call("git_show", {"commit": incident_env.suspect, "path": "src/demo/transform.py"})
    assert "Simplify date handling" in out
    assert "-    return re.sub" in out and "+    return value" in out


@pytest.mark.parametrize("commit", ["; rm -rf /", "HEAD", "--output=/tmp/x", "abc", ""])
def test_git_show_rejects_anything_but_a_hash(toolbox, commit):
    assert toolbox.call("git_show", {"commit": commit}).startswith("error:")


def test_git_tools_report_errors_instead_of_raising(incident_env, contract, tmp_path):
    not_a_repo = tmp_path / "plain"
    (not_a_repo / "src").mkdir(parents=True)
    box = ToolBox(not_a_repo, incident_env.batch_dir, contract)
    assert box.call("git_log", {}).startswith("error:")


# ---- misc -----------------------------------------------------------------------------


def test_unknown_tool_lists_the_available_ones(toolbox):
    out = toolbox.call("delete_everything", {})
    assert out.startswith("error: unknown tool") and "read_file" in out


def test_clip_marks_truncation():
    text = "x" * (MAX_OUTPUT_CHARS + 50)
    out = clip(text)
    assert len(out) < len(text) and "truncated 50 characters" in out
    assert clip("short") == "short"


def test_specs_describe_every_tool(toolbox):
    names = [s.name for s in toolbox.specs()]
    assert names == ["get_run_summary", "get_violations", "read_file", "git_log", "git_show"]
    assert all(s.parameters["type"] == "object" for s in toolbox.specs())
