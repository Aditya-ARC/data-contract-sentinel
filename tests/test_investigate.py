from __future__ import annotations

import json
import subprocess

import pytest

from contractsentinel.agent.github import GitHubError
from contractsentinel.agent.investigate import (
    DRY_RUN,
    DUPLICATE,
    ISSUE,
    PULL_REQUEST,
    SKIPPED,
    InvestigationSettings,
    investigate_batch,
    investigate_runs,
)
from contractsentinel.agent.schema import SUBMIT_TOOL_NAME
from contractsentinel.cli import EXIT_ERROR, EXIT_OK, main
from contractsentinel.llm.scripted import ScriptedLLMClient, text_response, tool_response
from contractsentinel.pipeline import run_batch
from contractsentinel.sample_data import write_samples

from conftest import FakeGitHub, TRANSFORM_V1, TRANSFORM_V2, REGRESSION_TEST, git

BASE_REPORT = {
    "root_cause": "source_defect",
    "confidence": "high",
    "summary": "The vendor sent typos.",
    "evidence": ["status typos on 7 rows"],
    "impact": "7 rows quarantined",
    "recommended_actions": ["Contact the vendor"],
}


def regression_report(**over):
    report = dict(
        BASE_REPORT,
        root_cause="code_regression",
        summary="A refactor removed slash date handling.",
        suspect_commit="abc1234",
        fix={
            "summary": "Restore slash date normalization and add a regression test.",
            "edits": [
                {"path": "src/demo/transform.py", "old": TRANSFORM_V2, "new": TRANSFORM_V1},
                {"path": "tests/test_regression.py", "old": "", "new": REGRESSION_TEST},
            ],
        },
    )
    report.update(over)
    return report


def settings(env, contract, **kw):
    return InvestigationSettings(repo_root=env.repo, contract=contract, **kw)


def llm_for(report):
    return ScriptedLLMClient([tool_response(SUBMIT_TOOL_NAME, report)], model="scripted-model")


def remote_branches(env):
    return git(env.remote, "branch", "--list").replace("*", "").split()


# ---- issues ---------------------------------------------------------------------------


def test_source_defect_opens_an_issue_with_labels(incident_env, contract):
    gh = FakeGitHub()
    result = investigate_batch(incident_env.batch_dir, settings(incident_env, contract), llm_for(BASE_REPORT), gh)
    assert result.action == ISSUE and result.url.endswith("/issues/1")
    issue = gh.issues[0]
    assert issue["title"] == f"Data contract incident: {incident_env.batch_id} (partially_loaded)"
    assert issue["labels"] == ["data-contract-incident", "root-cause:source-defect"]
    assert "AI-generated" in issue["body"] and "sentinel-incident:" in issue["body"]
    assert "scripted-model" in issue["body"]
    assert gh.prs == []


def test_sensitive_values_never_reach_the_report(incident_env, contract):
    gh = FakeGitHub()
    investigate_batch(incident_env.batch_dir, settings(incident_env, contract), llm_for(BASE_REPORT), gh)
    raw = (incident_env.batch_dir / "violations.json").read_text()
    leaked = [
        s["value"]
        for v in json.loads(raw)["violations"]
        if v["column"] == "customer_id"
        for s in v["samples"]
        if s["value"]
    ]
    for value in leaked:
        assert value not in gh.issues[0]["body"]


def test_no_llm_still_files_a_deterministic_issue(incident_env, contract):
    gh = FakeGitHub()
    result = investigate_batch(incident_env.batch_dir, settings(incident_env, contract), None, gh)
    assert result.action == ISSUE
    assert gh.issues[0]["labels"] == ["data-contract-incident"]
    assert "No LLM is configured" in gh.issues[0]["body"]


def test_llm_failure_falls_back_to_deterministic_issue(incident_env, contract):
    gh = FakeGitHub()
    llm = ScriptedLLMClient([])  # raises on first call
    result = investigate_batch(incident_env.batch_dir, settings(incident_env, contract), llm, gh)
    assert result.action == ISSUE and "LLM call failed" in gh.issues[0]["body"]


def test_inconclusive_investigation_is_reported(incident_env, contract):
    gh = FakeGitHub()
    llm = ScriptedLLMClient([text_response("a"), text_response("b")])
    investigate_batch(incident_env.batch_dir, settings(incident_env, contract), llm, gh)
    assert "did not reach a conclusion" in gh.issues[0]["body"]


def test_duplicate_incident_is_not_filed_again(incident_env, contract):
    gh = FakeGitHub(existing={"html_url": "https://example.test/issues/99"})
    llm = llm_for(BASE_REPORT)
    result = investigate_batch(incident_env.batch_dir, settings(incident_env, contract), llm, gh)
    assert result.action == DUPLICATE and result.url.endswith("/99")
    assert gh.issues == [] and llm.requests == []  # no tokens spent either


def test_clean_batch_is_skipped(incident_env, contract, tmp_path):
    good, _ = write_samples(tmp_path / "landing2")
    res = run_batch(good, contract, None, incident_env.runs, dry_run=True)
    gh = FakeGitHub()
    result = investigate_batch(res.artifacts_dir, settings(incident_env, contract), llm_for(BASE_REPORT), gh)
    assert result.action == SKIPPED and gh.issues == []


def test_dry_run_creates_nothing_and_returns_the_body(incident_env, contract):
    gh = FakeGitHub()
    result = investigate_batch(
        incident_env.batch_dir, settings(incident_env, contract, dry_run=True), llm_for(BASE_REPORT), gh
    )
    assert result.action == DRY_RUN and "# Data contract incident" in result.body
    assert gh.issues == [] and gh.prs == []


# ---- pull requests --------------------------------------------------------------------


def test_regression_fix_that_passes_tests_opens_a_pr(incident_env, contract):
    gh = FakeGitHub()
    st = settings(incident_env, contract, test_runner=lambda root: (True, "ok"))
    result = investigate_batch(incident_env.batch_dir, st, llm_for(regression_report()), gh)

    assert result.action == PULL_REQUEST and result.url.endswith("/pull/1")
    pr = gh.prs[0]
    assert pr["base"] == "main" and pr["head"].startswith("sentinel/fix-")
    assert pr["labels"] == ["data-contract-incident", "needs-human-review"]
    assert "never merges" in pr["body"]
    assert gh.issues == []
    # branch is on the remote, working copy is back on main and clean
    assert pr["head"] in remote_branches(incident_env)
    assert git(incident_env.repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert git(incident_env.repo, "status", "--porcelain") == ""
    assert (incident_env.repo / "src/demo/transform.py").read_text() == TRANSFORM_V2
    # commit author is the bot and the fix is really in the branch
    assert git(incident_env.remote, "log", "-1", "--format=%an", pr["head"]) == "github-actions[bot]"
    assert "re.sub" in git(incident_env.remote, "show", f"{pr['head']}:src/demo/transform.py")


def test_real_test_suite_gates_the_fix(incident_env, contract):
    """Uses the real pytest runner: the regression test only passes with the fix applied."""
    gh = FakeGitHub()
    result = investigate_batch(incident_env.batch_dir, settings(incident_env, contract), llm_for(regression_report()), gh)
    assert result.action == PULL_REQUEST


def test_fix_that_fails_tests_becomes_an_issue_and_leaves_no_branch(incident_env, contract):
    gh = FakeGitHub()
    st = settings(incident_env, contract, test_runner=lambda root: (False, "1 failed"))
    result = investigate_batch(incident_env.batch_dir, st, llm_for(regression_report()), gh)
    assert result.action == ISSUE and gh.prs == []
    assert "failed the test suite" in gh.issues[0]["body"]
    assert git(incident_env.repo, "branch", "--list") .replace("*", "").split() == ["main"]
    assert remote_branches(incident_env) == ["main"]
    assert git(incident_env.repo, "status", "--porcelain") == ""
    assert (incident_env.repo / "src/demo/transform.py").read_text() == TRANSFORM_V2


def test_invalid_fix_is_rejected_and_never_applied(incident_env, contract):
    bad = regression_report()
    bad["fix"]["edits"] = [{"path": ".github/workflows/ci.yml", "old": "a", "new": "b"}]
    gh = FakeGitHub()
    called = []
    st = settings(incident_env, contract, test_runner=lambda root: called.append(1) or (True, ""))
    result = investigate_batch(incident_env.batch_dir, st, llm_for(bad), gh)
    assert result.action == ISSUE and called == []
    assert "fix was rejected" in gh.issues[0]["body"]


def test_contract_change_with_wrong_root_cause_is_rejected(incident_env, contract):
    report = regression_report(root_cause="source_defect")
    report["fix"]["edits"] = [
        {"path": "contracts/orders.yaml", "old": "allowed: [NA, EMEA, APAC, LATAM]", "new": "allowed: [NA]"}
    ]
    gh = FakeGitHub()
    result = investigate_batch(incident_env.batch_dir, settings(incident_env, contract), llm_for(report), gh)
    assert result.action == ISSUE and "contract_out_of_date" in gh.issues[0]["body"]


def test_contract_fix_gets_the_contract_change_label(incident_env, contract):
    report = regression_report(root_cause="contract_out_of_date")
    report["fix"]["edits"] = [
        {"path": "contracts/orders.yaml", "old": "allowed: [NA, EMEA, APAC, LATAM]", "new": "allowed: [NA, EMEA, APAC, LATAM, MEA]"}
    ]
    gh = FakeGitHub()
    st = settings(incident_env, contract, test_runner=lambda root: (True, "ok"))
    result = investigate_batch(incident_env.batch_dir, st, llm_for(report), gh)
    assert result.action == PULL_REQUEST and "contract-change" in gh.prs[0]["labels"]


def test_pr_creation_failure_falls_back_to_issue_and_mentions_the_setting(incident_env, contract):
    gh = FakeGitHub(pr_error=GitHubError(403, "GitHub Actions is not permitted to create or approve pull requests"))
    st = settings(incident_env, contract, test_runner=lambda root: (True, "ok"))
    result = investigate_batch(incident_env.batch_dir, st, llm_for(regression_report()), gh)
    assert result.action == ISSUE
    assert "Allow GitHub Actions to create and approve pull requests" in gh.issues[0]["body"]


def test_dirty_working_tree_blocks_the_fix(incident_env, contract):
    (incident_env.repo / "notes.txt").write_text("local\n")
    gh = FakeGitHub()
    st = settings(incident_env, contract, test_runner=lambda root: (True, "ok"))
    result = investigate_batch(incident_env.batch_dir, st, llm_for(regression_report()), gh)
    assert result.action == ISSUE and "uncommitted changes" in gh.issues[0]["body"]


def test_dry_run_validates_but_does_not_apply_the_fix(incident_env, contract):
    gh = FakeGitHub()
    result = investigate_batch(
        incident_env.batch_dir, settings(incident_env, contract, dry_run=True), llm_for(regression_report()), gh
    )
    assert result.action == DRY_RUN and any("would be tested" in n for n in result.notes)
    assert git(incident_env.repo, "branch", "--list").replace("*", "").split() == ["main"]


# ---- many batches and the CLI ---------------------------------------------------------


def test_investigate_runs_filters_by_batch(incident_env, contract, tmp_path):
    good, _ = write_samples(tmp_path / "l3")
    run_batch(good, contract, None, incident_env.runs, dry_run=True)
    gh = FakeGitHub()
    results = investigate_runs(incident_env.runs, settings(incident_env, contract), None, gh)
    assert sorted(r.action for r in results) == [ISSUE, SKIPPED]
    only = investigate_runs(incident_env.runs, settings(incident_env, contract), None, FakeGitHub(), only_batch=incident_env.batch_id)
    assert [r.batch_id for r in only] == [incident_env.batch_id]


def cli_args(env, *extra):
    return ["investigate", "--runs", str(env.runs), "--repo-root", str(env.repo), "--contract", str(env.repo / "contracts/orders.yaml"), *extra]


def test_cli_dry_run_without_llm(incident_env, capsys, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    code = main(cli_args(incident_env, "--dry-run", "--no-llm"))
    out = capsys.readouterr().out
    assert code == EXIT_OK and "dry_run" in out and "What broke" in out


def test_cli_requires_github_credentials_for_live_runs(incident_env, capsys, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert main(cli_args(incident_env, "--no-llm")) == EXIT_ERROR
    assert "GITHUB_TOKEN" in capsys.readouterr().err


def test_cli_unconfigured_llm_warns_and_continues(incident_env, capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = main(cli_args(incident_env, "--dry-run"))
    cap = capsys.readouterr()
    assert code == EXIT_OK and "warning" in cap.err and "deterministic" in cap.err


def test_cli_with_no_runs(tmp_path, incident_env, capsys):
    args = ["investigate", "--runs", str(tmp_path / "empty"), "--repo-root", str(incident_env.repo), "--dry-run", "--no-llm", "--contract", str(incident_env.repo / "contracts/orders.yaml")]
    assert main(args) == EXIT_OK
    assert "No run artifacts" in capsys.readouterr().out
