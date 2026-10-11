"""Orchestrates one incident: investigate, validate any fix, then open a PR or an Issue.

Order of preference:
  1. Pull request, when the agent proposed a fix that passed validation and the
     full test suite on a throwaway branch.
  2. Issue, in every other case (source defects, no fix, rejected fix, failing
     tests, no LLM, LLM errors).

The agent never merges. A human reviews every PR and every Issue.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from contractsentinel.agent.fixes import FixError, apply_fix, run_tests, touches_contract, validate_fix
from contractsentinel.agent.github import GitHubError
from contractsentinel.agent.loop import AgentConfig, investigate
from contractsentinel.agent.mask import mask_violations
from contractsentinel.agent.report import (
    INCIDENT_LABEL,
    ReportMeta,
    fingerprint,
    incident_marker,
    incident_title,
    render_body,
)
from contractsentinel.agent.schema import Analysis
from contractsentinel.agent.tools import ToolBox
from contractsentinel.contract import Contract
from contractsentinel.llm.base import LLMClient

BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"
GIT_TIMEOUT_SECONDS = 120

SKIPPED = "skipped"
DUPLICATE = "duplicate"
ISSUE = "issue"
PULL_REQUEST = "pull_request"
DRY_RUN = "dry_run"


@dataclass
class InvestigationSettings:
    repo_root: Path
    contract: Contract
    dry_run: bool = False
    run_url: str | None = None
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    test_runner: Callable[[Path], tuple[bool, str]] = run_tests


@dataclass
class IncidentResult:
    batch_id: str
    action: str
    title: str = ""
    body: str = ""
    url: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class FixAttempt:
    ok: bool
    notes: list[str]
    branch: str | None = None
    changed: list[str] = field(default_factory=list)


class GitFailure(Exception):
    pass


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        env=env,
    )
    if check and result.returncode != 0:
        raise GitFailure(f"git {args[0]} failed: {result.stderr.strip()[:300]}")
    return result


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:60]


def build_brief(summary: dict[str, Any], violations_doc: dict[str, Any]) -> str:
    counts = summary["counts"]
    decision = summary["decision"]
    lines = [
        f"Batch: {summary['batch_id']}",
        f"Status: {summary['status']} ({decision['action']}: {decision['reason']})",
        f"Rows: {counts['rows_total']} total, {counts['rows_loaded']} loaded, "
        f"{counts['rows_quarantined']} quarantined",
        f"Contract: {summary['contract']['name']} {summary['contract']['version']}",
        "Failed rules:",
    ]
    for v in violations_doc.get("violations", []):
        lines.append(f"- {v['rule_id']} ({v['severity']}) x{v['count']}: {v['description']}")
    lines.append("Investigate, then call submit_incident_report.")
    return "\n".join(lines)


def _attempt_fix(
    settings: InvestigationSettings, analysis: Analysis, batch_id: str, fp: str
) -> FixAttempt:
    """Apply the fix on a throwaway branch, run the tests, and push only if they pass."""
    root = settings.repo_root
    fix = analysis.fix
    assert fix is not None

    if _git(root, "status", "--porcelain").stdout.strip():
        return FixAttempt(False, ["Fix not attempted: the working tree has uncommitted changes."])

    head = _git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    original = head if head != "HEAD" else _git(root, "rev-parse", "HEAD").stdout.strip()
    branch = f"sentinel/fix-{_slug(batch_id)}-{fp[:7]}"

    try:
        _git(root, "checkout", "-q", "-b", branch)
    except GitFailure as exc:
        return FixAttempt(False, [f"Fix not attempted: could not create branch ({exc})."])

    pushed = False
    try:
        changed = apply_fix(fix, root)
        passed, output = settings.test_runner(root)
        if not passed:
            return FixAttempt(
                False,
                [
                    "The proposed fix failed the test suite and was discarded.",
                    f"Test output (tail): `{output[-300:].replace('`', chr(39))}`",
                ],
            )
        _git(root, "add", "--", *changed)
        _git(
            root,
            "-c", f"user.name={BOT_NAME}",
            "-c", f"user.email={BOT_EMAIL}",
            "commit", "-q",
            "-m", f"Fix data contract incident {batch_id}",
            "-m", fix.summary,
        )
        _git(root, "push", "-q", "origin", f"{branch}:{branch}")
        pushed = True
        return FixAttempt(True, [], branch=branch, changed=changed)
    except (FixError, GitFailure) as exc:
        return FixAttempt(False, [f"Fix not applied: {exc}"])
    finally:
        _git(root, "checkout", "-q", "-f", original, check=False)
        _git(root, "clean", "-fdq", check=False)
        if not pushed:
            _git(root, "branch", "-D", branch, check=False)


def investigate_batch(
    batch_dir: str | Path,
    settings: InvestigationSettings,
    llm: LLMClient | None,
    gh: Any | None,
) -> IncidentResult:
    batch_dir = Path(batch_dir)
    summary = json.loads((batch_dir / "summary.json").read_text(encoding="utf-8"))
    batch_id = summary["batch_id"]

    if not summary.get("needs_agent"):
        return IncidentResult(batch_id, SKIPPED, notes=["Batch passed the contract."])

    violations = json.loads((batch_dir / "violations.json").read_text(encoding="utf-8"))
    masked = mask_violations(violations, settings.contract)
    fp = fingerprint(masked)
    marker = incident_marker(batch_id, fp)

    live = gh is not None and not settings.dry_run
    if live:
        existing = gh.find_open_incident(marker, INCIDENT_LABEL)
        if existing:
            return IncidentResult(
                batch_id,
                DUPLICATE,
                url=existing.get("html_url"),
                notes=["An open incident already exists for this batch and rule set."],
            )

    meta = ReportMeta(llm_configured=llm is not None)
    analysis: Analysis | None = None
    if llm is not None:
        toolbox = ToolBox(settings.repo_root, batch_dir, settings.contract)
        outcome = investigate(llm, toolbox, build_brief(summary, masked), settings.agent_config)
        analysis = outcome.analysis
        meta.model = outcome.model
        meta.stop_reason = outcome.stop_reason
        meta.steps = outcome.steps
        meta.input_tokens = outcome.usage.input_tokens
        meta.output_tokens = outcome.usage.output_tokens
        meta.llm_error = outcome.error

    notes: list[str] = []
    attempt: FixAttempt | None = None
    if analysis is not None and analysis.fix is not None:
        problems = validate_fix(analysis.fix, analysis.root_cause, settings.repo_root)
        if problems:
            notes.append("The proposed fix was rejected: " + "; ".join(problems) + ".")
        elif settings.dry_run:
            notes.append("Dry run: the fix passed validation and would be tested, then opened as a PR.")
        elif gh is None:
            notes.append("No GitHub client configured, so the fix was not attempted.")
        else:
            attempt = _attempt_fix(settings, analysis, batch_id, fp)
            notes.extend(attempt.notes)

    title = incident_title(summary)

    if settings.dry_run:
        body = render_body(summary, masked, analysis, meta, marker, settings.run_url, notes)
        return IncidentResult(batch_id, DRY_RUN, title, body, notes=notes)

    assert gh is not None

    if attempt is not None and attempt.ok and analysis is not None and analysis.fix is not None:
        pr_notes = notes + [
            "Tests passed in the agent job before this PR was opened.",
            "A human must review before merging. The agent never merges.",
            "CI does not start automatically for PRs opened with the workflow token. "
            "Push an empty commit or close and reopen this PR to run it.",
        ]
        labels = [INCIDENT_LABEL, "needs-human-review"]
        if touches_contract(analysis.fix):
            labels.append("contract-change")
        body = render_body(summary, masked, analysis, meta, marker, settings.run_url, pr_notes)
        try:
            base = gh.default_branch()
            pr = gh.create_pull_request(
                f"Fix for data contract incident {batch_id}", body, attempt.branch, base, labels
            )
            return IncidentResult(batch_id, PULL_REQUEST, title, body, pr.get("html_url"), pr_notes)
        except GitHubError as exc:
            notes.append(
                f"The fix passed tests and was pushed to branch `{attempt.branch}`, but opening a PR "
                f"failed ({exc.message}). If this is a 403, enable Settings > Actions > General > "
                "Allow GitHub Actions to create and approve pull requests."
            )

    labels = [INCIDENT_LABEL]
    if analysis is not None:
        labels.append("root-cause:" + analysis.root_cause.replace("_", "-"))
    body = render_body(summary, masked, analysis, meta, marker, settings.run_url, notes)
    issue = gh.create_issue(title, body, labels)
    return IncidentResult(batch_id, ISSUE, title, body, issue.get("html_url"), notes)


def investigate_runs(
    runs_dir: str | Path,
    settings: InvestigationSettings,
    llm: LLMClient | None,
    gh: Any | None,
    only_batch: str | None = None,
) -> list[IncidentResult]:
    results = []
    for summary_path in sorted(Path(runs_dir).glob("*/summary.json")):
        batch_dir = summary_path.parent
        if only_batch and batch_dir.name != only_batch:
            continue
        results.append(investigate_batch(batch_dir, settings, llm, gh))
    return results
