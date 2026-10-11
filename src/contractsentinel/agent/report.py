"""Render the incident report.

Facts (counts, rules, file hash) come from the pipeline's own artifacts. Only
the Analysis section comes from the model, and it is labeled as such.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from contractsentinel.agent.schema import Analysis

INCIDENT_LABEL = "data-contract-incident"
MAX_SAMPLES_SHOWN = 3


@dataclass
class ReportMeta:
    llm_configured: bool = False
    model: str | None = None
    stop_reason: str | None = None
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_error: str | None = None


def fingerprint(violations_doc: dict[str, Any]) -> str:
    """Stable id for 'this batch failed these rules', used to avoid duplicate incidents."""
    rules = sorted(
        v["rule_id"] for v in violations_doc.get("violations", []) if v.get("severity") == "error"
    )
    return hashlib.sha1("|".join(rules).encode("utf-8")).hexdigest()[:12]


def incident_marker(batch_id: str, fp: str) -> str:
    return f"<!-- sentinel-incident:{batch_id}:{fp} -->"


def incident_title(summary: dict[str, Any]) -> str:
    return f"Data contract incident: {summary['batch_id']} ({summary['status']})"


def _code(text: Any, limit: int = 80) -> str:
    value = str(text).replace("`", "'").replace("\n", " ").replace("|", "/")
    if len(value) > limit:
        value = value[: limit - 3] + "..."
    return f"`{value}`" if value else "(empty)"


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _bullets(items) -> list[str]:
    return [f"- {_cell(item)}" for item in items]


def _analysis_section(analysis: Analysis | None, meta: ReportMeta) -> list[str]:
    lines = ["## Analysis"]
    if analysis is None:
        if not meta.llm_configured:
            reason = "No LLM is configured, so this is the deterministic report only."
        elif meta.llm_error:
            reason = f"The LLM call failed ({_cell(meta.llm_error)}), so this is the deterministic report only."
        else:
            reason = (
                "The investigation did not reach a conclusion "
                f"(stopped: {meta.stop_reason}), so this is the deterministic report only."
            )
        return lines + [reason, ""]

    lines += [
        "_AI-generated. Verify before acting._",
        "",
        f"**Root cause:** `{analysis.root_cause}` (confidence: {analysis.confidence})",
        "",
        analysis.summary,
        "",
        "**Evidence**",
        *_bullets(analysis.evidence),
        "",
        f"**Impact:** {analysis.impact}",
        "",
        "**Recommended actions**",
        *_bullets(analysis.recommended_actions),
        "",
    ]
    if analysis.suspect_commit:
        lines += [f"**Suspect commit:** `{analysis.suspect_commit}`", ""]
    return lines


def render_body(
    summary: dict[str, Any],
    violations_doc: dict[str, Any],
    analysis: Analysis | None,
    meta: ReportMeta,
    marker: str,
    run_url: str | None = None,
    fix_notes: list[str] | None = None,
) -> str:
    counts = summary["counts"]
    decision = summary["decision"]
    sha = summary.get("source_sha256", "")[:12]

    lines = [
        f"# Data contract incident: {_cell(summary['batch_id'])}",
        "",
        "| | |",
        "|---|---|",
        f"| Status | {summary['status']} |",
        f"| Pipeline decision | {decision['action']}: {_cell(decision['reason'])} |",
        f"| Rows | {counts['rows_total']} total, {counts['rows_loaded']} loaded, "
        f"{counts['rows_quarantined']} quarantined |",
        f"| Contract | {summary['contract']['name']} {summary['contract']['version']} |",
        f"| Source file | {_code(summary.get('source_file', ''))} (sha256 `{sha}`) |",
    ]
    if run_url:
        lines.append(f"| Workflow run | {run_url} |")

    lines += [
        "",
        "## What broke (from the pipeline)",
        "",
        "| Rule | Severity | Rows | Description |",
        "|---|---|---|---|",
    ]
    for v in violations_doc.get("violations", []):
        lines.append(
            f"| `{v['rule_id']}` | {v['severity']} | {v['count']} | {_cell(v['description'])} |"
        )

    sample_lines = []
    for v in violations_doc.get("violations", []):
        for s in v.get("samples", [])[:MAX_SAMPLES_SHOWN]:
            shown = _code(s["value"])
            if "raw_value" in s:
                shown += f" (raw: {_code(s['raw_value'])})"
            sample_lines.append(f"- `{v['rule_id']}` row {s['row_number']}: {shown}")
    if sample_lines:
        lines += ["", "Sample values (sensitive columns are masked):", "", *sample_lines]

    lines += [""] + _analysis_section(analysis, meta)

    if analysis and analysis.fix:
        lines += ["## Proposed fix", "", analysis.fix.summary, ""]
    if fix_notes:
        lines += ["## Fix status", "", *_bullets(fix_notes), ""]

    footer = "Counts, rules and the file hash come from the pipeline. Only the Analysis section is model-written."
    if meta.model:
        footer += (
            f" Model: `{meta.model}`, {meta.steps} step(s), "
            f"{meta.input_tokens} input and {meta.output_tokens} output tokens."
        )
    lines += ["---", f"_{footer}_", "", marker]
    return "\n".join(lines) + "\n"
