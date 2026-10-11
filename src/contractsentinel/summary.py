"""Render run artifacts as Markdown (used for the GitHub Actions job summary)."""

from __future__ import annotations

import json
from pathlib import Path


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_summary(runs_dir: str | Path) -> str:
    runs = Path(runs_dir)
    summaries = sorted(runs.glob("*/summary.json")) if runs.is_dir() else []
    if not summaries:
        return "No pipeline runs found.\n"

    lines = [
        "## Pipeline run summary",
        "",
        "| Batch | Status | Rows | Loaded | Quarantined | Agent needed |",
        "|---|---|---|---|---|---|",
    ]
    details: list[str] = []

    for path in summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        counts = summary["counts"]
        batch = _cell(summary["batch_id"])
        lines.append(
            f"| {batch} | {summary['status']} | {counts['rows_total']} | {counts['rows_loaded']} "
            f"| {counts['rows_quarantined']} | {'yes' if summary['needs_agent'] else 'no'} |"
        )

        violations_path = path.with_name("violations.json")
        if violations_path.exists():
            violations = json.loads(violations_path.read_text(encoding="utf-8"))["violations"]
            if violations:
                details.append(f"### {batch}")
                details.append("")
                for v in violations:
                    details.append(
                        f"- `{v['rule_id']}` ({v['severity']}) x{v['count']}: {_cell(v['description'])}"
                    )
                details.append("")

    return "\n".join(lines + [""] + details) + "\n"
