"""Read-only tools the investigation agent may call.

Everything here is sandboxed: paths must stay inside the repository and inside
an allowlist, git is invoked without a shell, outputs are size-capped, and
sensitive sample values are masked before they are returned.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from contractsentinel.agent.mask import mask_violations
from contractsentinel.contract import Contract
from contractsentinel.llm.base import ToolSpec

READABLE_PREFIXES = ("src/", "tests/", "contracts/", "docs/", ".github/")
READABLE_FILES = ("README.md", "pyproject.toml")
MAX_READ_LINES = 120
MAX_OUTPUT_CHARS = 3500
MAX_DIFF_CHARS = 6000
MAX_VIOLATIONS_CHARS = 6000
GIT_TIMEOUT_SECONDS = 20
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


class ToolError(Exception):
    """A tool call that cannot be completed. The message is shown to the model."""


def clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} characters]"


def _int_arg(arguments: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"'{key}' must be an integer")
    return max(low, min(high, value))


class ToolBox:
    def __init__(self, repo_root: str | Path, batch_dir: str | Path, contract: Contract) -> None:
        self.root = Path(repo_root).resolve()
        self.batch_dir = Path(batch_dir)
        self.contract = contract

    # ---- specs ----------------------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        no_args = {"type": "object", "properties": {}}
        return [
            ToolSpec(
                "get_run_summary",
                "Pipeline decision, row counts and contract version for this batch.",
                no_args,
            ),
            ToolSpec(
                "get_violations",
                "Every failed contract rule with counts and a few sample rows. "
                "Sensitive columns are masked.",
                no_args,
            ),
            ToolSpec(
                "read_file",
                "Read a repository file (src/, tests/, contracts/, docs/, .github/, README.md, "
                "pyproject.toml). Returns numbered lines, at most 120 per call.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "description": "1-based, default 1"},
                        "end_line": {"type": "integer"},
                    },
                    "required": ["path"],
                },
            ),
            ToolSpec(
                "git_log",
                "Recent commits (hash, date, author, subject), optionally for one path.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_count": {"type": "integer", "description": "1 to 20, default 10"},
                    },
                },
            ),
            ToolSpec(
                "git_show",
                "The diff for one commit, optionally limited to one path.",
                {
                    "type": "object",
                    "properties": {"commit": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["commit"],
                },
            ),
        ]

    # ---- dispatch -------------------------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool and return text for the model. Errors are returned, not raised."""
        handlers = {
            "get_run_summary": self._get_run_summary,
            "get_violations": self._get_violations,
            "read_file": self._read_file,
            "git_log": self._git_log,
            "git_show": self._git_show,
        }
        handler = handlers.get(name)
        if handler is None:
            return f"error: unknown tool '{name}'. Available: {', '.join(handlers)}"
        try:
            return handler(arguments)
        except ToolError as exc:
            return f"error: {exc}"

    # ---- helpers --------------------------------------------------------------------

    def _safe_path(self, raw: Any) -> Path:
        if not isinstance(raw, str) or not raw:
            raise ToolError("'path' must be a non-empty string")
        rel = PurePosixPath(raw)
        if rel.is_absolute() or ".." in rel.parts:
            raise ToolError("path must be relative and stay inside the repository")
        text = rel.as_posix()
        if text not in READABLE_FILES and not text.startswith(READABLE_PREFIXES):
            allowed = ", ".join(READABLE_PREFIXES + READABLE_FILES)
            raise ToolError(f"path not readable. Allowed: {allowed}")
        full = (self.root / rel).resolve()
        if not full.is_relative_to(self.root):
            raise ToolError("path escapes the repository")
        return full

    def _git(self, *args: str) -> str:
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=GIT_TIMEOUT_SECONDS,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ToolError(f"git failed: {exc}") from exc
        if result.returncode != 0:
            raise ToolError(f"git failed: {clip(result.stderr.strip(), 400)}")
        return result.stdout

    # ---- tools ----------------------------------------------------------------------

    def _get_run_summary(self, _: dict[str, Any]) -> str:
        path = self.batch_dir / "summary.json"
        if not path.is_file():
            raise ToolError("summary.json not found for this batch")
        data = json.loads(path.read_text(encoding="utf-8"))
        keep = ("batch_id", "status", "decision", "counts", "contract", "destination", "needs_agent")
        return json.dumps({k: data[k] for k in keep if k in data}, separators=(",", ":"))

    def _get_violations(self, _: dict[str, Any]) -> str:
        path = self.batch_dir / "violations.json"
        if not path.is_file():
            raise ToolError("violations.json not found for this batch")
        doc = mask_violations(json.loads(path.read_text(encoding="utf-8")), self.contract)
        return clip(json.dumps(doc, separators=(",", ":")), MAX_VIOLATIONS_CHARS)

    def _read_file(self, arguments: dict[str, Any]) -> str:
        full = self._safe_path(arguments.get("path"))
        if not full.is_file():
            raise ToolError(f"file not found: {arguments['path']}")
        lines = full.read_text(encoding="utf-8", errors="replace").splitlines()
        start = _int_arg(arguments, "start_line", 1, 1, max(1, len(lines)))
        end_default = start + MAX_READ_LINES - 1
        end = _int_arg(arguments, "end_line", end_default, start, start + MAX_READ_LINES - 1)
        end = min(end, len(lines))
        body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        header = f"{arguments['path']} lines {start}-{end} of {len(lines)}"
        return clip(f"{header}\n{body}")

    def _git_log(self, arguments: dict[str, Any]) -> str:
        count = _int_arg(arguments, "max_count", 10, 1, 20)
        args = ["log", f"--max-count={count}", "--date=short", "--format=%h %ad %an: %s"]
        if arguments.get("path"):
            rel = self._safe_path(arguments["path"]).relative_to(self.root).as_posix()
            args += ["--", rel]
        out = self._git(*args).strip() or "(no commits found)"
        shallow = self._git("rev-parse", "--is-shallow-repository").strip() == "true"
        if shallow:
            out += "\n(note: shallow clone, older history is not available)"
        return clip(out)

    def _git_show(self, arguments: dict[str, Any]) -> str:
        commit = arguments.get("commit")
        if not isinstance(commit, str) or not _COMMIT_RE.match(commit):
            raise ToolError("'commit' must be a 7 to 40 character hex hash")
        args = ["show", "--no-color", "--stat", "--patch", "--format=%h %an %ad%n%s%n%b", commit]
        if arguments.get("path"):
            rel = self._safe_path(arguments["path"]).relative_to(self.root).as_posix()
            args += ["--", rel]
        return clip(self._git(*args), MAX_DIFF_CHARS)
