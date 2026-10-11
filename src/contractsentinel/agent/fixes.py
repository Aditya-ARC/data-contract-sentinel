"""Validate, apply and test a fix proposed by the agent.

The agent proposes exact search-and-replace edits. Nothing is applied unless it
passes these checks, and nothing is published unless the test suite passes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path, PurePosixPath

from contractsentinel.agent.schema import ProposedFix

ALLOWED_FIX_PREFIXES = ("src/", "tests/", "contracts/")
MAX_FIX_CHARS = 20000
TEST_TIMEOUT_SECONDS = 600


class FixError(Exception):
    """A fix that cannot be applied."""


def touches_contract(fix: ProposedFix) -> bool:
    return any(e.path.startswith("contracts/") for e in fix.edits)


def validate_fix(fix: ProposedFix, root_cause: str, repo_root: str | Path) -> list[str]:
    """Return a list of problems. An empty list means the fix may be attempted."""
    root = Path(repo_root).resolve()
    problems: list[str] = []
    total = 0

    for edit in fix.edits:
        rel = PurePosixPath(edit.path)
        if rel.is_absolute() or ".." in rel.parts or not edit.path.startswith(ALLOWED_FIX_PREFIXES):
            problems.append(f"{edit.path}: outside the allowed paths {list(ALLOWED_FIX_PREFIXES)}")
            continue
        full = (root / rel).resolve()
        if not full.is_relative_to(root):
            problems.append(f"{edit.path}: escapes the repository")
            continue

        total += len(edit.old) + len(edit.new)

        if edit.path.startswith("contracts/") and root_cause != "contract_out_of_date":
            problems.append(
                f"{edit.path}: contract changes are only allowed when root_cause is contract_out_of_date"
            )

        if edit.old == "":
            if not edit.path.startswith("tests/"):
                problems.append(f"{edit.path}: only new files under tests/ may be created")
            elif full.exists():
                problems.append(f"{edit.path}: already exists, so 'old' text is required")
        elif not full.is_file():
            problems.append(f"{edit.path}: file does not exist")
        else:
            count = full.read_text(encoding="utf-8").count(edit.old)
            if count != 1:
                problems.append(f"{edit.path}: 'old' text must match exactly once, found {count}")

    if total > MAX_FIX_CHARS:
        problems.append(f"fix is too large ({total} characters, limit {MAX_FIX_CHARS})")
    return problems


def apply_fix(fix: ProposedFix, repo_root: str | Path) -> list[str]:
    """Apply edits in order. Returns the changed paths. Call validate_fix first."""
    root = Path(repo_root).resolve()
    changed: list[str] = []
    for edit in fix.edits:
        full = root / edit.path
        if edit.old == "":
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(edit.new, encoding="utf-8")
        else:
            text = full.read_text(encoding="utf-8")
            if text.count(edit.old) != 1:
                raise FixError(f"{edit.path}: 'old' text no longer matches exactly once")
            full.write_text(text.replace(edit.old, edit.new, 1), encoding="utf-8")
        if edit.path not in changed:
            changed.append(edit.path)
    return changed


def run_tests(repo_root: str | Path) -> tuple[bool, str]:
    """Run the project's test suite. Returns (passed, tail of the output)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
            cwd=Path(repo_root),
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, f"tests timed out after {TEST_TIMEOUT_SECONDS} seconds"
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output[-1500:]
