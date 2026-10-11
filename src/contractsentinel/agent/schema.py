"""The structured result the agent must return, and its validation.

The model finishes by calling ``submit_incident_report``. Arguments are checked
here and any problem is sent back to the model so it can correct itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contractsentinel.llm.base import ToolSpec

SUBMIT_TOOL_NAME = "submit_incident_report"

ROOT_CAUSES = (
    "source_defect",
    "code_regression",
    "contract_out_of_date",
    "schema_drift",
    "unknown",
)
CONFIDENCE_LEVELS = ("low", "medium", "high")

MAX_SUMMARY_CHARS = 1500
MAX_ITEM_CHARS = 400
MAX_ITEMS = 8
MAX_EDITS = 5
MAX_EDIT_TEXT_CHARS = 6000


class AnalysisError(ValueError):
    """The submitted report is invalid. The message goes back to the model."""


@dataclass(frozen=True)
class Edit:
    path: str
    old: str  # exact text to replace; empty means create a new file (tests/ only)
    new: str


@dataclass(frozen=True)
class ProposedFix:
    summary: str
    edits: tuple[Edit, ...]


@dataclass(frozen=True)
class Analysis:
    root_cause: str
    confidence: str
    summary: str
    evidence: tuple[str, ...]
    impact: str
    recommended_actions: tuple[str, ...]
    suspect_commit: str | None = None
    fix: ProposedFix | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def submit_spec() -> ToolSpec:
    return ToolSpec(
        name=SUBMIT_TOOL_NAME,
        description=(
            "Finish the investigation. Call exactly once with your findings. "
            "Include `fix` only when you are confident in an exact, minimal code or contract change."
        ),
        parameters={
            "type": "object",
            "properties": {
                "root_cause": {"type": "string", "enum": list(ROOT_CAUSES)},
                "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
                "summary": {"type": "string", "description": "What broke and why, in 2 to 4 sentences."},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Facts you verified with tools, such as a commit or a code line.",
                },
                "impact": {"type": "string", "description": "Effect on downstream data."},
                "recommended_actions": {"type": "array", "items": {"type": "string"}},
                "suspect_commit": {"type": "string", "description": "Short hash, if one caused this."},
                "fix": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "old": {
                                        "type": "string",
                                        "description": "Exact text to replace; it must appear once. "
                                        "Empty string creates a new file under tests/.",
                                    },
                                    "new": {"type": "string"},
                                },
                                "required": ["path", "old", "new"],
                            },
                        },
                    },
                    "required": ["summary", "edits"],
                },
            },
            "required": ["root_cause", "confidence", "summary", "evidence", "impact", "recommended_actions"],
        },
    )


def _text(args: dict[str, Any], key: str, limit: int, required: bool = True) -> str:
    value = args.get(key)
    if value is None or value == "":
        if required:
            raise AnalysisError(f"'{key}' is required")
        return ""
    if not isinstance(value, str):
        raise AnalysisError(f"'{key}' must be a string")
    if len(value) > limit:
        raise AnalysisError(f"'{key}' is too long ({len(value)} characters, limit {limit})")
    return value.strip()


def _text_list(args: dict[str, Any], key: str) -> tuple[str, ...]:
    value = args.get(key)
    if not isinstance(value, list) or not value:
        raise AnalysisError(f"'{key}' must be a non-empty list of strings")
    if len(value) > MAX_ITEMS:
        raise AnalysisError(f"'{key}' has too many items (limit {MAX_ITEMS})")
    items = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise AnalysisError(f"'{key}' must contain only non-empty strings")
        if len(item) > MAX_ITEM_CHARS:
            raise AnalysisError(f"an item in '{key}' is too long (limit {MAX_ITEM_CHARS} characters)")
        items.append(item.strip())
    return tuple(items)


def _parse_fix(raw: Any) -> ProposedFix:
    if not isinstance(raw, dict):
        raise AnalysisError("'fix' must be an object")
    summary = _text(raw, "summary", MAX_SUMMARY_CHARS)
    edits_raw = raw.get("edits")
    if not isinstance(edits_raw, list) or not edits_raw:
        raise AnalysisError("'fix.edits' must be a non-empty list")
    if len(edits_raw) > MAX_EDITS:
        raise AnalysisError(f"'fix.edits' has too many edits (limit {MAX_EDITS})")
    edits = []
    for i, item in enumerate(edits_raw):
        if not isinstance(item, dict):
            raise AnalysisError(f"fix.edits[{i}] must be an object")
        path, old, new = item.get("path"), item.get("old"), item.get("new")
        if not isinstance(path, str) or not path:
            raise AnalysisError(f"fix.edits[{i}].path must be a non-empty string")
        if not isinstance(old, str) or not isinstance(new, str):
            raise AnalysisError(f"fix.edits[{i}].old and .new must be strings")
        if len(old) > MAX_EDIT_TEXT_CHARS or len(new) > MAX_EDIT_TEXT_CHARS:
            raise AnalysisError(f"fix.edits[{i}] text is too long (limit {MAX_EDIT_TEXT_CHARS})")
        edits.append(Edit(path=path, old=old, new=new))
    return ProposedFix(summary=summary, edits=tuple(edits))


def parse_analysis(args: dict[str, Any]) -> Analysis:
    root_cause = args.get("root_cause")
    if root_cause not in ROOT_CAUSES:
        raise AnalysisError(f"'root_cause' must be one of {list(ROOT_CAUSES)}")
    confidence = args.get("confidence")
    if confidence not in CONFIDENCE_LEVELS:
        raise AnalysisError(f"'confidence' must be one of {list(CONFIDENCE_LEVELS)}")

    suspect = _text(args, "suspect_commit", 40, required=False) or None
    fix = _parse_fix(args["fix"]) if args.get("fix") else None

    return Analysis(
        root_cause=root_cause,
        confidence=confidence,
        summary=_text(args, "summary", MAX_SUMMARY_CHARS),
        evidence=_text_list(args, "evidence"),
        impact=_text(args, "impact", MAX_ITEM_CHARS * 2),
        recommended_actions=_text_list(args, "recommended_actions"),
        suspect_commit=suspect,
        fix=fix,
    )
