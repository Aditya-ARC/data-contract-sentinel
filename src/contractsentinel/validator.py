"""Deterministic contract validation.

No AI is involved here. Given a batch (all values as strings) and a contract,
the validator returns structured violations and a per-row failure map.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from contractsentinel.contract import Contract, FieldSpec

MAX_SAMPLES = 5
MAX_VALUE_CHARS = 120

ERROR = "error"
WARNING = "warning"


@dataclass
class Violation:
    rule_id: str
    column: str | None
    rule: str
    description: str
    severity: str
    scope: str  # "row" or "batch"
    count: int
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    batch_id: str
    contract_name: str
    contract_version: str
    total_rows: int
    violations: list[Violation]
    # DataFrame index -> rule ids that failed on that row (row-level errors only).
    row_failures: dict[int, list[str]]

    @property
    def error_violations(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == ERROR]

    @property
    def warning_violations(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == WARNING]

    @property
    def has_batch_level_error(self) -> bool:
        return any(v.scope == "batch" for v in self.error_violations)

    @property
    def invalid_row_count(self) -> int:
        return len(self.row_failures)

    @property
    def invalid_rate(self) -> float:
        return self.invalid_row_count / self.total_rows if self.total_rows else 0.0

    @property
    def passed(self) -> bool:
        return not self.error_violations

    def invalid_index(self) -> list[int]:
        return sorted(self.row_failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "contract": {"name": self.contract_name, "version": self.contract_version},
            "total_rows": self.total_rows,
            "invalid_rows": self.invalid_row_count,
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
        }


def _clip(value: Any) -> str:
    text = str(value)
    return text if len(text) <= MAX_VALUE_CHARS else text[: MAX_VALUE_CHARS - 3] + "..."


class _Recorder:
    def __init__(self, contract: Contract, df: pd.DataFrame, raw_df: pd.DataFrame | None):
        self.contract = contract
        self.df = df
        self.raw_df = raw_df
        self.violations: list[Violation] = []
        self.row_failures: dict[int, list[str]] = {}

    def _sample(self, idx: int, column: str) -> dict[str, Any]:
        value = self.df.at[idx, column]
        entry: dict[str, Any] = {"row_number": idx + 2, "value": _clip(value)}
        if self.raw_df is not None and column in self.raw_df.columns:
            raw = self.raw_df.at[idx, column]
            if raw != value:
                entry["raw_value"] = _clip(raw)
        return entry

    def row_rule(self, column: str, rule: str, description: str, mask: pd.Series) -> None:
        mask = mask.astype(bool)
        if not mask.any():
            return
        rule_id = f"{self.contract.name}.{column}.{rule}"
        indices = [int(i) for i in self.df.index[mask]]
        self.violations.append(
            Violation(
                rule_id=rule_id,
                column=column,
                rule=rule,
                description=description,
                severity=ERROR,
                scope="row",
                count=len(indices),
                samples=[self._sample(i, column) for i in indices[:MAX_SAMPLES]],
            )
        )
        for i in indices:
            self.row_failures.setdefault(i, []).append(rule_id)

    def batch_rule(self, column: str | None, rule: str, description: str, severity: str) -> None:
        target = column if column else "batch"
        self.violations.append(
            Violation(
                rule_id=f"{self.contract.name}.{target}.{rule}",
                column=column,
                rule=rule,
                description=description,
                severity=severity,
                scope="batch",
                count=len(self.df),
            )
        )


def _check_field(rec: _Recorder, spec: FieldSpec, series: pd.Series) -> None:
    contract = rec.contract
    name = spec.name
    s = series.astype(str).str.strip()
    nulls = s.isin(contract.null_tokens)
    present = ~nulls

    if not spec.nullable:
        rec.row_rule(name, "not_null", f"{name} must not be null", nulls)

    typed_ok = present.copy()
    numeric: pd.Series | None = None

    if spec.type in ("integer", "decimal"):
        numeric = pd.to_numeric(s.where(present), errors="coerce").astype("float64")
        bad = present & (numeric.isna() | np.isinf(numeric))
        if spec.type == "integer":
            bad = bad | (present & numeric.notna() & ((numeric % 1) != 0))
        rec.row_rule(name, "type", f"{name} must be a valid {spec.type}", bad)
        typed_ok &= ~bad

        if spec.type == "decimal":
            frac = s.str.extract(r"^[+-]?\d*\.(\d+)$")[0]
            too_precise = typed_ok & (frac.str.len() > spec.scale).fillna(False)
            rec.row_rule(
                name,
                "scale",
                f"{name} must have at most {spec.scale} decimal places",
                too_precise,
            )
            typed_ok &= ~too_precise

    elif spec.type == "date":
        fmt = spec.format or "%Y-%m-%d"
        parsed = pd.to_datetime(s.where(present), format=fmt, errors="coerce")
        roundtrip = parsed.dt.strftime(fmt) == s
        ok = parsed.notna() & roundtrip.fillna(False)
        bad = present & ~ok
        rec.row_rule(name, "format", f"{name} must be a date formatted as {fmt}", bad)
        typed_ok &= ~bad

    if spec.pattern:
        matches = s.str.fullmatch(spec.pattern).fillna(False)
        rec.row_rule(
            name, "pattern", f"{name} must match pattern {spec.pattern}", present & ~matches
        )

    if spec.allowed:
        rec.row_rule(
            name,
            "allowed_values",
            f"{name} must be one of {list(spec.allowed)}",
            present & ~s.isin(spec.allowed),
        )

    if numeric is not None and spec.min is not None:
        rec.row_rule(name, "min", f"{name} must be >= {spec.min:g}", typed_ok & (numeric < spec.min))
    if numeric is not None and spec.max is not None:
        rec.row_rule(name, "max", f"{name} must be <= {spec.max:g}", typed_ok & (numeric > spec.max))

    if spec.unique:
        rec.row_rule(
            name,
            "unique",
            f"{name} must be unique within the batch",
            present & s.duplicated(keep="first"),
        )


def validate(
    df: pd.DataFrame,
    contract: Contract,
    batch_id: str,
    raw_df: pd.DataFrame | None = None,
) -> ValidationResult:
    """Validate a batch against the contract.

    ``df`` must have a default RangeIndex so row numbers (file line numbers,
    header = line 1) can be reported. ``raw_df`` is the same batch before
    transformation; when given, samples show the raw value if it differs.
    """
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError("validate() requires a DataFrame with a default RangeIndex")

    rec = _Recorder(contract, df, raw_df)

    if len(df) == 0:
        rec.batch_rule(None, "empty_batch", "The batch contains no data rows", ERROR)

    for col in contract.field_names:
        if col not in df.columns:
            rec.batch_rule(
                col,
                "missing_column",
                f"Required column '{col}' is absent from the source file",
                ERROR,
            )

    for col in df.columns:
        if col not in contract.field_names:
            rec.batch_rule(
                col,
                "unexpected_column",
                f"Column '{col}' is not in the contract and will be dropped",
                WARNING,
            )

    for spec in contract.fields:
        if spec.name in df.columns:
            _check_field(rec, spec, df[spec.name])

    return ValidationResult(
        batch_id=batch_id,
        contract_name=contract.name,
        contract_version=contract.version,
        total_rows=len(df),
        violations=rec.violations,
        row_failures=rec.row_failures,
    )
