"""Pipeline transformation code.

This is the intentionally small piece of "pipeline code" that sits between the
raw source file and validation. It is also what the investigation agent reads
when it traces a contract failure back to a cause.

Two steps:
  1. normalize(): light cleanup applied before validation.
  2. to_typed(): casts rows that already passed validation to destination types.
"""

from __future__ import annotations

import pandas as pd

from contractsentinel.contract import Contract

# Source sends mixed case for these; the contract expects a fixed case.
CASE_RULES = {"status": "lower", "region": "upper"}


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace, fix known casing, and normalize slash-separated dates."""
    out = raw.copy()
    for col in out.columns:
        out[col] = out[col].astype(str).str.strip()

    for col, mode in CASE_RULES.items():
        if col in out.columns:
            out[col] = out[col].str.lower() if mode == "lower" else out[col].str.upper()

    if "order_date" in out.columns:
        out["order_date"] = out["order_date"].str.replace(
            r"^(\d{4})/(\d{2})/(\d{2})$", r"\1-\2-\3", regex=True
        )

    return out


def to_typed(valid: pd.DataFrame, contract: Contract) -> pd.DataFrame:
    """Cast already-validated rows to destination types. Drops non-contract columns."""
    out = pd.DataFrame(index=valid.index)
    for spec in contract.fields:
        s = valid[spec.name].astype(str).str.strip()
        null = s.isin(contract.null_tokens)
        if spec.type == "integer":
            out[spec.name] = pd.to_numeric(s.where(~null), errors="raise").astype("Int64")
        elif spec.type == "decimal":
            out[spec.name] = pd.to_numeric(s.where(~null), errors="raise").astype("Float64")
        elif spec.type == "date":
            parsed = pd.to_datetime(s.where(~null), format=spec.format, errors="raise")
            out[spec.name] = pd.Series(
                [None if pd.isna(v) else v.date() for v in parsed],
                index=valid.index,
                dtype="object",
            )
        else:
            out[spec.name] = pd.Series(
                [None if n else v for v, n in zip(s, null)],
                index=valid.index,
                dtype="object",
            )
    return out
