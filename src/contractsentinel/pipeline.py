"""Pipeline orchestration: read, transform, validate, decide, load, record.

Flow for one batch file:
  raw CSV -> normalize -> validate (deterministic) -> failure policy -> load
  -> write run artifacts (summary.json, violations.json) for the agent and audit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from contractsentinel.adapters.base import QuarantineRecord, WarehouseAdapter
from contractsentinel.contract import Contract
from contractsentinel.transform import normalize, to_typed
from contractsentinel.validator import ValidationResult, validate

LOAD_ALL = "load_all"
QUARANTINE_ROWS = "quarantine_bad_rows"
BLOCK_BATCH = "block_batch"


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str


@dataclass
class RunResult:
    batch_id: str
    source_file: str
    source_sha256: str
    decision: Decision
    validation: ValidationResult
    rows_total: int
    rows_loaded: int
    rows_quarantined: int
    artifacts_dir: Path
    warehouse: str | None
    dry_run: bool

    @property
    def status(self) -> str:
        if self.decision.action == LOAD_ALL:
            return "loaded"
        if self.decision.action == QUARANTINE_ROWS:
            return "partially_loaded"
        return "blocked"

    @property
    def needs_agent(self) -> bool:
        """True when a contract rule failed, which is the agent's trigger."""
        return not self.validation.passed


def decide(validation: ValidationResult, contract: Contract) -> Decision:
    """Apply the contract's failure policy. Pure and deterministic."""
    if validation.passed:
        return Decision(LOAD_ALL, "No contract violations.")
    if validation.has_batch_level_error:
        return Decision(BLOCK_BATCH, "Batch-level violation (schema or empty batch).")
    if contract.policy.mode == "block_batch":
        return Decision(BLOCK_BATCH, "Policy is block_batch and a rule failed.")
    limit = contract.policy.max_violation_rate
    if validation.invalid_rate > limit:
        return Decision(
            BLOCK_BATCH,
            f"{validation.invalid_rate:.1%} of rows are invalid, above the {limit:.0%} limit.",
        )
    return Decision(
        QUARANTINE_ROWS,
        f"{validation.invalid_row_count} invalid row(s) quarantined; valid rows loaded.",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw(path: Path) -> pd.DataFrame:
    """Read the source file with every value as a string and no NA coercion."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _quarantine_records(
    raw: pd.DataFrame, validation: ValidationResult, blocked: bool
) -> list[QuarantineRecord]:
    if blocked:
        indices = list(raw.index)
    else:
        indices = validation.invalid_index()
    records = []
    for idx in indices:
        rules = list(validation.row_failures.get(idx, []))
        if not rules:
            rules = ["batch.blocked"]
        records.append(
            QuarantineRecord(
                row_number=int(idx) + 2,
                rules=rules,
                raw_record={str(k): str(v) for k, v in raw.loc[idx].to_dict().items()},
            )
        )
    return records


def _write_artifacts(result: RunResult, contract: Contract) -> None:
    out = result.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "batch_id": result.batch_id,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": result.source_file,
        "source_sha256": result.source_sha256,
        "contract": {"name": contract.name, "version": contract.version},
        "warehouse": result.warehouse,
        "dry_run": result.dry_run,
        "status": result.status,
        "decision": {"action": result.decision.action, "reason": result.decision.reason},
        "needs_agent": result.needs_agent,
        "counts": {
            "rows_total": result.rows_total,
            "rows_loaded": result.rows_loaded,
            "rows_quarantined": result.rows_quarantined,
            "rows_invalid": result.validation.invalid_row_count,
        },
        "destination": {
            "clean_table": contract.clean_table,
            "quarantine_table": contract.quarantine_full_name,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "violations.json").write_text(
        json.dumps(result.validation.to_dict(), indent=2), encoding="utf-8"
    )


def run_batch(
    path: str | Path,
    contract: Contract,
    adapter: WarehouseAdapter | None,
    artifacts_root: str | Path = "runs",
    dry_run: bool = False,
) -> RunResult:
    """Process one batch file end to end. ``adapter`` may be None when dry_run=True."""
    path = Path(path)
    if not dry_run and adapter is None:
        raise ValueError("an adapter is required unless dry_run=True")

    batch_id = path.stem
    raw = read_raw(path)
    transformed = normalize(raw)
    validation = validate(transformed, contract, batch_id, raw_df=raw)
    decision = decide(validation, contract)

    blocked = decision.action == BLOCK_BATCH
    if blocked:
        valid_rows = transformed.iloc[0:0]
    else:
        invalid = set(validation.invalid_index())
        valid_rows = transformed[~transformed.index.isin(invalid)]

    quarantine = _quarantine_records(raw, validation, blocked) if not validation.passed else []

    if not dry_run:
        assert adapter is not None
        adapter.ensure_tables(contract)
        # A blocked batch loads nothing, and may lack contract columns entirely,
        # so only cast when there are rows to load.
        typed = (
            to_typed(valid_rows, contract)
            if len(valid_rows)
            else pd.DataFrame(columns=list(contract.field_names))
        )
        adapter.replace_batch_clean(contract, batch_id, typed)
        adapter.replace_batch_quarantine(contract, batch_id, quarantine)

    result = RunResult(
        batch_id=batch_id,
        source_file=str(path),
        source_sha256=_sha256(path),
        decision=decision,
        validation=validation,
        rows_total=len(raw),
        rows_loaded=0 if dry_run else len(valid_rows),
        rows_quarantined=len(quarantine),
        artifacts_dir=Path(artifacts_root) / batch_id,
        warehouse=None if adapter is None else adapter.name,
        dry_run=dry_run,
    )
    _write_artifacts(result, contract)
    return result
