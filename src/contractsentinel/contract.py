"""Load and validate the data contract that drives the pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_TYPES = ("string", "integer", "decimal", "date")
POLICY_MODES = ("quarantine", "block_batch")
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
DEFAULT_DECIMAL_SCALE = 2

# Identifiers from the contract end up in SQL, so they are restricted.
IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")

_FIELD_KEYS = {
    "name",
    "type",
    "nullable",
    "unique",
    "pattern",
    "allowed",
    "format",
    "scale",
    "min",
    "max",
    "description",
}


class ContractError(ValueError):
    """Raised when a contract file is malformed."""


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str = "string"
    nullable: bool = True
    unique: bool = False
    pattern: str | None = None
    allowed: tuple[str, ...] | None = None
    format: str | None = None
    scale: int = DEFAULT_DECIMAL_SCALE
    min: float | None = None
    max: float | None = None
    description: str = ""


@dataclass(frozen=True)
class FailurePolicy:
    mode: str = "quarantine"
    max_violation_rate: float = 0.2


@dataclass(frozen=True)
class Contract:
    name: str
    version: str
    owner: str
    description: str
    source_system: str
    null_tokens: tuple[str, ...]
    schema: str
    table: str
    quarantine_table: str
    policy: FailurePolicy
    fields: tuple[FieldSpec, ...]

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    @property
    def clean_table(self) -> str:
        return f"{self.schema}.{self.table}"

    @property
    def quarantine_full_name(self) -> str:
        return f"{self.schema}.{self.quarantine_table}"


def _require(mapping: Any, key: str, where: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise ContractError(f"{where}: missing required key '{key}'")
    return mapping[key]


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.match(value):
        raise ContractError(
            f"{where}: {value!r} is not a valid identifier "
            "(lowercase letters, digits and underscores, not starting with a digit)"
        )
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}: expected a number, got {value!r}")
    return float(value)


def _parse_field(item: Any, index: int) -> FieldSpec:
    where = f"fields[{index}]"
    if not isinstance(item, dict):
        raise ContractError(f"{where}: expected a mapping")
    unknown = set(item) - _FIELD_KEYS
    if unknown:
        raise ContractError(f"{where}: unknown keys {sorted(unknown)}")

    name = _identifier(_require(item, "name", where), f"{where}.name")
    where = f"fields[{name}]"
    ftype = item.get("type", "string")
    if ftype not in SUPPORTED_TYPES:
        raise ContractError(f"{where}: unsupported type {ftype!r}; use one of {SUPPORTED_TYPES}")

    pattern = item.get("pattern")
    if pattern is not None:
        if ftype != "string":
            raise ContractError(f"{where}: 'pattern' only applies to string fields")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ContractError(f"{where}: invalid regex {pattern!r}: {exc}") from exc

    allowed = item.get("allowed")
    if allowed is not None:
        if ftype != "string":
            raise ContractError(f"{where}: 'allowed' only applies to string fields")
        if not isinstance(allowed, list) or not allowed:
            raise ContractError(f"{where}: 'allowed' must be a non-empty list")
        allowed = tuple(str(v) for v in allowed)

    fmt = item.get("format")
    if ftype == "date":
        fmt = fmt or DEFAULT_DATE_FORMAT
    elif fmt is not None:
        raise ContractError(f"{where}: 'format' only applies to date fields")

    scale = item.get("scale")
    if scale is not None and ftype != "decimal":
        raise ContractError(f"{where}: 'scale' only applies to decimal fields")
    scale = DEFAULT_DECIMAL_SCALE if scale is None else int(scale)
    if not 0 <= scale <= 12:
        raise ContractError(f"{where}: 'scale' must be between 0 and 12")

    lo = item.get("min")
    hi = item.get("max")
    if (lo is not None or hi is not None) and ftype not in ("integer", "decimal"):
        raise ContractError(f"{where}: 'min'/'max' only apply to integer and decimal fields")
    lo = None if lo is None else _number(lo, f"{where}.min")
    hi = None if hi is None else _number(hi, f"{where}.max")
    if lo is not None and hi is not None and lo > hi:
        raise ContractError(f"{where}: min is greater than max")

    return FieldSpec(
        name=name,
        type=ftype,
        nullable=bool(item.get("nullable", True)),
        unique=bool(item.get("unique", False)),
        pattern=pattern,
        allowed=allowed,
        format=fmt,
        scale=scale,
        min=lo,
        max=hi,
        description=str(item.get("description", "")),
    )


def parse_contract(raw: Any) -> Contract:
    if not isinstance(raw, dict):
        raise ContractError("contract file must contain a YAML mapping")

    meta = _require(raw, "contract", "contract file")
    source = _require(raw, "source", "contract file")
    dest = _require(raw, "destination", "contract file")
    policy_raw = raw.get("failure_policy") or {}
    fields_raw = _require(raw, "fields", "contract file")

    name = _identifier(_require(meta, "name", "contract"), "contract.name")
    version = str(_require(meta, "version", "contract"))

    mode = policy_raw.get("mode", "quarantine")
    if mode not in POLICY_MODES:
        raise ContractError(f"failure_policy.mode: {mode!r} is not one of {POLICY_MODES}")
    rate = _number(policy_raw.get("max_violation_rate", 0.2), "failure_policy.max_violation_rate")
    if not 0.0 <= rate <= 1.0:
        raise ContractError("failure_policy.max_violation_rate must be between 0 and 1")

    if not isinstance(fields_raw, list) or not fields_raw:
        raise ContractError("fields: must be a non-empty list")
    fields = tuple(_parse_field(item, i) for i, item in enumerate(fields_raw))
    names = [f.name for f in fields]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ContractError(f"fields: duplicate field names {dupes}")

    null_tokens = source.get("null_tokens", [""])
    if not isinstance(null_tokens, list):
        raise ContractError("source.null_tokens must be a list")

    return Contract(
        name=name,
        version=version,
        owner=str(meta.get("owner", "")),
        description=str(meta.get("description", "")),
        source_system=str(source.get("system", "")),
        null_tokens=tuple(str(t) for t in null_tokens),
        schema=_identifier(_require(dest, "schema", "destination"), "destination.schema"),
        table=_identifier(_require(dest, "table", "destination"), "destination.table"),
        quarantine_table=_identifier(
            _require(dest, "quarantine_table", "destination"), "destination.quarantine_table"
        ),
        policy=FailurePolicy(mode=mode, max_violation_rate=rate),
        fields=fields,
    )


def load_contract(path: str | Path) -> Contract:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"contract file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ContractError(f"contract file is not valid YAML: {exc}") from exc
    return parse_contract(raw)
