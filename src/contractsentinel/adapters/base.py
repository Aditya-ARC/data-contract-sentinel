"""Warehouse adapter interface.

Adapters own all destination-specific SQL. The pipeline only talks to this
interface, so DuckDB (local) and Snowflake (production-like) are swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from contractsentinel.contract import Contract


@dataclass(frozen=True)
class QuarantineRecord:
    row_number: int  # file line number, header = line 1
    rules: list[str]
    raw_record: dict[str, str]  # exactly what the source sent


class WarehouseAdapter(ABC):
    name: str

    @abstractmethod
    def ensure_tables(self, contract: Contract) -> None:
        """Create the schema, clean table and quarantine table if missing."""

    @abstractmethod
    def replace_batch_clean(self, contract: Contract, batch_id: str, df: pd.DataFrame) -> None:
        """Atomically replace all clean rows for ``batch_id`` with ``df`` (may be empty)."""

    @abstractmethod
    def replace_batch_quarantine(
        self, contract: Contract, batch_id: str, records: list[QuarantineRecord]
    ) -> None:
        """Atomically replace all quarantined rows for ``batch_id`` (may be empty)."""

    @abstractmethod
    def count_rows(self, table: str, batch_id: str | None = None) -> int:
        """Row count for a fully qualified table, optionally for one batch."""

    @abstractmethod
    def close(self) -> None:
        """Release the connection."""

    def __enter__(self) -> "WarehouseAdapter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def to_python(value: Any) -> Any:
    """Convert pandas/numpy scalars to plain Python values (NA -> None)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def rows_for_insert(df: pd.DataFrame, columns: list[str]) -> list[tuple[Any, ...]]:
    return [tuple(to_python(row[c]) for c in columns) for _, row in df.iterrows()]
