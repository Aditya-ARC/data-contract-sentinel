"""DuckDB warehouse adapter: free, local, and the default."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from contractsentinel.adapters.base import QuarantineRecord, WarehouseAdapter
from contractsentinel.contract import Contract

_TYPE_MAP = {
    "string": "VARCHAR",
    "integer": "BIGINT",
    "date": "DATE",
}


def _column_type(spec) -> str:
    if spec.type == "decimal":
        return f"DECIMAL(18,{spec.scale})"
    return _TYPE_MAP[spec.type]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DuckDBAdapter(WarehouseAdapter):
    name = "duckdb"

    def __init__(self, database: str = "data/warehouse.duckdb") -> None:
        if database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        self.database = database
        self.con = duckdb.connect(database)

    def ensure_tables(self, contract: Contract) -> None:
        cols = ",\n  ".join(f"{f.name} {_column_type(f)}" for f in contract.fields)
        self.con.execute(f"CREATE SCHEMA IF NOT EXISTS {contract.schema}")
        self.con.execute(
            f"""CREATE TABLE IF NOT EXISTS {contract.clean_table} (
  {cols},
  _batch_id VARCHAR,
  _loaded_at TIMESTAMP
)"""
        )
        self.con.execute(
            f"""CREATE TABLE IF NOT EXISTS {contract.quarantine_full_name} (
  _batch_id VARCHAR,
  row_number INTEGER,
  rules VARCHAR,
  raw_record VARCHAR,
  _quarantined_at TIMESTAMP
)"""
        )

    def replace_batch_clean(self, contract: Contract, batch_id: str, df: pd.DataFrame) -> None:
        columns = list(contract.field_names)
        self.con.execute("BEGIN")
        try:
            self.con.execute(f"DELETE FROM {contract.clean_table} WHERE _batch_id = ?", [batch_id])
            if len(df):
                staged = df[columns].copy()
                staged["_batch_id"] = batch_id
                staged["_loaded_at"] = _utc_now()
                self.con.register("staged_clean", staged)
                try:
                    col_list = ", ".join(columns + ["_batch_id", "_loaded_at"])
                    self.con.execute(
                        f"INSERT INTO {contract.clean_table} ({col_list}) "
                        f"SELECT {col_list} FROM staged_clean"
                    )
                finally:
                    self.con.unregister("staged_clean")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def replace_batch_quarantine(
        self, contract: Contract, batch_id: str, records: list[QuarantineRecord]
    ) -> None:
        self.con.execute("BEGIN")
        try:
            self.con.execute(
                f"DELETE FROM {contract.quarantine_full_name} WHERE _batch_id = ?", [batch_id]
            )
            if records:
                now = _utc_now()
                staged = pd.DataFrame(
                    {
                        "_batch_id": [batch_id] * len(records),
                        "row_number": [r.row_number for r in records],
                        "rules": [json.dumps(r.rules) for r in records],
                        "raw_record": [json.dumps(r.raw_record) for r in records],
                        "_quarantined_at": [now] * len(records),
                    }
                )
                self.con.register("staged_quarantine", staged)
                try:
                    self.con.execute(
                        f"INSERT INTO {contract.quarantine_full_name} "
                        "(_batch_id, row_number, rules, raw_record, _quarantined_at) "
                        "SELECT _batch_id, row_number, rules, raw_record, _quarantined_at "
                        "FROM staged_quarantine"
                    )
                finally:
                    self.con.unregister("staged_quarantine")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def count_rows(self, table: str, batch_id: str | None = None) -> int:
        if batch_id is None:
            row = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        else:
            row = self.con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE _batch_id = ?", [batch_id]
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self.con.close()
