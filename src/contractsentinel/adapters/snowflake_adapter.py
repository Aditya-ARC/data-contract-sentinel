"""Snowflake warehouse adapter.

Mirrors the DuckDB adapter's behavior. The connector is imported lazily so the
project installs and runs without it. Install with:

    pip install "data-contract-sentinel[snowflake]"

Authentication comes from environment variables (see .env.example). Key-pair
auth is supported and preferred; password auth is available for accounts that
still allow it.

Inserts use parameterized executemany, which is fine for demo-sized batches.
For large batches, stage the data and use COPY INTO or write_pandas instead.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from contractsentinel.adapters.base import QuarantineRecord, WarehouseAdapter, rows_for_insert
from contractsentinel.contract import Contract

_TYPE_MAP = {
    "string": "VARCHAR",
    "integer": "NUMBER(38,0)",
    "date": "DATE",
}

_REQUIRED_ENV = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE")


def _column_type(spec) -> str:
    if spec.type == "decimal":
        return f"NUMBER(18,{spec.scale})"
    return _TYPE_MAP[spec.type]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def connection_params_from_env() -> dict[str, Any]:
    missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing Snowflake settings: {', '.join(missing)}. See .env.example.")

    params: dict[str, Any] = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        "database": os.environ["SNOWFLAKE_DATABASE"],
    }
    if os.getenv("SNOWFLAKE_ROLE"):
        params["role"] = os.environ["SNOWFLAKE_ROLE"]

    key_file = os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    if key_file:
        params["private_key_file"] = key_file
        if os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"):
            params["private_key_file_pwd"] = os.environ["SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"]
    elif password:
        params["password"] = password
    else:
        raise RuntimeError(
            "Set SNOWFLAKE_PRIVATE_KEY_FILE (preferred) or SNOWFLAKE_PASSWORD. See .env.example."
        )
    return params


def _connect() -> Any:
    params = connection_params_from_env()  # configuration problems are reported first
    try:
        import snowflake.connector  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            'snowflake-connector-python is not installed. Run: pip install "data-contract-sentinel[snowflake]"'
        ) from exc
    return snowflake.connector.connect(**params)


class SnowflakeAdapter(WarehouseAdapter):
    name = "snowflake"

    def __init__(self, connection: Any | None = None) -> None:
        """Pass ``connection`` to inject a connection (used by tests)."""
        self.conn = connection if connection is not None else _connect()

    def _execute(self, sql: str, params: list[Any] | None = None) -> Any:
        cur = self.conn.cursor()
        try:
            cur.execute(sql, params) if params is not None else cur.execute(sql)
            return cur
        except Exception:
            cur.close()
            raise

    def _transaction(self, statements) -> None:
        self._execute("BEGIN").close()
        try:
            statements()
            self._execute("COMMIT").close()
        except Exception:
            self._execute("ROLLBACK").close()
            raise

    def ensure_tables(self, contract: Contract) -> None:
        cols = ",\n  ".join(f"{f.name} {_column_type(f)}" for f in contract.fields)
        self._execute(f"CREATE SCHEMA IF NOT EXISTS {contract.schema}").close()
        self._execute(
            f"""CREATE TABLE IF NOT EXISTS {contract.clean_table} (
  {cols},
  _batch_id VARCHAR,
  _loaded_at TIMESTAMP_NTZ
)"""
        ).close()
        self._execute(
            f"""CREATE TABLE IF NOT EXISTS {contract.quarantine_full_name} (
  _batch_id VARCHAR,
  row_number NUMBER(38,0),
  rules VARCHAR,
  raw_record VARCHAR,
  _quarantined_at TIMESTAMP_NTZ
)"""
        ).close()

    def replace_batch_clean(self, contract: Contract, batch_id: str, df: pd.DataFrame) -> None:
        columns = list(contract.field_names)

        def work() -> None:
            self._execute(
                f"DELETE FROM {contract.clean_table} WHERE _batch_id = %s", [batch_id]
            ).close()
            if len(df):
                now = _utc_now()
                rows = [row + (batch_id, now) for row in rows_for_insert(df, columns)]
                col_list = ", ".join(columns + ["_batch_id", "_loaded_at"])
                placeholders = ", ".join(["%s"] * (len(columns) + 2))
                cur = self.conn.cursor()
                try:
                    cur.executemany(
                        f"INSERT INTO {contract.clean_table} ({col_list}) VALUES ({placeholders})",
                        rows,
                    )
                finally:
                    cur.close()

        self._transaction(work)

    def replace_batch_quarantine(
        self, contract: Contract, batch_id: str, records: list[QuarantineRecord]
    ) -> None:
        def work() -> None:
            self._execute(
                f"DELETE FROM {contract.quarantine_full_name} WHERE _batch_id = %s", [batch_id]
            ).close()
            if records:
                now = _utc_now()
                rows = [
                    (batch_id, r.row_number, json.dumps(r.rules), json.dumps(r.raw_record), now)
                    for r in records
                ]
                cur = self.conn.cursor()
                try:
                    cur.executemany(
                        f"INSERT INTO {contract.quarantine_full_name} "
                        "(_batch_id, row_number, rules, raw_record, _quarantined_at) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        rows,
                    )
                finally:
                    cur.close()

        self._transaction(work)

    def count_rows(self, table: str, batch_id: str | None = None) -> int:
        if batch_id is None:
            cur = self._execute(f"SELECT COUNT(*) FROM {table}")
        else:
            cur = self._execute(f"SELECT COUNT(*) FROM {table} WHERE _batch_id = %s", [batch_id])
        try:
            return int(cur.fetchone()[0])
        finally:
            cur.close()

    def close(self) -> None:
        self.conn.close()
