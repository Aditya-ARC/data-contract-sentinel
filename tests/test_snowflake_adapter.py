"""The Snowflake adapter is tested against a fake connection that records SQL.

This verifies the SQL and the transaction handling, not behavior against a real
Snowflake account.
"""

from __future__ import annotations

import datetime as dt
import sys

import pytest

from contractsentinel.adapters import get_adapter
from contractsentinel.adapters.base import QuarantineRecord
from contractsentinel.adapters.snowflake_adapter import (
    SnowflakeAdapter,
    _connect,
    connection_params_from_env,
)
from contractsentinel.transform import normalize, to_typed

ENV_KEYS = [
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_ROLE",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_PRIVATE_KEY_FILE",
    "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE",
]


def squash(sql: str) -> str:
    return " ".join(sql.split())


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def _maybe_fail(self, sql):
        if self.conn.fail_on and self.conn.fail_on in sql:
            raise RuntimeError("boom")

    def execute(self, sql, params=None):
        self.conn.log.append(("execute", squash(sql), params))
        self._maybe_fail(sql)

    def executemany(self, sql, rows):
        self.conn.log.append(("executemany", squash(sql), list(rows)))
        self._maybe_fail(sql)

    def fetchone(self):
        return (self.conn.count,)

    def close(self):
        pass


class FakeConnection:
    def __init__(self, count=0, fail_on=None):
        self.log = []
        self.count = count
        self.fail_on = fail_on
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True

    def statements(self):
        return [entry[1] for entry in self.log]


@pytest.fixture()
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_ensure_tables_creates_schema_and_both_tables(contract):
    conn = FakeConnection()
    SnowflakeAdapter(conn).ensure_tables(contract)
    sql = " | ".join(conn.statements())
    assert "CREATE SCHEMA IF NOT EXISTS analytics" in sql
    assert "CREATE TABLE IF NOT EXISTS analytics.orders_clean" in sql
    assert "amount NUMBER(18,2)" in sql and "order_date DATE" in sql
    assert "_loaded_at TIMESTAMP_NTZ" in sql
    assert "CREATE TABLE IF NOT EXISTS analytics.orders_quarantine" in sql


def test_replace_batch_clean_runs_delete_then_insert_in_a_transaction(contract, make_df):
    conn = FakeConnection()
    typed = to_typed(normalize(make_df({"amount": "12.30"}, {"amount": "7"})), contract)

    SnowflakeAdapter(conn).replace_batch_clean(contract, "b1", typed)

    kinds = [(k, s.split(" ")[0]) for k, s, _ in conn.log]
    assert kinds == [
        ("execute", "BEGIN"),
        ("execute", "DELETE"),
        ("executemany", "INSERT"),
        ("execute", "COMMIT"),
    ]
    delete = conn.log[1]
    assert delete[1] == "DELETE FROM analytics.orders_clean WHERE _batch_id = %s"
    assert delete[2] == ["b1"]

    insert_sql, rows = conn.log[2][1], conn.log[2][2]
    assert insert_sql.startswith(
        "INSERT INTO analytics.orders_clean (order_id, customer_id, order_date, amount, status, region, _batch_id, _loaded_at)"
    )
    assert insert_sql.count("%s") == 8
    assert len(rows) == 2
    first = rows[0]
    assert first[0] == "O100001" and first[2] == dt.date(2026, 10, 1)
    assert first[3] == 12.3 and type(first[3]) is float  # plain Python, not numpy
    assert first[6] == "b1" and isinstance(first[7], dt.datetime)


def test_replace_batch_clean_with_no_rows_only_deletes(contract, make_df):
    conn = FakeConnection()
    empty = to_typed(make_df({}).iloc[0:0], contract)
    SnowflakeAdapter(conn).replace_batch_clean(contract, "b1", empty)
    assert [s.split(" ")[0] for s in conn.statements()] == ["BEGIN", "DELETE", "COMMIT"]


def test_failed_insert_rolls_back_and_reraises(contract, make_df):
    conn = FakeConnection(fail_on="INSERT INTO")
    typed = to_typed(normalize(make_df({})), contract)
    with pytest.raises(RuntimeError, match="boom"):
        SnowflakeAdapter(conn).replace_batch_clean(contract, "b1", typed)
    statements = conn.statements()
    assert statements[-1] == "ROLLBACK" and "COMMIT" not in statements


def test_replace_batch_quarantine_serializes_rules_and_raw_record(contract):
    conn = FakeConnection()
    records = [
        QuarantineRecord(3, ["orders.customer_id.not_null"], {"order_id": "O1", "customer_id": ""})
    ]
    SnowflakeAdapter(conn).replace_batch_quarantine(contract, "b1", records)

    delete = next(e for e in conn.log if e[1].startswith("DELETE"))
    assert delete[1] == "DELETE FROM analytics.orders_quarantine WHERE _batch_id = %s"
    insert = next(e for e in conn.log if e[0] == "executemany")
    row = insert[2][0]
    assert row[:2] == ("b1", 3)
    assert row[2] == '["orders.customer_id.not_null"]'
    assert row[3] == '{"order_id": "O1", "customer_id": ""}'


def test_count_rows_with_and_without_batch(contract):
    conn = FakeConnection(count=5)
    adapter = SnowflakeAdapter(conn)
    assert adapter.count_rows("analytics.orders_clean") == 5
    assert adapter.count_rows("analytics.orders_clean", "b1") == 5
    assert conn.log[0][1] == "SELECT COUNT(*) FROM analytics.orders_clean"
    assert conn.log[1][2] == ["b1"]


def test_close_closes_the_connection():
    conn = FakeConnection()
    with SnowflakeAdapter(conn):
        pass
    assert conn.closed


# ---- configuration --------------------------------------------------------------------


def test_missing_settings_are_listed(clean_env):
    clean_env.setenv("SNOWFLAKE_ACCOUNT", "acct")
    with pytest.raises(RuntimeError) as err:
        connection_params_from_env()
    message = str(err.value)
    assert "SNOWFLAKE_USER" in message and "SNOWFLAKE_ACCOUNT" not in message


def _base_env(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    monkeypatch.setenv("SNOWFLAKE_USER", "me")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "wh")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "db")


def test_key_pair_auth_is_preferred_over_password(clean_env):
    _base_env(clean_env)
    clean_env.setenv("SNOWFLAKE_PASSWORD", "pw")
    clean_env.setenv("SNOWFLAKE_PRIVATE_KEY_FILE", "/keys/rsa.p8")
    clean_env.setenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "phrase")
    params = connection_params_from_env()
    assert params["private_key_file"] == "/keys/rsa.p8"
    assert params["private_key_file_pwd"] == "phrase"
    assert "password" not in params


def test_password_auth_and_optional_role(clean_env):
    _base_env(clean_env)
    clean_env.setenv("SNOWFLAKE_PASSWORD", "pw")
    clean_env.setenv("SNOWFLAKE_ROLE", "LOADER")
    params = connection_params_from_env()
    assert params["password"] == "pw" and params["role"] == "LOADER"


def test_some_credential_is_required(clean_env):
    _base_env(clean_env)
    with pytest.raises(RuntimeError, match="SNOWFLAKE_PRIVATE_KEY_FILE"):
        connection_params_from_env()


def test_missing_connector_gives_an_install_hint(clean_env):
    _base_env(clean_env)
    clean_env.setenv("SNOWFLAKE_PASSWORD", "pw")
    clean_env.setitem(sys.modules, "snowflake", None)  # makes the import fail
    with pytest.raises(RuntimeError, match=r"data-contract-sentinel\[snowflake\]"):
        _connect()


def test_unknown_warehouse_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown warehouse"):
        get_adapter("oracle")
