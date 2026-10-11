from __future__ import annotations

import dataclasses
import json

import pandas as pd
import pytest

from contractsentinel.adapters.duckdb_adapter import DuckDBAdapter
from contractsentinel.pipeline import BLOCK_BATCH, LOAD_ALL, QUARANTINE_ROWS, decide, run_batch
from contractsentinel.sample_data import BAD_NAME, GOOD_NAME, write_samples
from contractsentinel.validator import validate


@pytest.fixture()
def adapter():
    with DuckDBAdapter(":memory:") as a:
        yield a


@pytest.fixture()
def samples(tmp_path):
    good, bad = write_samples(tmp_path / "landing")
    return good, bad


def clean_count(adapter, contract, batch=None):
    return adapter.count_rows(contract.clean_table, batch)


def quarantine_count(adapter, contract, batch=None):
    return adapter.count_rows(contract.quarantine_full_name, batch)


# ---- decide() -------------------------------------------------------------------------


def test_decide_loads_everything_when_clean(contract, make_df):
    result = validate(make_df({}, {}), contract, "b")
    assert decide(result, contract).action == LOAD_ALL


def test_decide_quarantines_a_small_share_of_bad_rows(contract, make_df):
    df = make_df(*[{} for _ in range(9)], {"customer_id": ""})  # 10%
    assert decide(validate(df, contract, "b"), contract).action == QUARANTINE_ROWS


def test_decide_blocks_above_the_violation_rate(contract, make_df):
    df = make_df({}, {"customer_id": ""}, {"customer_id": ""})  # 67%
    decision = decide(validate(df, contract, "b"), contract)
    assert decision.action == BLOCK_BATCH and "limit" in decision.reason


def test_decide_blocks_on_schema_problems(contract, make_df):
    df = make_df({}).drop(columns=["amount"])
    assert decide(validate(df, contract, "b"), contract).action == BLOCK_BATCH


def test_decide_blocks_any_failure_in_block_batch_mode(contract, make_df):
    strict = dataclasses.replace(
        contract, policy=dataclasses.replace(contract.policy, mode="block_batch")
    )
    df = make_df(*[{} for _ in range(19)], {"customer_id": ""})  # 5%
    assert decide(validate(df, strict, "b"), strict).action == BLOCK_BATCH


def test_warnings_alone_do_not_block(contract, make_df):
    df = make_df({}, {})
    df["coupon"] = "X"
    assert decide(validate(df, contract, "b"), contract).action == LOAD_ALL


# ---- end to end -----------------------------------------------------------------------


def test_clean_batch_loads_fully(contract, adapter, samples, tmp_path):
    good, _ = samples
    result = run_batch(good, contract, adapter, tmp_path / "runs")

    assert result.status == "loaded" and not result.needs_agent
    assert (result.rows_total, result.rows_loaded, result.rows_quarantined) == (40, 40, 0)
    assert clean_count(adapter, contract) == 40
    assert quarantine_count(adapter, contract) == 0


def test_bad_batch_is_split_into_clean_and_quarantine(contract, adapter, samples, tmp_path):
    _, bad = samples
    result = run_batch(bad, contract, adapter, tmp_path / "runs")

    assert result.status == "partially_loaded" and result.needs_agent
    assert (result.rows_loaded, result.rows_quarantined) == (33, 7)
    assert clean_count(adapter, contract) == 33
    assert quarantine_count(adapter, contract) == 7


def test_nothing_invalid_reaches_the_clean_table(contract, adapter, samples, tmp_path):
    _, bad = samples
    run_batch(bad, contract, adapter, tmp_path / "runs")
    con = adapter.con
    table = contract.clean_table
    assert con.execute(f"SELECT COUNT(*) FROM {table} WHERE customer_id IS NULL").fetchone()[0] == 0
    assert con.execute(f"SELECT COUNT(*) FROM {table} WHERE amount < 0").fetchone()[0] == 0
    assert (
        con.execute(f"SELECT COUNT(*) FROM {table} WHERE status NOT IN "
                    "('pending','shipped','delivered','cancelled','refunded')").fetchone()[0] == 0
    )
    assert con.execute(f"SELECT COUNT(DISTINCT order_id) - COUNT(*) FROM {table}").fetchone()[0] == 0


def test_clean_table_uses_destination_types(contract, adapter, samples, tmp_path):
    good, _ = samples
    run_batch(good, contract, adapter, tmp_path / "runs")
    types = dict(
        adapter.con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'analytics' AND table_name = 'orders_clean'"
        ).fetchall()
    )
    assert types["order_date"] == "DATE"
    assert types["amount"] == "DECIMAL(18,2)"
    assert types["order_id"] == "VARCHAR"


def test_quarantine_keeps_the_raw_source_values(contract, adapter, samples, tmp_path):
    _, bad = samples
    run_batch(bad, contract, adapter, tmp_path / "runs")
    rows = adapter.con.execute(
        f"SELECT row_number, rules, raw_record FROM {contract.quarantine_full_name} ORDER BY row_number"
    ).fetchall()
    records = {r[0]: (json.loads(r[1]), json.loads(r[2])) for r in rows}
    raw_values = {rec["order_date"] for _, rec in records.values()}
    amounts = {rec["amount"] for _, rec in records.values()}
    assert "10/02/2026" in raw_values  # the original, not a transformed value
    assert "twelve dollars" in amounts
    assert all(rules for rules, _ in records.values())


def test_rerunning_a_batch_is_idempotent(contract, adapter, samples, tmp_path):
    _, bad = samples
    for _ in range(3):
        run_batch(bad, contract, adapter, tmp_path / "runs")
    assert clean_count(adapter, contract) == 33
    assert quarantine_count(adapter, contract) == 7


def test_fixing_a_batch_and_rerunning_replaces_its_rows(contract, adapter, samples, tmp_path):
    good, bad = samples
    broken = tmp_path / "orders_fix.csv"
    broken.write_bytes(bad.read_bytes())
    run_batch(broken, contract, adapter, tmp_path / "runs")
    assert quarantine_count(adapter, contract, "orders_fix") == 7

    broken.write_bytes(good.read_bytes())  # source resends a corrected file under the same name
    result = run_batch(broken, contract, adapter, tmp_path / "runs")
    assert result.status == "loaded"
    assert clean_count(adapter, contract, "orders_fix") == 40
    assert quarantine_count(adapter, contract, "orders_fix") == 0


def test_batches_do_not_interfere_with_each_other(contract, adapter, samples, tmp_path):
    good, bad = samples
    run_batch(good, contract, adapter, tmp_path / "runs")
    run_batch(bad, contract, adapter, tmp_path / "runs")
    assert clean_count(adapter, contract, good.stem) == 40
    assert clean_count(adapter, contract, bad.stem) == 33
    assert clean_count(adapter, contract) == 73


def test_batch_over_the_limit_is_blocked_and_fully_quarantined(contract, adapter, make_df, tmp_path):
    df = make_df(*[{"customer_id": ""} for _ in range(10)])
    path = tmp_path / "orders_all_null.csv"
    df.to_csv(path, index=False)

    result = run_batch(path, contract, adapter, tmp_path / "runs")
    assert result.status == "blocked"
    assert clean_count(adapter, contract) == 0
    assert quarantine_count(adapter, contract) == 10


def test_blocked_batch_quarantines_rows_that_were_individually_fine(contract, adapter, make_df, tmp_path):
    df = make_df(*[{"customer_id": ""} for _ in range(5)], *[{} for _ in range(3)])
    path = tmp_path / "orders_mixed.csv"
    df.to_csv(path, index=False)

    run_batch(path, contract, adapter, tmp_path / "runs")
    rules = [
        json.loads(r[0])
        for r in adapter.con.execute(
            f"SELECT rules FROM {contract.quarantine_full_name} ORDER BY row_number"
        ).fetchall()
    ]
    assert rules[-1] == ["batch.blocked"]  # a valid row held back with the batch
    assert rules[0] == ["orders.customer_id.not_null"]


def test_missing_column_blocks_the_batch(contract, adapter, make_df, tmp_path):
    path = tmp_path / "orders_drift.csv"
    make_df({}, {}).drop(columns=["customer_id"]).to_csv(path, index=False)

    result = run_batch(path, contract, adapter, tmp_path / "runs")
    assert result.status == "blocked" and result.needs_agent
    assert clean_count(adapter, contract) == 0
    assert quarantine_count(adapter, contract) == 2


def test_artifacts_are_written_for_the_agent(contract, adapter, samples, tmp_path):
    _, bad = samples
    result = run_batch(bad, contract, adapter, tmp_path / "runs")

    summary = json.loads((result.artifacts_dir / "summary.json").read_text())
    violations = json.loads((result.artifacts_dir / "violations.json").read_text())

    assert result.artifacts_dir == tmp_path / "runs" / bad.stem
    assert summary["needs_agent"] is True
    assert summary["status"] == "partially_loaded"
    assert summary["counts"] == {
        "rows_total": 40,
        "rows_loaded": 33,
        "rows_quarantined": 7,
        "rows_invalid": 7,
    }
    assert summary["contract"] == {"name": "orders", "version": "1.0.0"}
    assert len(summary["source_sha256"]) == 64
    assert violations["invalid_rows"] == 7 and violations["violations"]


def test_source_hash_changes_with_content(contract, adapter, samples, tmp_path):
    good, bad = samples
    a = run_batch(good, contract, adapter, tmp_path / "runs")
    b = run_batch(bad, contract, adapter, tmp_path / "runs")
    assert a.source_sha256 != b.source_sha256


def test_dry_run_writes_artifacts_but_touches_no_warehouse(contract, samples, tmp_path):
    _, bad = samples
    result = run_batch(bad, contract, None, tmp_path / "runs", dry_run=True)
    assert result.dry_run and result.warehouse is None and result.rows_loaded == 0
    assert result.needs_agent
    assert (result.artifacts_dir / "violations.json").exists()


def test_adapter_is_required_unless_dry_run(contract, samples, tmp_path):
    good, _ = samples
    with pytest.raises(ValueError, match="adapter is required"):
        run_batch(good, contract, None, tmp_path / "runs")


def test_sample_files_have_the_documented_names(samples):
    good, bad = samples
    assert good.name == GOOD_NAME and bad.name == BAD_NAME
    assert len(pd.read_csv(good)) == 40
