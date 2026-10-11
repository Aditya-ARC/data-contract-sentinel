from __future__ import annotations

from contractsentinel.cli import EXIT_ERROR, EXIT_OK, EXIT_VIOLATIONS, main
from conftest import CONTRACT_PATH


def base_args(tmp_path):
    return [
        "--contract",
        str(CONTRACT_PATH),
        "--db",
        str(tmp_path / "wh.duckdb"),
        "--artifacts",
        str(tmp_path / "runs"),
    ]


def make_samples(tmp_path):
    assert main(["make-sample", "--out", str(tmp_path / "landing")]) == EXIT_OK
    return tmp_path / "landing" / "orders_2026-10-01.csv", tmp_path / "landing" / "orders_2026-10-02.csv"


def test_make_sample_writes_both_files(tmp_path, capsys):
    good, bad = make_samples(tmp_path)
    assert good.exists() and bad.exists()
    assert "defective batch" in capsys.readouterr().out


def test_clean_batch_exits_zero(tmp_path, capsys):
    good, _ = make_samples(tmp_path)
    code = main(["run", str(good), *base_args(tmp_path)])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "status:     loaded" in out and "agent:" not in out


def test_defective_batch_exits_one_and_names_the_trigger(tmp_path, capsys):
    _, bad = make_samples(tmp_path)
    code = main(["run", str(bad), *base_args(tmp_path)])
    out = capsys.readouterr().out
    assert code == EXIT_VIOLATIONS
    assert "orders.customer_id.not_null" in out
    assert "investigation should be triggered" in out


def test_dry_run_does_not_create_a_database(tmp_path, capsys):
    _, bad = make_samples(tmp_path)
    code = main(["run", str(bad), "--dry-run", *base_args(tmp_path)])
    assert code == EXIT_VIOLATIONS
    assert not (tmp_path / "wh.duckdb").exists()
    assert "dry run, nothing was loaded" in capsys.readouterr().out


def test_missing_batch_file_is_a_tool_error(tmp_path, capsys):
    code = main(["run", str(tmp_path / "missing.csv"), *base_args(tmp_path)])
    assert code == EXIT_ERROR
    assert "not found" in capsys.readouterr().err


def test_bad_contract_path_is_a_tool_error(tmp_path, capsys):
    good, _ = make_samples(tmp_path)
    args = ["run", str(good), "--contract", str(tmp_path / "nope.yaml")]
    assert main(args) == EXIT_ERROR
    assert "contract file not found" in capsys.readouterr().err


def test_summarize_lists_every_run_and_its_violations(tmp_path, capsys):
    good, bad = make_samples(tmp_path)
    main(["run", str(good), *base_args(tmp_path)])
    main(["run", str(bad), *base_args(tmp_path)])
    capsys.readouterr()

    assert main(["summarize", "--runs", str(tmp_path / "runs")]) == EXIT_OK
    out = capsys.readouterr().out
    assert "| orders_2026-10-01 | loaded | 40 | 40 | 0 | no |" in out
    assert "| orders_2026-10-02 | partially_loaded | 40 | 33 | 7 | yes |" in out
    assert "`orders.customer_id.not_null` (error) x2" in out


def test_summarize_with_no_runs_says_so(tmp_path, capsys):
    assert main(["summarize", "--runs", str(tmp_path / "empty")]) == EXIT_OK
    assert "No pipeline runs found" in capsys.readouterr().out


def test_snowflake_without_credentials_is_a_clear_error(tmp_path, capsys, monkeypatch):
    for key in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE"):
        monkeypatch.delenv(key, raising=False)
    good, _ = make_samples(tmp_path)
    code = main(["run", str(good), "--warehouse", "snowflake", *base_args(tmp_path)])
    assert code == EXIT_ERROR
    assert "SNOWFLAKE" in capsys.readouterr().err
