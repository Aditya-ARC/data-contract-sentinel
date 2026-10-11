from __future__ import annotations

import pandas as pd
import pytest

from contractsentinel.validator import validate


def rules(result):
    return {v.rule_id: v for v in result.violations}


def test_clean_batch_passes(contract, make_df):
    result = validate(make_df({}, {}, {}), contract, "b1")
    assert result.passed
    assert result.violations == []
    assert result.invalid_row_count == 0


def test_null_customer_id_is_reported_with_row_number(contract, make_df):
    result = validate(make_df({}, {"customer_id": ""}, {"customer_id": "   "}), contract, "b1")
    v = rules(result)["orders.customer_id.not_null"]
    assert v.count == 2
    assert [s["row_number"] for s in v.samples] == [3, 4]  # header is line 1
    assert "orders.customer_id.pattern" not in rules(result)  # nulls are not also pattern errors
    assert result.invalid_index() == [1, 2]


@pytest.mark.parametrize("bad", ["10/02/2026", "2026-1-5", "2026-13-40", "not a date", "20261001"])
def test_bad_date_formats_are_rejected(contract, make_df, bad):
    result = validate(make_df({"order_date": bad}), contract, "b1")
    assert "orders.order_date.format" in rules(result)


def test_empty_date_is_a_null_not_a_format_error(contract, make_df):
    result = validate(make_df({"order_date": ""}), contract, "b1")
    assert set(rules(result)) == {"orders.order_date.not_null"}


@pytest.mark.parametrize("bad", ["twelve dollars", "NaN", "inf", "1,234.50", "$5"])
def test_non_numeric_amounts_are_type_errors(contract, make_df, bad):
    result = validate(make_df({"amount": bad}), contract, "b1")
    assert "orders.amount.type" in rules(result)


def test_negative_amount_violates_min(contract, make_df):
    result = validate(make_df({"amount": "-45.00"}), contract, "b1")
    assert set(rules(result)) == {"orders.amount.min"}


def test_min_boundary_is_exact(contract, make_df):
    assert "orders.amount.min" in rules(validate(make_df({"amount": "-0.01"}), contract, "b"))
    assert validate(make_df({"amount": "0.00"}), contract, "b").passed


def test_max_boundary_is_exact(build_contract, make_df):
    def cap_amount(raw):
        raw["fields"][3]["max"] = 100

    capped = build_contract(cap_amount)
    assert validate(make_df({"amount": "100.00"}), capped, "b").passed
    result = validate(make_df({"amount": "100.01"}), capped, "b")
    assert set(rules(result)) == {"orders.amount.max"}


def test_integer_fields_reject_fractions(build_contract, make_df):
    def amount_as_integer(raw):
        raw["fields"][3] = {"name": "amount", "type": "integer", "nullable": False}

    as_int = build_contract(amount_as_integer)
    assert validate(make_df({"amount": "12"}), as_int, "b").passed
    assert "orders.amount.type" in rules(validate(make_df({"amount": "12.5"}), as_int, "b"))


def test_too_many_decimals_is_a_scale_error(contract, make_df):
    result = validate(make_df({"amount": "10.999"}), contract, "b1")
    assert set(rules(result)) == {"orders.amount.scale"}


@pytest.mark.parametrize("ok", ["10", "10.5", "0", "0.00", "999999.99"])
def test_valid_amounts_pass(contract, make_df, ok):
    assert validate(make_df({"amount": ok}), contract, "b1").passed


def test_unknown_status_and_region(contract, make_df):
    result = validate(make_df({"status": "shiped", "region": "EU"}), contract, "b1")
    assert set(rules(result)) == {
        "orders.status.allowed_values",
        "orders.region.allowed_values",
    }
    assert sorted(result.row_failures[0]) == [
        "orders.region.allowed_values",
        "orders.status.allowed_values",
    ]


def test_pattern_violation(contract, make_df):
    result = validate(make_df({"customer_id": "12345"}), contract, "b1")
    assert set(rules(result)) == {"orders.customer_id.pattern"}


def test_duplicates_flag_only_later_occurrences(contract, make_df):
    df = make_df({}, {}, {})
    df.loc[2, "order_id"] = df.loc[0, "order_id"]
    result = validate(df, contract, "b1")
    assert rules(result)["orders.order_id.unique"].count == 1
    assert result.invalid_index() == [2]


def test_missing_required_column_is_a_batch_level_error(contract, make_df):
    df = make_df({}).drop(columns=["customer_id"])
    result = validate(df, contract, "b1")
    v = rules(result)["orders.customer_id.missing_column"]
    assert v.scope == "batch" and v.severity == "error"
    assert not result.passed and result.has_batch_level_error


def test_unexpected_column_is_only_a_warning(contract, make_df):
    df = make_df({})
    df["coupon"] = "X"
    result = validate(df, contract, "b1")
    v = rules(result)["orders.coupon.unexpected_column"]
    assert v.severity == "warning"
    assert result.passed and result.warning_violations == [v]


def test_empty_batch_is_an_error(contract, make_df):
    result = validate(make_df({}).iloc[0:0].reset_index(drop=True), contract, "b1")
    assert not result.passed
    assert "orders.batch.empty_batch" in rules(result)


def test_samples_are_capped_and_values_clipped(contract, make_df):
    df = make_df(*[{"customer_id": ""} for _ in range(12)])
    result = validate(df, contract, "b1")
    v = rules(result)["orders.customer_id.not_null"]
    assert v.count == 12 and len(v.samples) == 5

    long = make_df({"status": "x" * 500})
    sample = rules(validate(long, contract, "b1"))["orders.status.allowed_values"].samples[0]
    assert len(sample["value"]) <= 120


def test_sample_shows_raw_value_when_it_differs_from_transformed(contract, make_df):
    raw = make_df({"status": " SHIPED "})
    transformed = raw.copy()
    transformed["status"] = "shiped"
    result = validate(transformed, contract, "b1", raw_df=raw)
    sample = rules(result)["orders.status.allowed_values"].samples[0]
    assert sample["value"] == "shiped" and sample["raw_value"] == " SHIPED "


def test_invalid_rate(contract, make_df):
    df = make_df({}, {"customer_id": ""}, {}, {})
    assert validate(df, contract, "b1").invalid_rate == 0.25


def test_requires_default_index(contract, make_df):
    df = make_df({}, {}).iloc[1:]
    with pytest.raises(ValueError, match="RangeIndex"):
        validate(df, contract, "b1")


def test_result_serializes_to_plain_dict(contract, make_df):
    result = validate(make_df({"customer_id": ""}), contract, "b1")
    data = result.to_dict()
    assert data["contract"] == {"name": "orders", "version": "1.0.0"}
    assert data["passed"] is False and data["invalid_rows"] == 1
    assert data["violations"][0]["rule_id"] == "orders.customer_id.not_null"
