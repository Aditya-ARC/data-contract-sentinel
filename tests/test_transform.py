from __future__ import annotations

import datetime as dt

import pandas as pd

from contractsentinel.transform import normalize, to_typed
from contractsentinel.validator import validate


def test_normalize_trims_and_fixes_case(make_df):
    raw = make_df({"status": " Shipped ", "region": "emea"})
    out = normalize(raw)
    assert out.loc[0, "status"] == "shipped"
    assert out.loc[0, "region"] == "EMEA"


def test_normalize_does_not_touch_other_date_formats(make_df):
    out = normalize(make_df({"order_date": "10/02/2026"}))
    assert out.loc[0, "order_date"] == "10/02/2026"


def test_normalize_keeps_the_input_unchanged(make_df):
    raw = make_df({"status": "SHIPPED"})
    normalize(raw)
    assert raw.loc[0, "status"] == "SHIPPED"


def test_normalized_batch_then_validates(contract, make_df):
    raw = make_df({"status": "DELIVERED", "region": "apac", "order_date": "2026-10-04"})
    assert validate(normalize(raw), contract, "b1").passed


def test_to_typed_casts_to_destination_types(contract, make_df):
    typed = to_typed(normalize(make_df({"amount": "12.30"}, {"amount": "7"})), contract)
    assert list(typed.columns) == list(contract.field_names)
    assert typed.loc[0, "order_date"] == dt.date(2026, 10, 1)
    assert str(typed["amount"].dtype) == "Float64"
    assert typed["amount"].tolist() == [12.3, 7.0]
    assert typed.loc[0, "customer_id"] == "C12345"


def test_to_typed_drops_non_contract_columns_and_keeps_nulls(build_contract, make_df):
    def make_region_nullable(raw):
        raw["fields"][5]["nullable"] = True

    contract = build_contract(make_region_nullable)
    df = make_df({"region": ""})
    df["coupon"] = "X"
    typed = to_typed(df, contract)
    assert "coupon" not in typed.columns
    assert typed.loc[0, "region"] is None


def test_to_typed_on_empty_frame(contract, make_df):
    typed = to_typed(make_df({}).iloc[0:0], contract)
    assert len(typed) == 0 and list(typed.columns) == list(contract.field_names)
    assert isinstance(typed, pd.DataFrame)
