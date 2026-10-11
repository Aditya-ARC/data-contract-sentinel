from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from contractsentinel.contract import Contract, load_contract, parse_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts" / "orders.yaml"

BASE_ROW = {
    "order_id": "O100001",
    "customer_id": "C12345",
    "order_date": "2026-10-01",
    "amount": "10.50",
    "status": "pending",
    "region": "NA",
}


@pytest.fixture(scope="session")
def contract() -> Contract:
    return load_contract(CONTRACT_PATH)


@pytest.fixture()
def raw_contract() -> dict:
    """A fresh, mutable dict of the real contract file."""
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def build_contract(raw_contract):
    def _build(mutate=None) -> Contract:
        if mutate:
            mutate(raw_contract)
        return parse_contract(raw_contract)

    return _build


@pytest.fixture()
def make_df():
    """make_df({}, {"customer_id": ""}) -> frame with unique order ids and overrides per row."""

    def _make(*overrides: dict) -> pd.DataFrame:
        rows = []
        for i, override in enumerate(overrides or ({},)):
            row = dict(BASE_ROW)
            row["order_id"] = f"O{100001 + i}"
            row.update(override)
            rows.append(row)
        return pd.DataFrame(rows)

    return _make
