from __future__ import annotations

import pytest

from contractsentinel.contract import ContractError, load_contract, parse_contract


def test_loads_the_orders_contract(contract):
    assert contract.name == "orders"
    assert contract.version == "1.0.0"
    assert contract.clean_table == "analytics.orders_clean"
    assert contract.quarantine_full_name == "analytics.orders_quarantine"
    assert contract.field_names == (
        "order_id",
        "customer_id",
        "order_date",
        "amount",
        "status",
        "region",
    )
    assert contract.policy.mode == "quarantine"
    assert contract.policy.max_violation_rate == 0.2


def test_field_details_are_parsed(contract):
    by_name = {f.name: f for f in contract.fields}
    assert by_name["customer_id"].nullable is False
    assert by_name["order_id"].unique is True
    assert by_name["order_date"].format == "%Y-%m-%d"
    assert by_name["amount"].scale == 2 and by_name["amount"].min == 0
    assert by_name["region"].allowed == ("NA", "EMEA", "APAC", "LATAM")


def test_missing_file_raises(tmp_path):
    with pytest.raises(ContractError, match="not found"):
        load_contract(tmp_path / "nope.yaml")


def test_invalid_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("fields: [unclosed", encoding="utf-8")
    with pytest.raises(ContractError, match="not valid YAML"):
        load_contract(bad)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r["fields"][0].update(type="uuid"), "unsupported type"),
        (lambda r: r["fields"][0].update(colour="red"), "unknown keys"),
        (lambda r: r["fields"][0].update(name="Order ID"), "not a valid identifier"),
        (lambda r: r["fields"][2].update(pattern="^x$"), "only applies to string"),
        (lambda r: r["fields"][0].update(pattern="(unclosed"), "invalid regex"),
        (lambda r: r["fields"][3].update(min=10, max=1), "min is greater than max"),
        (lambda r: r["fields"][1].update(min=0), "only apply to integer and decimal"),
        (lambda r: r["fields"][0].update(format="%Y"), "only applies to date"),
        (lambda r: r["fields"][0].update(scale=2), "only applies to decimal"),
        (lambda r: r["failure_policy"].update(mode="yolo"), "not one of"),
        (lambda r: r["failure_policy"].update(max_violation_rate=1.5), "between 0 and 1"),
        (lambda r: r["fields"].append(dict(r["fields"][0])), "duplicate field names"),
        (lambda r: r.pop("destination"), "missing required key 'destination'"),
        (lambda r: r.update(fields=[]), "non-empty list"),
    ],
)
def test_malformed_contracts_are_rejected(raw_contract, mutate, message):
    mutate(raw_contract)
    with pytest.raises(ContractError, match=message):
        parse_contract(raw_contract)


@pytest.mark.parametrize(
    "key", ["schema", "table", "quarantine_table"]
)
def test_destination_names_cannot_carry_sql(raw_contract, key):
    raw_contract["destination"][key] = "orders; DROP TABLE users"
    with pytest.raises(ContractError, match="not a valid identifier"):
        parse_contract(raw_contract)


def test_date_format_defaults_to_iso(raw_contract):
    del raw_contract["fields"][2]["format"]
    contract = parse_contract(raw_contract)
    assert contract.fields[2].format == "%Y-%m-%d"
