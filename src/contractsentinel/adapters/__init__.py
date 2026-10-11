"""Warehouse adapters."""

from __future__ import annotations

from contractsentinel.adapters.base import QuarantineRecord, WarehouseAdapter

WAREHOUSES = ("duckdb", "snowflake")


def get_adapter(warehouse: str = "duckdb", database: str = "data/warehouse.duckdb") -> WarehouseAdapter:
    """Build an adapter by name. Imports are lazy so optional extras stay optional."""
    if warehouse == "duckdb":
        from contractsentinel.adapters.duckdb_adapter import DuckDBAdapter

        return DuckDBAdapter(database)
    if warehouse == "snowflake":
        from contractsentinel.adapters.snowflake_adapter import SnowflakeAdapter

        return SnowflakeAdapter()
    raise ValueError(f"Unknown warehouse {warehouse!r}. Choose one of: {', '.join(WAREHOUSES)}")


__all__ = ["QuarantineRecord", "WarehouseAdapter", "WAREHOUSES", "get_adapter"]
