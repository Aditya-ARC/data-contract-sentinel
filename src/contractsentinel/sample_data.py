"""Generate a clean sample batch and a broken one for demos and tests."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

STATUSES = ["pending", "shipped", "delivered", "cancelled", "refunded"]
REGIONS = ["NA", "EMEA", "APAC", "LATAM"]

GOOD_NAME = "orders_2026-10-01.csv"
BAD_NAME = "orders_2026-10-02.csv"


def generate_orders(rows: int, order_date: str, seed: int = 7, id_start: int = 100001) -> pd.DataFrame:
    rng = random.Random(seed)
    data = {
        "order_id": [f"O{id_start + i}" for i in range(rows)],
        "customer_id": [f"C{rng.randint(10000, 99999)}" for _ in range(rows)],
        "order_date": [order_date] * rows,
        "amount": [f"{rng.uniform(5, 900):.2f}" for _ in range(rows)],
        "status": [rng.choice(STATUSES) for _ in range(rows)],
        "region": [rng.choice(REGIONS) for _ in range(rows)],
    }
    return pd.DataFrame(data)


def inject_defects(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Return a copy with realistic defects on seven distinct rows."""
    if len(df) < 20:
        raise ValueError("need at least 20 rows to inject the sample defects")
    rng = random.Random(seed + 1)
    # Start at 1 so row 0 can serve as the original for the duplicate below.
    rows = rng.sample(range(1, len(df)), 7)
    out = df.copy()

    # Two rows with empty customer ids.
    out.loc[rows[0], "customer_id"] = ""
    out.loc[rows[1], "customer_id"] = ""
    # A date in the wrong format (US style instead of ISO).
    out.loc[rows[2], "order_date"] = "10/02/2026"
    # Text where a number belongs.
    out.loc[rows[3], "amount"] = "twelve dollars"
    # A negative amount.
    out.loc[rows[4], "amount"] = "-45.00"
    # A status typo and an unknown region on the same row.
    out.loc[rows[5], "status"] = "shiped"
    out.loc[rows[5], "region"] = "EU"
    # A duplicate order id: this row reuses the id of row 0, which comes first.
    out.loc[rows[6], "order_id"] = out.loc[0, "order_id"]
    return out


def write_samples(out_dir: str | Path, rows: int = 40, seed: int = 7) -> tuple[Path, Path]:
    """Write a clean batch and a defective batch into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    good = generate_orders(rows, "2026-10-01", seed=seed, id_start=100001)
    bad = inject_defects(
        generate_orders(rows, "2026-10-02", seed=seed + 100, id_start=200001), seed=seed
    )

    good_path = out_dir / GOOD_NAME
    bad_path = out_dir / BAD_NAME
    good.to_csv(good_path, index=False)
    bad.to_csv(bad_path, index=False)
    return good_path, bad_path


SLASH_NAME = "orders_2026-10-03.csv"


def write_slash_date_sample(out_dir: str | Path, rows: int = 40, seed: int = 7) -> Path:
    """A fully valid batch whose vendor writes dates as 2026/10/03.

    The pipeline normalizes these to ISO dates, so this batch passes. It is used for the
    code-regression demo: if date normalization breaks, every row fails the date rule.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_orders(rows, "2026/10/03", seed=seed + 200, id_start=300001)
    path = out_dir / SLASH_NAME
    df.to_csv(path, index=False)
    return path
