"""Shared fixtures for the gold and snapshot tests.

Both need a small silver-shaped Parquet file whose values are easy to hand-compute
expected results from. Rows are specified only by the fields a test actually cares about;
everything else comes from SILVER_ROW_DEFAULTS. revenue is derived from quantity *
unit_price unless a test overrides it, so a fixture cannot accidentally contradict the
invariant the silver schema enforces.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

SILVER_ROW_DEFAULTS: dict[str, Any] = {
    "invoice_no": "500001",
    "stock_code": "85048",
    "description": "15CM CHRISTMAS GLASS BALL",
    "quantity": 1,
    "invoice_date": pd.Timestamp("2010-01-05 09:00:00"),
    "unit_price": 10.0,
    "customer_id": 1001,
    "country": "United Kingdom",
    "is_cancellation": False,
    "is_adjustment": False,
    "is_product": True,
    "is_zero_price": False,
    "is_quantity_outlier": False,
    "is_price_outlier": False,
    "source_sheet": "Year 2009-2010",
    "cleaned_at": "2026-01-01T00:00:00+00:00",
}

SILVER_COLUMNS = [*SILVER_ROW_DEFAULTS, "revenue"]

SILVER_FILENAME = "online_retail_ii.parquet"


@pytest.fixture
def write_silver() -> Callable[[Path, Sequence[dict[str, Any]]], pd.DataFrame]:
    """Write a silver-schema Parquet file from partial row dicts and return the frame."""

    def _write(silver_dir: Path, rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
        silver_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for row in rows:
            merged = {**SILVER_ROW_DEFAULTS, **row}
            merged.setdefault("revenue", merged["quantity"] * merged["unit_price"])
            records.append(merged)

        df = pd.DataFrame(records)[SILVER_COLUMNS]
        df["customer_id"] = df["customer_id"].astype("Int64")
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        df.to_parquet(silver_dir / SILVER_FILENAME, index=False)
        return df

    return _write
