"""Shared fixtures for the gold, snapshot and analytics tests.

The warehouse fixtures need a small silver-shaped Parquet file whose values are easy to
hand-compute expected results from. Rows are specified only by the fields a test actually
cares about; everything else comes from SILVER_ROW_DEFAULTS. revenue is derived from
quantity * unit_price unless a test overrides it, so a fixture cannot accidentally
contradict the invariant the silver schema enforces.

The analytics fixtures build minimal fact_orders / fact_order_lines frames directly.
Analytics functions are pure, so they can be tested without rebuilding the warehouse.
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

ORDER_ROW_DEFAULTS: dict[str, Any] = {
    "invoice_no": "500001",
    "invoice_date": pd.Timestamp("2010-01-05 09:00:00"),
    "customer_id": 1001,
    "is_cancellation": False,
    "is_adjustment": False,
    "product_count": 1,
    "net_revenue": 10.0,
    "gross_revenue": 10.0,
    "product_revenue": 10.0,
}

LINE_ROW_DEFAULTS: dict[str, Any] = {
    "stock_code": "85048",
    "is_product": True,
    "is_cancellation": False,
    "is_adjustment": False,
    "quantity": 1,
    "revenue": 10.0,
}


def _frame(rows: Sequence[dict[str, Any]], defaults: dict[str, Any]) -> pd.DataFrame:
    records = [{**defaults, **row} for row in rows]
    frame = pd.DataFrame(records, columns=list(defaults))
    if "invoice_date" in frame:
        frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    if "customer_id" in frame:
        frame["customer_id"] = frame["customer_id"].astype("Int64")
    return frame


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


@pytest.fixture
def make_orders() -> Callable[[Sequence[dict[str, Any]]], pd.DataFrame]:
    """Build a fact_orders-shaped frame from partial row dicts."""

    def _make(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
        return _frame(rows, ORDER_ROW_DEFAULTS)

    return _make


@pytest.fixture
def make_lines() -> Callable[[Sequence[dict[str, Any]]], pd.DataFrame]:
    """Build a fact_order_lines-shaped frame from partial row dicts."""

    def _make(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
        return _frame(rows, LINE_ROW_DEFAULTS)

    return _make
