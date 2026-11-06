"""Loaders for the gold layer, so every analysis reads the same files the same way.

Analysis code never reaches into `data/` with its own path logic: that is how one
notebook ends up reading a stale table while another reads the rebuilt one.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GOLD_DIR = PROJECT_ROOT / "data" / "gold"

GOLD_TABLES = (
    "fact_order_lines",
    "fact_orders",
    "dim_customer",
    "dim_product",
    "dim_date",
    "customer_snapshots",
)

# Table -> module that builds it, used only to make the failure message actionable.
_BUILDERS = {
    "customer_snapshots": "hub.features.snapshots",
}
_DEFAULT_BUILDER = "hub.ingestion.gold"


def load_table(name: str, gold_dir: Path = GOLD_DIR) -> pd.DataFrame:
    path = gold_dir / f"{name}.parquet"
    if not path.exists():
        builder = _BUILDERS.get(name, _DEFAULT_BUILDER)
        raise FileNotFoundError(f"{path} not found. Run `uv run python -m {builder}` first.")
    return pd.read_parquet(path)


def load_gold(
    gold_dir: Path = GOLD_DIR, tables: Iterable[str] = GOLD_TABLES
) -> dict[str, pd.DataFrame]:
    return {name: load_table(name, gold_dir) for name in tables}
