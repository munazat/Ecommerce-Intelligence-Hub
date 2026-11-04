"""Point-in-time customer snapshots: the feature store that keeps models leak-free.

Built on top of the gold fact tables. sql/gold/customer_snapshots.sql states the invariant
this module exists to guarantee -- features see only the past, labels see only the future
-- and tests/test_snapshots.py proves it holds by mutating post-snapshot data and requiring
the feature columns to come back identical.

Usage:
    uv run python -m hub.features.snapshots
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import duckdb

from hub.ingestion.gold import register_parquet_view
from hub.validation.gold_schemas import customer_snapshots_schema

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
SQL_DIR = PROJECT_ROOT / "sql" / "gold"

FACT_TABLE = "fact_order_lines"
SNAPSHOT_TABLE = "customer_snapshots"


@dataclass
class SnapshotBuildResult:
    table: str
    path: Path
    rows: int
    snapshots: int
    customers: int
    complete_window_rows: int


def build_customer_snapshots(
    gold_dir: Path = GOLD_DIR, sql_dir: Path = SQL_DIR
) -> SnapshotBuildResult:
    src = gold_dir / f"{FACT_TABLE}.parquet"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Run `uv run python -m hub.ingestion.gold` first."
        )

    con = duckdb.connect()
    try:
        register_parquet_view(con, FACT_TABLE, src)
        con.execute((sql_dir / f"{SNAPSHOT_TABLE}.sql").read_text(encoding="utf-8"))
        df = con.execute(f"SELECT * FROM {SNAPSHOT_TABLE}").df()
    finally:
        con.close()

    df = customer_snapshots_schema.validate(df, lazy=True)
    out_path = gold_dir / f"{SNAPSHOT_TABLE}.parquet"
    df.to_parquet(out_path, index=False)

    return SnapshotBuildResult(
        table=SNAPSHOT_TABLE,
        path=out_path,
        rows=len(df),
        snapshots=int(df["snapshot_date"].nunique()),
        customers=int(df["customer_id"].nunique()),
        complete_window_rows=int(df["is_label_window_complete"].sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = build_customer_snapshots()
    print(
        f"[snapshots] {result.rows:,} rows across {result.snapshots} monthly snapshots "
        f"and {result.customers:,} customers "
        f"({result.complete_window_rows:,} rows with a complete label window) -> {result.path}"
    )


if __name__ == "__main__":
    main()
