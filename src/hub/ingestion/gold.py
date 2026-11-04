"""Gold layer: silver -> dimensional warehouse, modelled in SQL and run by DuckDB.

Silver does row-level cleaning in pandas, where each rule is easy to unit test in
isolation. Gold does set-based modelling in SQL, which is where DuckDB earns its keep:
joins and aggregation over ~1M rows with no infrastructure and nothing to babysit.

Each model lives in its own file under sql/gold/ so the transformation reads as SQL and
reviews as a diff, rather than being buried in a Python string literal.

Usage:
    uv run python -m hub.ingestion.gold
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import duckdb

from hub.validation.gold_schemas import GOLD_SCHEMAS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
SQL_DIR = PROJECT_ROOT / "sql" / "gold"

SILVER_VIEW = "silver_online_retail_ii"
SILVER_TABLE = "online_retail_ii"

# Dependency order: fact_order_lines is the atomic fact table every other model reads
# from, so revenue reconciles by construction instead of by coincidence.
GOLD_MODELS = [
    "fact_order_lines",
    "dim_date",
    "fact_orders",
    "dim_customer",
    "dim_product",
]


@dataclass
class GoldBuildResult:
    table: str
    path: Path
    rows: int


def register_parquet_view(con: duckdb.DuckDBPyConnection, view: str, path: Path) -> None:
    # The path is quoted, not interpolated raw: real dataset paths contain spaces.
    escaped = path.as_posix().replace("'", "''")
    con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{escaped}')")


def build_gold(
    silver_dir: Path = SILVER_DIR, gold_dir: Path = GOLD_DIR, sql_dir: Path = SQL_DIR
) -> list[GoldBuildResult]:
    src = silver_dir / f"{SILVER_TABLE}.parquet"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Run `uv run python -m hub.ingestion.silver` first."
        )

    gold_dir.mkdir(parents=True, exist_ok=True)
    results = []
    con = duckdb.connect()
    try:
        register_parquet_view(con, SILVER_VIEW, src)
        for model in GOLD_MODELS:
            con.execute((sql_dir / f"{model}.sql").read_text(encoding="utf-8"))
            df = con.execute(f"SELECT * FROM {model}").df()
            df = GOLD_SCHEMAS[model].validate(df, lazy=True)
            out_path = gold_dir / f"{model}.parquet"
            df.to_parquet(out_path, index=False)
            results.append(GoldBuildResult(table=model, path=out_path, rows=len(df)))
    finally:
        con.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for result in build_gold():
        print(f"[gold] {result.table}: {result.rows:,} rows -> {result.path}")


if __name__ == "__main__":
    main()
