"""Bronze layer: raw source files -> Parquet, unmodified except for ingestion metadata.

Bronze never cleans, filters, renames, or re-types business columns — that's the silver
layer's job. The only additions here are columns that record *where a row came from*
(so every downstream row can be traced back to a raw file and an ingestion run).

Usage:
    uv run python -m hub.ingestion.bronze --dataset online_retail_ii
    uv run python -m hub.ingestion.bronze --dataset olist
    uv run python -m hub.ingestion.bronze --dataset all
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
BRONZE_DIR = Path(__file__).resolve().parents[3] / "data" / "bronze"

ONLINE_RETAIL_II_SHEETS = ["Year 2009-2010", "Year 2010-2011"]

# Exact Kaggle filenames for olistbr/brazilian-ecommerce (without the .csv extension).
OLIST_TABLES = [
    "olist_customers_dataset",
    "olist_geolocation_dataset",
    "olist_order_items_dataset",
    "olist_order_payments_dataset",
    "olist_order_reviews_dataset",
    "olist_orders_dataset",
    "olist_products_dataset",
    "olist_sellers_dataset",
    "product_category_name_translation",
]


@dataclass
class BronzeLoadResult:
    table: str
    path: Path
    rows: int
    source_file: str
    ingested_at: str


def _add_ingestion_metadata(df: pd.DataFrame, source_file: str, ingested_at: str) -> pd.DataFrame:
    df = df.copy()
    df["_source_file"] = source_file
    df["_ingested_at"] = ingested_at
    return df


def load_online_retail_ii_to_bronze(
    raw_dir: Path = RAW_DIR, bronze_dir: Path = BRONZE_DIR
) -> BronzeLoadResult:
    """Read both sheets of the raw workbook, stack them, write one bronze Parquet file.

    Invoice/StockCode/Description are read as strings, not because bronze "cleans" the
    data, but because these columns are genuinely non-numeric in the source (Invoice
    carries a 'C' prefix for cancellations; StockCode includes non-product codes like
    'POST' and 'DOT'; a handful of raw Description cells contain stray numeric-only
    values that would otherwise break a single consistent column type) — reading them
    as anything else would silently coerce/lose values, which bronze must never do.
    """
    src = raw_dir / "online_retail_II.xlsx"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Run "
            "`uv run python -m hub.ingestion.download --dataset online_retail_ii` first."
        )

    ingested_at = datetime.now(UTC).isoformat()
    frames = []
    for sheet in ONLINE_RETAIL_II_SHEETS:
        sheet_df = pd.read_excel(
            src,
            sheet_name=sheet,
            dtype={"Invoice": str, "StockCode": str, "Description": str},
        )
        sheet_df["_source_sheet"] = sheet
        frames.append(sheet_df)
    combined = pd.concat(frames, ignore_index=True)
    combined = _add_ingestion_metadata(combined, src.name, ingested_at)

    bronze_dir.mkdir(parents=True, exist_ok=True)
    out_path = bronze_dir / "online_retail_ii.parquet"
    combined.to_parquet(out_path, index=False)

    return BronzeLoadResult(
        table="online_retail_ii",
        path=out_path,
        rows=len(combined),
        source_file=src.name,
        ingested_at=ingested_at,
    )


def load_olist_to_bronze(
    raw_dir: Path = RAW_DIR, bronze_dir: Path = BRONZE_DIR
) -> list[BronzeLoadResult]:
    """Read each raw Olist CSV as-is and write it to its own bronze Parquet table.

    Tolerant of a partial download (missing tables are skipped, not errored) so this
    can be re-run incrementally as more Olist files become available.
    """
    olist_raw = raw_dir / "olist"
    if not olist_raw.exists():
        raise FileNotFoundError(
            f"{olist_raw} not found. Run "
            "`uv run python -m hub.ingestion.download --dataset olist` first "
            "(requires Kaggle auth)."
        )

    ingested_at = datetime.now(UTC).isoformat()
    out_dir = bronze_dir / "olist"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for table in OLIST_TABLES:
        csv_path = olist_raw / f"{table}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df = _add_ingestion_metadata(df, csv_path.name, ingested_at)
        out_path = out_dir / f"{table}.parquet"
        df.to_parquet(out_path, index=False)
        results.append(
            BronzeLoadResult(
                table=table,
                path=out_path,
                rows=len(df),
                source_file=csv_path.name,
                ingested_at=ingested_at,
            )
        )
    return results


DATASETS = {
    "online_retail_ii": load_online_retail_ii_to_bronze,
    "olist": load_olist_to_bronze,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=[*DATASETS.keys(), "all"], default="all", help="Which dataset to load"
    )
    args = parser.parse_args()

    targets = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    for name in targets:
        print(f"[{name}] loading to bronze...")
        result = DATASETS[name]()
        results = result if isinstance(result, list) else [result]
        for r in results:
            print(f"[{name}] {r.table}: {r.rows:,} rows -> {r.path}")


if __name__ == "__main__":
    main()
