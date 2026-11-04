"""Silver layer: clean, type, and flag the bronze Online Retail II table.

Cleaning is done in pandas here rather than SQL/DuckDB — the roadmap allows either;
this repo does row-level cleaning in pandas (easy to unit test step-by-step) and
reserves SQL/DuckDB for the gold layer's fact/dim modeling, where set-based joins and
aggregation actually pay off.

Documented cleaning decisions (row-count impact printed by `main()`):

1. Exact full-row duplicates (~3.2% of rows) are dropped — known data-entry
   duplication in this dataset, not real repeated line items.
2. Invoice-prefix-marked negative-value events are KEPT and flagged, never dropped, so
   that revenue totals stay true while customer/product analysis can exclude them:
   - `is_cancellation` — Invoice starting 'C' (19,104 rows, 8,292 invoices,
     -£1,462,050.61). They carry negative quantity/revenue and net out real returns;
     dropping them would overstate revenue.
   - `is_adjustment` — Invoice starting 'A' (6 rows, -£147,614). These are bad-debt
     write-offs, not merchandise movements: every one has Description "Adjust bad
     debt" and stock code 'B'. Left unflagged they are indistinguishable from returns
     and would silently corrupt customer-level revenue.
   With both prefixes excluded, all 45,330 remaining orders have non-negative net
   revenue — asserted in tests.
3. Administrative/non-merchandise stock codes (postage, bank charges, discounts,
   manual adjustments, gift vouchers, test rows) are KEPT (they're real cash
   movements) but flagged `is_product = False` so product-catalog analysis can
   exclude them while revenue totals still include them. Classification was checked
   against each code's Description text, not guessed from the code alone — e.g.
   'DCGS0058'/'PADS'/'SP1002' look non-standard but are real products ("MISO PRETTY
   GUM", "PADS TO MATCH ALL CUSHIONS", "KID'S CHALKBOARD/EASEL"), while 'POST'/'D'/'M'
   etc. are confirmed fees/discounts/manual entries.
4. Missing Customer ID (~22.8%) is kept as null — excluded from customer-level models
   downstream, retained for order/revenue totals.
5. Zero-price rows are kept, flagged `is_zero_price` (samples/giveaways/promo lines,
   not errors).
6. No blanket winsorizing of Quantity/Price: this retailer explicitly serves
   wholesalers, so large quantities are real business signal, not data errors. Rows
   are instead flagged `is_quantity_outlier` / `is_price_outlier` at fixed thresholds
   (>200 units / >£25 — the approximate 99.5th percentile of the full 2009-2011
   history) so scale-sensitive models (e.g. clustering) can decide whether to cap
   them, without silver silently destroying signal.
7. Timestamps are kept naive (no timezone in the source; assumed UK local time, not
   converted). All prices are GBP (single-currency dataset); no conversion applied.
8. String columns are whitespace-trimmed; values otherwise kept verbatim (e.g.
   Country == "Unspecified" is a valid raw value, not treated as missing).
9. Negative quantities on non-'C' invoices (3,393 rows) are NOT returns: every one has
   unit_price == 0 and a correction-style Description ("damages", "lost", "short",
   "mixed", or blank). They are inventory write-downs with exactly £0 revenue impact,
   so they are kept as-is and flagged only by the existing `is_zero_price`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from hub.validation.schemas import silver_online_retail_ii_schema

BRONZE_DIR = Path(__file__).resolve().parents[3] / "data" / "bronze"
SILVER_DIR = Path(__file__).resolve().parents[3] / "data" / "silver"

# Evidence-checked against each code's Description text in the 2009-2011 history (see
# module docstring point 3) — every entry here is a fee, adjustment, discount, manual
# correction, or test row, never real merchandise. Matched case-insensitively.
NON_PRODUCT_STOCK_CODES = {
    "ADJUST",
    "ADJUST2",
    "AMAZONFEE",
    "B",
    "BANK CHARGES",
    "C2",
    "C3",
    "CRUK",
    "D",
    "DOT",
    "M",
    "POST",
    "S",
    "TEST001",
    "TEST002",
    "GIFT",
}

QUANTITY_OUTLIER_THRESHOLD = 200  # ~99.5th percentile of |Quantity|, 2009-2011 history
PRICE_OUTLIER_THRESHOLD = 25.0  # ~99.5th percentile of Price, 2009-2011 history

COLUMN_RENAME = {
    "Invoice": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
    "_source_sheet": "source_sheet",
}


@dataclass
class SilverCleanResult:
    table: str
    path: Path
    rows_in: int
    rows_out: int
    duplicates_dropped: int
    cleaned_at: str


def _is_gift_voucher(stock_code: pd.Series) -> pd.Series:
    return stock_code.str.match(r"^gift_0001", case=False, na=False)


def clean_online_retail_ii(
    bronze_dir: Path = BRONZE_DIR, silver_dir: Path = SILVER_DIR
) -> SilverCleanResult:
    src = bronze_dir / "online_retail_ii.parquet"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Run "
            "`uv run python -m hub.ingestion.bronze --dataset online_retail_ii` first."
        )

    df = pd.read_parquet(src)
    rows_in = len(df)

    for col in ["Invoice", "StockCode", "Description", "Country"]:
        df[col] = df[col].str.strip()

    business_cols = [c for c in df.columns if not c.startswith("_")]
    duplicates_mask = df.duplicated(subset=business_cols, keep="first")
    duplicates_dropped = int(duplicates_mask.sum())
    df = df.loc[~duplicates_mask].copy()

    df = df.rename(columns=COLUMN_RENAME)
    df = df[list(COLUMN_RENAME.values())]

    df["customer_id"] = df["customer_id"].astype("Int64")
    df["is_cancellation"] = df["invoice_no"].str.startswith("C")
    df["is_adjustment"] = df["invoice_no"].str.startswith("A")
    df["is_product"] = ~(
        df["stock_code"].str.upper().isin(NON_PRODUCT_STOCK_CODES)
        | _is_gift_voucher(df["stock_code"])
    )
    df["is_zero_price"] = df["unit_price"] == 0
    df["is_quantity_outlier"] = df["quantity"].abs() > QUANTITY_OUTLIER_THRESHOLD
    df["is_price_outlier"] = df["unit_price"] > PRICE_OUTLIER_THRESHOLD
    df["revenue"] = df["quantity"] * df["unit_price"]
    df["cleaned_at"] = datetime.now(UTC).isoformat()

    df = silver_online_retail_ii_schema.validate(df, lazy=True)

    silver_dir.mkdir(parents=True, exist_ok=True)
    out_path = silver_dir / "online_retail_ii.parquet"
    df.to_parquet(out_path, index=False)

    return SilverCleanResult(
        table="online_retail_ii",
        path=out_path,
        rows_in=rows_in,
        rows_out=len(df),
        duplicates_dropped=duplicates_dropped,
        cleaned_at=df["cleaned_at"].iloc[0],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = clean_online_retail_ii()
    print(
        f"[online_retail_ii] {result.rows_in:,} -> {result.rows_out:,} rows "
        f"({result.duplicates_dropped:,} exact duplicates dropped) -> {result.path}"
    )


if __name__ == "__main__":
    main()
