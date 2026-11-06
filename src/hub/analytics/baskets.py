"""Market-basket association rules, computed in DuckDB over the atomic fact table.

Baskets are invoices, items are stock codes, and only ordinary product lines count.
The pairwise self-join is genuinely large (~44M candidate pairs on the real dataset), so
the work is filtered by item support first and only then grouped; DuckDB does it in
about half a minute with no infrastructure.

Both directions of every pair are emitted. Confidence is asymmetric, so reporting only
A -> B would hide that B -> A is often the stronger rule for merchandising.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


@dataclass(frozen=True)
class BasketSummary:
    baskets: int
    items: int
    eligible_items: int
    avg_items: float
    median_items: float
    max_items: int


@dataclass(frozen=True)
class BasketResult:
    summary: BasketSummary
    rules: pd.DataFrame
    min_item_support: float
    min_pair_support: float
    min_confidence: float


def analyse_baskets(
    lines_path: Path,
    *,
    min_item_support: float = 0.001,
    min_pair_support: float = 0.002,
    min_confidence: float = 0.05,
    top_n: int | None = None,
) -> BasketResult:
    for name, value in {
        "min_item_support": min_item_support,
        "min_pair_support": min_pair_support,
        "min_confidence": min_confidence,
    }.items():
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {value}")

    escaped = lines_path.as_posix().replace("'", "''")
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE basket_items AS
            SELECT DISTINCT invoice_no, stock_code
            FROM read_parquet('{escaped}')
            WHERE is_product AND NOT is_cancellation AND NOT is_adjustment
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE basket_totals AS
            SELECT COUNT(DISTINCT invoice_no) AS n_baskets,
                   COUNT(DISTINCT stock_code) AS n_items
            FROM basket_items
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE item_stats AS
            SELECT stock_code, COUNT(*) AS basket_count
            FROM basket_items
            GROUP BY 1
            """
        )
        summary = _summary(con, min_item_support)
        rules = _rules(con, min_item_support, min_pair_support, min_confidence, top_n)
    finally:
        con.close()

    return BasketResult(
        summary=summary,
        rules=rules,
        min_item_support=min_item_support,
        min_pair_support=min_pair_support,
        min_confidence=min_confidence,
    )


def _summary(con: duckdb.DuckDBPyConnection, min_item_support: float) -> BasketSummary:
    row = con.execute(
        f"""
        WITH baskets AS (
            SELECT invoice_no, COUNT(*) AS items FROM basket_items GROUP BY 1
        )
        SELECT
            (SELECT n_baskets FROM basket_totals) AS n_baskets,
            (SELECT n_items   FROM basket_totals) AS n_items,
            (SELECT COUNT(*) FROM item_stats s, basket_totals t
              WHERE s.basket_count >= {float(min_item_support):.10f} * t.n_baskets)
                AS eligible_items,
            (SELECT AVG(items)    FROM baskets) AS avg_items,
            (SELECT median(items) FROM baskets) AS median_items,
            (SELECT MAX(items)    FROM baskets) AS max_items
        """
    ).fetchone()
    return BasketSummary(
        baskets=int(row[0]),
        items=int(row[1]),
        eligible_items=int(row[2]),
        avg_items=float(row[3]),
        median_items=float(row[4]),
        max_items=int(row[5]),
    )


def _rules(
    con: duckdb.DuckDBPyConnection,
    min_item_support: float,
    min_pair_support: float,
    min_confidence: float,
    top_n: int | None,
) -> pd.DataFrame:
    limit = f"LIMIT {int(top_n)}" if top_n is not None else ""
    frame = con.execute(
        f"""
        WITH eligible AS (
            SELECT b.invoice_no, b.stock_code
            FROM basket_items b
            JOIN item_stats s USING (stock_code)
            CROSS JOIN basket_totals t
            WHERE s.basket_count >= {float(min_item_support):.10f} * t.n_baskets
        ),
        pairs AS (
            SELECT a.stock_code AS antecedent, b.stock_code AS consequent,
                   COUNT(*) AS pair_baskets
            FROM eligible a
            JOIN eligible b
              ON a.invoice_no = b.invoice_no AND a.stock_code < b.stock_code
            GROUP BY 1, 2
        ),
        directed AS (
            SELECT antecedent, consequent, pair_baskets FROM pairs
            UNION ALL
            SELECT consequent AS antecedent, antecedent AS consequent, pair_baskets FROM pairs
        )
        SELECT d.antecedent,
               d.consequent,
               d.pair_baskets,
               d.pair_baskets::DOUBLE / t.n_baskets           AS support_pair,
               sa.basket_count::DOUBLE / t.n_baskets          AS support_antecedent,
               sb.basket_count::DOUBLE / t.n_baskets          AS support_consequent,
               d.pair_baskets::DOUBLE / sa.basket_count       AS confidence,
               (d.pair_baskets::DOUBLE / sa.basket_count)
                   / (sb.basket_count::DOUBLE / t.n_baskets)  AS lift
        FROM directed d
        JOIN item_stats sa ON sa.stock_code = d.antecedent
        JOIN item_stats sb ON sb.stock_code = d.consequent
        CROSS JOIN basket_totals t
        WHERE d.pair_baskets >= {float(min_pair_support):.10f} * t.n_baskets
          AND d.pair_baskets::DOUBLE / sa.basket_count >= {float(min_confidence):.10f}
        ORDER BY lift DESC, support_pair DESC
        {limit}
        """
    ).df()
    return frame
