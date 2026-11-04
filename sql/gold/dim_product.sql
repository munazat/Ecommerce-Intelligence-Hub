-- dim_product: one row per StockCode.
--
-- Non-merchandise codes (POST, BANK CHARGES, DOT, B, ...) are deliberately included so
-- that postage and fee revenue stays traceable to a code; is_product distinguishes them.
--
-- Description is NOT 1:1 with StockCode in this dataset. The same code appears as
-- "85123a mixed", "damages", "short", blank, and so on, because staff overwrote the
-- description field to annotate corrections. The most frequent non-null description
-- wins, with the description text itself as a deterministic tie-break so rebuilds are
-- reproducible.

CREATE OR REPLACE TABLE dim_product AS
WITH description_rank AS (
    SELECT
        stock_code,
        description,
        row_number() OVER (
            PARTITION BY stock_code
            ORDER BY COUNT(*) DESC, description
        ) AS rn
    FROM fact_order_lines
    WHERE description IS NOT NULL
    GROUP BY stock_code, description
)
SELECT
    l.stock_code,
    d.description                                         AS description,
    bool_or(l.is_product)                                 AS is_product,
    MIN(l.invoice_date)                                   AS first_seen,
    MAX(l.invoice_date)                                   AS last_seen,
    COUNT(*)                                              AS line_count,
    COUNT(DISTINCT l.invoice_no)                          AS invoice_count,
    SUM(l.quantity)                                       AS units,
    SUM(l.revenue)                                        AS net_revenue,
    SUM(CASE WHEN NOT l.is_cancellation AND NOT l.is_adjustment
             THEN l.revenue ELSE 0 END)                   AS gross_revenue,
    median(l.unit_price)                                  AS median_unit_price,
    SUM(CASE WHEN l.is_cancellation THEN 1 ELSE 0 END)    AS cancellation_line_count,
    CAST(SUM(CASE WHEN l.is_cancellation THEN 1 ELSE 0 END) AS DOUBLE)
        / COUNT(*)                                        AS cancellation_rate
FROM fact_order_lines l
LEFT JOIN description_rank d
       ON d.stock_code = l.stock_code
      AND d.rn = 1
GROUP BY l.stock_code, d.description;
