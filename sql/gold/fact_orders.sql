-- fact_orders: one row per invoice.
--
-- Grain: invoice_no. Verified against the data before choosing it, because the source
-- dataset has two quirks that would silently break a naive order grain:
--
--   * 83 invoices carry more than one distinct InvoiceDate. Every one of those spans
--     less than 9 minutes (max observed: 529368, 10:07 -> 10:16). These are a single
--     basket written line-by-line, not two orders reusing a number. invoice_no is
--     therefore a valid order grain, and invoice_date is taken as MIN() -- the moment
--     the basket started. Per-line timestamps are preserved in fact_order_lines.
--   * 0 invoices carry more than one Customer ID or Country, so collapsing those with
--     MIN() loses nothing.
--
-- gross_revenue excludes cancellations ('C') and bad-debt adjustments ('A'), because
-- those are reversals and write-offs, not sales. net_revenue is the true cash effect
-- and is what must tie back to fact_order_lines.

CREATE OR REPLACE TABLE fact_orders AS
SELECT
    invoice_no,
    MIN(invoice_date)                                     AS invoice_date,
    CAST(strftime(MIN(invoice_date), '%Y%m%d') AS INTEGER) AS invoice_date_key,
    MIN(customer_id)                                      AS customer_id,
    MIN(country)                                          AS country,
    bool_or(is_cancellation)                              AS is_cancellation,
    bool_or(is_adjustment)                                AS is_adjustment,
    COUNT(*)                                              AS line_count,
    COUNT(DISTINCT CASE WHEN is_product THEN stock_code END) AS product_count,
    SUM(quantity)                                         AS units,
    SUM(revenue)                                          AS net_revenue,
    SUM(CASE WHEN NOT is_cancellation AND NOT is_adjustment
             THEN revenue ELSE 0 END)                     AS gross_revenue,
    SUM(CASE WHEN is_product THEN revenue ELSE 0 END)     AS product_revenue,
    SUM(CASE WHEN NOT is_product THEN revenue ELSE 0 END) AS non_product_revenue
FROM fact_order_lines
GROUP BY invoice_no;
