-- dim_customer: one row per Customer ID.
--
-- Rows with a null Customer ID (22.8% of silver) are excluded by construction: they
-- cannot be attributed to a person. They stay in fact_order_lines and fact_orders, so
-- revenue totals still include them. That split is the documented silver policy for
-- missing customer ids, and it is why customer-level and order-level revenue must be
-- reconciled with an explicit null-customer term rather than expected to match.
--
-- country: 13 of 5,942 customers ship to more than one country (e.g. customer 12370
-- has 43 lines to Austria and 158 to Cyprus). arg_max takes the country of their most
-- recent order -- the current shipping country, which is the one a retention campaign
-- would actually act on. Picking MIN() instead would silently assign an arbitrary one.

CREATE OR REPLACE TABLE dim_customer AS
SELECT
    customer_id,
    arg_max(country, invoice_date)                        AS country,
    MIN(invoice_date)                                     AS first_invoice_date,
    MAX(invoice_date)                                     AS last_invoice_date,
    COUNT(DISTINCT invoice_no)                            AS invoice_count,
    COUNT(DISTINCT CASE WHEN NOT is_cancellation AND NOT is_adjustment
                        THEN invoice_no END)              AS purchase_count,
    COUNT(*)                                              AS line_count,
    COUNT(DISTINCT CASE WHEN is_product THEN stock_code END) AS distinct_products,
    SUM(quantity)                                         AS units,
    SUM(revenue)                                          AS net_revenue,
    SUM(CASE WHEN NOT is_cancellation AND NOT is_adjustment
             THEN revenue ELSE 0 END)                     AS gross_revenue,
    SUM(CASE WHEN is_cancellation THEN 1 ELSE 0 END)      AS cancellation_line_count,
    SUM(CASE WHEN is_adjustment THEN 1 ELSE 0 END)        AS adjustment_line_count
FROM fact_order_lines
WHERE customer_id IS NOT NULL
GROUP BY customer_id;
