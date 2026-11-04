-- fact_order_lines: the atomic fact table.
--
-- Grain: one row per cleaned silver line item.
--
-- Every other gold model derives from this one, so revenue reconciles by construction
-- rather than by coincidence -- there is exactly one place a line's revenue is stated.
--
-- line_key is a surrogate key. It is reproducible across runs because silver drops exact
-- full-row duplicates, which makes the eight business columns jointly unique; ordering by
-- all of them is therefore a total order, not an arbitrary tie-break.

CREATE OR REPLACE TABLE fact_order_lines AS
SELECT
    row_number() OVER (
        ORDER BY
            invoice_no,
            stock_code,
            invoice_date,
            quantity,
            unit_price,
            description,
            country,
            customer_id
    )                                                  AS line_key,
    invoice_no,
    stock_code,
    description,
    customer_id,
    invoice_date,
    CAST(strftime(invoice_date, '%Y%m%d') AS INTEGER)  AS invoice_date_key,
    quantity,
    unit_price,
    revenue,
    is_cancellation,
    is_adjustment,
    is_product,
    is_zero_price,
    is_quantity_outlier,
    is_price_outlier,
    country,
    source_sheet,
    cleaned_at
FROM silver_online_retail_ii;
