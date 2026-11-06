-- customer_snapshots: the point-in-time feature store, and this project's leakage firewall.
--
-- Grain: one row per (customer_id, snapshot_date).
--
-- THE INVARIANT: every feature column is computed only from rows with
-- invoice_date STRICTLY BEFORE snapshot_date. Every label column is computed only from
-- rows with invoice_date ON OR AFTER snapshot_date. No row contributes to both sides.
-- tests/test_snapshots.py enforces this by mutating post-snapshot data and requiring the
-- feature columns to come back bit-identical.
--
-- Why this table exists at all: in non-contractual retail nobody "cancels", so churn is
-- never observed -- it has to be defined against a clock. Computing features and labels
-- from the same window is textbook leakage and produces a model that predicts the past.
-- Freezing the clock per snapshot is what makes a temporal train/test split meaningful.
--
-- Snapshot grid: first day of each month, derived from data bounds (not hardcoded),
-- starting two months in so the earliest snapshot has real history to summarise.
-- Month-start is used as the as-of boundary so "before the snapshot" is unambiguous --
-- a month-end boundary would leave the status of that day's transactions undefined.
--
-- Label horizon: 90 days, matching the business question "how much revenue will existing
-- customers generate over the next 90 days". This is a documented placeholder: the window
-- is meant to be re-justified from the empirical inter-purchase-time distribution.
--
-- Censoring: snapshots whose 90-day window runs past the end of the data are KEPT but
-- flagged is_label_window_complete = FALSE. Their labels are right-censored -- a customer
-- who simply ran out of observed history would otherwise be recorded as churned. Keeping
-- the rows (flagged) rather than dropping them means the features stay usable for scoring
-- while the labels stay out of training.

CREATE OR REPLACE TABLE customer_snapshots AS
WITH params AS (
    SELECT
        90                                                        AS label_horizon_days,
        MAX(invoice_date)                                         AS data_max,
        date_trunc('month', MIN(invoice_date)) + INTERVAL 2 MONTH AS first_snapshot,
        date_trunc('month', MAX(invoice_date))                    AS last_snapshot
    FROM fact_order_lines
),
snapshot_dates AS (
    SELECT
        CAST(unnest(
            generate_series(first_snapshot, last_snapshot, INTERVAL 1 MONTH)
        ) AS DATE)             AS snapshot_date,
        label_horizon_days,
        data_max
    FROM params
),
-- Everything strictly before the snapshot. Null customer ids are excluded: a snapshot is
-- per-customer by definition, and unattributable revenue stays in the fact tables instead.
history AS (
    SELECT
        sd.snapshot_date,
        sd.label_horizon_days,
        l.customer_id,
        l.invoice_no,
        l.invoice_date,
        l.stock_code,
        l.quantity,
        l.revenue,
        l.is_cancellation,
        l.is_adjustment,
        l.is_product
    FROM snapshot_dates sd
    JOIN fact_order_lines l
      ON l.customer_id IS NOT NULL
     AND l.invoice_date < CAST(sd.snapshot_date AS TIMESTAMP)
),
features AS (
    SELECT
        snapshot_date,
        label_horizon_days,
        customer_id,
        MIN(invoice_date)                                  AS first_invoice_date,
        MAX(invoice_date)                                  AS last_invoice_date,
        COUNT(DISTINCT invoice_no)                         AS invoice_count,
        COUNT(DISTINCT CASE WHEN NOT is_cancellation AND NOT is_adjustment
                            THEN invoice_no END)           AS frequency,
        COUNT(*)                                           AS line_count,
        COUNT(DISTINCT CASE WHEN is_product THEN stock_code END) AS distinct_products,
        COUNT(DISTINCT CAST(invoice_date AS DATE))         AS active_days,
        SUM(quantity)                                      AS units,
        SUM(revenue)                                       AS net_revenue,
        SUM(CASE WHEN NOT is_cancellation AND NOT is_adjustment
                 THEN revenue ELSE 0 END)                  AS gross_revenue,
        SUM(CASE WHEN is_cancellation THEN 1 ELSE 0 END)   AS cancellation_line_count,
        -- "Active before" needs its own definition, otherwise a customer who lapsed two
        -- years ago is scored as a fresh churn risk. Restricted to the same horizon, and
        -- to real purchases: a refund in the window is not activity (see the metric
        -- dictionary's purchase-occasion definition).
        bool_or(NOT is_cancellation AND NOT is_adjustment
                AND invoice_date >= CAST(
            snapshot_date - (label_horizon_days * INTERVAL 1 DAY) AS TIMESTAMP
        ))                                                 AS was_active_in_prior_window
    FROM history
    GROUP BY snapshot_date, label_horizon_days, customer_id
),
-- The label window: [snapshot_date, snapshot_date + horizon).
future AS (
    SELECT
        sd.snapshot_date,
        l.customer_id,
        COUNT(DISTINCT l.invoice_no)                       AS future_invoices,
        COUNT(DISTINCT CASE WHEN NOT l.is_cancellation AND NOT l.is_adjustment
                            THEN l.invoice_no END)         AS future_orders,
        COUNT(*)                                           AS future_lines,
        SUM(l.quantity)                                    AS future_units,
        SUM(l.revenue)                                     AS future_revenue
    FROM snapshot_dates sd
    JOIN fact_order_lines l
      ON l.customer_id IS NOT NULL
     AND l.invoice_date >= CAST(sd.snapshot_date AS TIMESTAMP)
     AND l.invoice_date <  CAST(
             sd.snapshot_date + (sd.label_horizon_days * INTERVAL 1 DAY) AS TIMESTAMP)
    GROUP BY sd.snapshot_date, l.customer_id
)
SELECT
    f.snapshot_date,
    f.customer_id,
    f.label_horizon_days,

    -- Features. History only.
    f.first_invoice_date,
    f.last_invoice_date,
    date_diff('day', CAST(f.last_invoice_date AS DATE), f.snapshot_date)  AS recency_days,
    date_diff('day', CAST(f.first_invoice_date AS DATE), f.snapshot_date) AS tenure_days,
    f.invoice_count,
    f.frequency,
    f.gross_revenue / NULLIF(f.frequency, 0)               AS monetary,
    f.line_count,
    f.distinct_products,
    f.active_days,
    f.units,
    f.net_revenue,
    f.gross_revenue,
    f.cancellation_line_count,
    CAST(f.cancellation_line_count AS DOUBLE) / f.line_count AS return_rate,
    -- Average inter-purchase interval, span-based rather than tenure-based so a long
    -- dormant tail does not inflate it. Null for single-purchase customers: there is no
    -- interval to average, and 0 would falsely read as "buys every day".
    CASE WHEN f.frequency > 1
         THEN date_diff('day',
                        CAST(f.first_invoice_date AS DATE),
                        CAST(f.last_invoice_date AS DATE))::DOUBLE / (f.frequency - 1)
    END                                                    AS avg_days_between_orders,
    f.was_active_in_prior_window,

    -- Labels. Future window only. A customer with no rows in the window is absent from
    -- `future`, so every label is coalesced to 0 -- absence of purchase IS the churn
    -- signal, not a missing value.
    COALESCE(fu.future_invoices, 0)                        AS future_invoices,
    COALESCE(fu.future_orders, 0)                          AS future_orders,
    COALESCE(fu.future_lines, 0)                           AS future_lines,
    COALESCE(fu.future_units, 0)                           AS future_units,
    COALESCE(fu.future_revenue, 0.0)                       AS future_revenue,
    COALESCE(fu.future_orders, 0) > 0                      AS is_active,
    COALESCE(fu.future_orders, 0) = 0                      AS churned,

    CAST(f.snapshot_date + (f.label_horizon_days * INTERVAL 1 DAY) AS TIMESTAMP)
        <= sd.data_max                                     AS is_label_window_complete
FROM features f
LEFT JOIN future fu
       ON fu.snapshot_date = f.snapshot_date
      AND fu.customer_id = f.customer_id
JOIN snapshot_dates sd
     ON sd.snapshot_date = f.snapshot_date;
