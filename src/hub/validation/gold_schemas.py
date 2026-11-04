"""Pandera contracts for the gold layer: grain and invariants, enforced loudly.

These schemas deliberately pin *primary keys* and *cross-field invariants* rather than
column dtypes. Types are already fixed by the SQL that produces each table and by the
Parquet file it writes; re-declaring them here would only add a second place for them to
drift. What a warehouse actually loses silently is its grain -- a duplicated invoice_no in
fact_orders, or a snapshot row that saw the future -- so that is what these check.

`unique=` is what makes a grain a contract instead of a comment: the build fails rather
than writing a table that quietly double-counts revenue.
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

fact_order_lines_schema = DataFrameSchema(
    columns={
        "line_key": Column(nullable=False, checks=[Check.gt(0)]),
        "invoice_no": Column(nullable=False, checks=Check.str_length(min_value=1)),
        "stock_code": Column(nullable=False, checks=Check.str_length(min_value=1)),
        "invoice_date": Column(nullable=False),
        "invoice_date_key": Column(nullable=False),
    },
    checks=[
        Check(
            lambda df: (df["quantity"] * df["unit_price"] - df["revenue"]).abs().lt(1e-6),
            error="revenue must equal quantity * unit_price",
        ),
        Check(
            lambda df: ~(df["is_cancellation"] & df["is_adjustment"]),
            error="a line cannot be both a cancellation and an adjustment",
        ),
    ],
    unique=["line_key"],
)

fact_orders_schema = DataFrameSchema(
    columns={
        "invoice_no": Column(nullable=False, checks=Check.str_length(min_value=1)),
        "invoice_date": Column(nullable=False),
        "line_count": Column(nullable=False, checks=Check.gt(0)),
    },
    checks=[
        # The roadmap's "no negative revenue after returns handling". Reversals ('C') and
        # bad-debt write-offs ('A') are *supposed* to be negative -- they are the handling.
        # Everything else must not be.
        Check(
            lambda df: df.loc[~(df["is_cancellation"] | df["is_adjustment"]), "net_revenue"] >= 0,
            error="ordinary orders must not have negative net revenue",
        ),
        Check(
            lambda df: (
                (
                    df.loc[~(df["is_cancellation"] | df["is_adjustment"]), "net_revenue"]
                    - df.loc[~(df["is_cancellation"] | df["is_adjustment"]), "gross_revenue"]
                ).abs()
                < 1e-6
            ),
            error="gross and net revenue must agree once reversals are excluded",
        ),
        Check(
            lambda df: ~(df["is_cancellation"] & df["is_adjustment"]),
            error="an order cannot be both a cancellation and an adjustment",
        ),
        Check(
            lambda df: (
                (df["product_revenue"] + df["non_product_revenue"] - df["net_revenue"])
                .abs()
                .lt(1e-6)
            ),
            error="product and non-product revenue must sum to net revenue",
        ),
    ],
    unique=["invoice_no"],
)

dim_customer_schema = DataFrameSchema(
    columns={
        "customer_id": Column(nullable=False, checks=Check.gt(0)),
        "country": Column(nullable=False, checks=Check.str_length(min_value=1)),
        "invoice_count": Column(nullable=False, checks=Check.gt(0)),
    },
    checks=[
        Check(
            lambda df: df["purchase_count"] <= df["invoice_count"],
            error="purchase occasions cannot exceed total invoices",
        ),
        Check(
            lambda df: df["last_invoice_date"] >= df["first_invoice_date"],
            error="last invoice cannot precede first invoice",
        ),
    ],
    unique=["customer_id"],
)

dim_product_schema = DataFrameSchema(
    columns={
        "stock_code": Column(nullable=False, checks=Check.str_length(min_value=1)),
        "line_count": Column(nullable=False, checks=Check.gt(0)),
        "cancellation_rate": Column(nullable=False, checks=Check.in_range(0.0, 1.0)),
    },
    unique=["stock_code"],
)

dim_date_schema = DataFrameSchema(
    columns={
        "date_key": Column(nullable=False),
        "date_day": Column(nullable=False),
        "month": Column(nullable=False, checks=Check.in_range(1, 12)),
        "iso_week": Column(nullable=False, checks=Check.in_range(1, 53)),
        "iso_day_of_week": Column(nullable=False, checks=Check.in_range(1, 7)),
        "day_of_week_sun0": Column(nullable=False, checks=Check.in_range(0, 6)),
    },
    checks=[
        Check(
            lambda df: df["is_weekend"] == df["day_of_week_sun0"].isin([0, 6]),
            error="is_weekend must agree with the Sunday-based day-of-week index",
        ),
        Check(
            lambda df: df["is_q4"] == (df["quarter"] == 4),
            error="is_q4 must agree with quarter",
        ),
    ],
    unique=["date_key"],
)

customer_snapshots_schema = DataFrameSchema(
    columns={
        "snapshot_date": Column(nullable=False),
        "customer_id": Column(nullable=False, checks=Check.gt(0)),
        "label_horizon_days": Column(nullable=False, checks=Check.gt(0)),
        "recency_days": Column(nullable=False, checks=Check.ge(0)),
        "tenure_days": Column(nullable=False, checks=Check.ge(0)),
        "frequency": Column(nullable=False, checks=Check.ge(0)),
    },
    checks=[
        # recency is measured from the last purchase, tenure from the first; the last
        # purchase can never precede the first, so recency can never exceed tenure.
        Check(
            lambda df: df["recency_days"] <= df["tenure_days"],
            error="recency_days cannot exceed tenure_days",
        ),
        Check(
            lambda df: df["is_active"] != df["churned"],
            error="is_active and churned must be exact complements",
        ),
        Check(
            lambda df: df["is_active"] == (df["future_orders"] > 0),
            error="is_active must agree with future_orders",
        ),
        Check(
            lambda df: df.loc[df["frequency"] <= 1, "avg_days_between_orders"].isna().all(),
            error="avg_days_between_orders must be null when there is no interval to average",
        ),
        Check(
            lambda df: df["was_active_in_prior_window"].notna().all(),
            error="was_active_in_prior_window must never be null",
        ),
    ],
    unique=["customer_id", "snapshot_date"],
)

GOLD_SCHEMAS = {
    "fact_order_lines": fact_order_lines_schema,
    "fact_orders": fact_orders_schema,
    "dim_customer": dim_customer_schema,
    "dim_product": dim_product_schema,
    "dim_date": dim_date_schema,
    "customer_snapshots": customer_snapshots_schema,
}
