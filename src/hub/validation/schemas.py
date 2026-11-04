"""Pandera schemas: data contracts, not vibes. The pipeline fails loudly on violations."""

from __future__ import annotations

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

silver_online_retail_ii_schema = DataFrameSchema(
    columns={
        "invoice_no": Column(str, nullable=False, checks=Check.str_length(min_value=1)),
        "stock_code": Column(str, nullable=False, checks=Check.str_length(min_value=1)),
        "description": Column(str, nullable=True),
        "quantity": Column(int, checks=Check.ne(0)),
        "invoice_date": Column(
            "datetime64[us]",
            nullable=False,
            checks=Check.in_range(
                pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-31"), include_max=True
            ),
        ),
        "unit_price": Column(float, nullable=False),
        "customer_id": Column(pd.Int64Dtype(), nullable=True, checks=Check.gt(0)),
        "country": Column(str, nullable=False, checks=Check.str_length(min_value=1)),
        "is_cancellation": Column(bool, nullable=False),
        "is_adjustment": Column(bool, nullable=False),
        "is_product": Column(bool, nullable=False),
        "is_zero_price": Column(bool, nullable=False),
        "is_quantity_outlier": Column(bool, nullable=False),
        "is_price_outlier": Column(bool, nullable=False),
        "revenue": Column(float, nullable=False),
        "source_sheet": Column(str, checks=Check.isin(["Year 2009-2010", "Year 2010-2011"])),
        "cleaned_at": Column(str, nullable=False),
    },
    checks=[
        # Every cancellation-flagged row must actually carry the 'C' prefix, and
        # nothing else may — the flag and the raw invoice number must never drift.
        Check(
            lambda df: (df["invoice_no"].str.startswith("C")) == df["is_cancellation"],
            error="is_cancellation must exactly match invoice_no starting with 'C'",
        ),
        Check(
            lambda df: (df["invoice_no"].str.startswith("A")) == df["is_adjustment"],
            error="is_adjustment must exactly match invoice_no starting with 'A'",
        ),
        Check(
            lambda df: ~(df["is_cancellation"] & df["is_adjustment"]),
            error="an invoice cannot be both a cancellation and an adjustment",
        ),
        Check(
            lambda df: (df["quantity"] * df["unit_price"] - df["revenue"]).abs().lt(1e-6),
            error="revenue must equal quantity * unit_price",
        ),
    ],
    strict=True,
    coerce=True,
)
