"""Cohort tests: the triangle, censoring, and the net-revenue basis.

The fixture spans four complete months so the retention triangle can be checked cell by
cell, including the cells that must be NaN rather than zero.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hub.analytics import cohorts


def _dates(*days: str) -> list[dict]:
    return [{"invoice_date": pd.Timestamp(day)} for day in days]


@pytest.fixture
def orders(make_orders):
    return make_orders(
        [
            # C1: Jan, Feb, Mar. Revenue 100, 50, 0.
            {
                "invoice_no": "1",
                "customer_id": 1,
                "invoice_date": pd.Timestamp("2010-01-01"),
                "net_revenue": 100.0,
            },
            {
                "invoice_no": "2",
                "customer_id": 1,
                "invoice_date": pd.Timestamp("2010-02-10"),
                "net_revenue": 50.0,
            },
            {
                "invoice_no": "3",
                "customer_id": 1,
                "invoice_date": pd.Timestamp("2010-03-10"),
                "net_revenue": 0.0,
            },
            # C2: Jan only, but refunds 30 in Feb. A refund is revenue, not an occasion.
            {
                "invoice_no": "4",
                "customer_id": 2,
                "invoice_date": pd.Timestamp("2010-01-15"),
                "net_revenue": 100.0,
            },
            {
                "invoice_no": "C5",
                "customer_id": 2,
                "invoice_date": pd.Timestamp("2010-02-15"),
                "net_revenue": -30.0,
                "gross_revenue": 0.0,
                "product_revenue": -30.0,
                "is_cancellation": True,
            },
            # C3: Feb and Apr, skipping Mar.
            {
                "invoice_no": "6",
                "customer_id": 3,
                "invoice_date": pd.Timestamp("2010-02-10"),
                "net_revenue": 40.0,
            },
            {
                "invoice_no": "7",
                "customer_id": 3,
                "invoice_date": pd.Timestamp("2010-04-30"),
                "net_revenue": 20.0,
            },
            # C4: Mar and Apr.
            {
                "invoice_no": "8",
                "customer_id": 4,
                "invoice_date": pd.Timestamp("2010-03-05"),
                "net_revenue": 60.0,
            },
            {
                "invoice_no": "9",
                "customer_id": 4,
                "invoice_date": pd.Timestamp("2010-04-20"),
                "net_revenue": 40.0,
            },
        ]
    )


def test_retention_triangle(orders):
    tables = cohorts.build_cohorts(orders)

    assert list(tables.retention.index) == [
        pd.Timestamp("2010-01-01"),
        pd.Timestamp("2010-02-01"),
        pd.Timestamp("2010-03-01"),
    ]
    assert list(tables.retention.columns) == [0, 1, 2, 3]
    assert tables.cohort_sizes.to_dict() == {
        pd.Timestamp("2010-01-01"): 2,
        pd.Timestamp("2010-02-01"): 1,
        pd.Timestamp("2010-03-01"): 1,
    }

    assert tables.first_complete_month == pd.Timestamp("2010-01-01")
    assert tables.last_complete_month == pd.Timestamp("2010-04-01")
    assert tables.excluded_partial_months == ()
    assert tables.customers_outside_window == 0

    jan, feb, mar = tables.retention.index
    # Jan: both bought in month 0, only C1 in months 1-3. April is fully observed,
    # so month 3 is a real zero, not a NaN.
    assert tables.retention.loc[jan].tolist() == [1.0, 0.5, 0.5, 0.0]
    # Feb: C3 skipped March and came back in April.
    assert tables.retention.loc[feb].tolist()[:3] == [1.0, 0.0, 1.0]
    assert pd.isna(tables.retention.loc[feb, 3])
    # Mar: only months 0 and 1 have elapsed before the data ends.
    assert tables.retention.loc[mar, 0] == 1.0
    assert tables.retention.loc[mar, 1] == 1.0
    assert pd.isna(tables.retention.loc[mar, 2])


def test_revenue_uses_net_and_excludes_refunds_from_activity(orders):
    tables = cohorts.build_cohorts(orders)
    jan = pd.Timestamp("2010-01-01")

    # Jan month 0: (100 + 100) / 2. Month 1: (50 - 30) / 2 -- the refund reduces the
    # cohort's revenue even though it does not count as an occasion.
    assert tables.revenue_per_customer.loc[jan].tolist() == [100.0, 10.0, 0.0, 0.0]
    assert tables.revenue_retention.loc[jan].tolist() == pytest.approx([1.0, 0.1, 0.0, 0.0])
    assert tables.cumulative_revenue_per_customer.loc[jan].tolist() == [
        100.0,
        110.0,
        110.0,
        110.0,
    ]

    feb = pd.Timestamp("2010-02-01")
    assert tables.revenue_per_customer.loc[feb].tolist()[:3] == [40.0, 0.0, 20.0]
    assert tables.revenue_retention.loc[feb].tolist()[:3] == [1.0, 0.0, 0.5]

    mar = pd.Timestamp("2010-03-01")
    assert tables.revenue_per_customer.loc[mar, 0] == 60.0
    assert tables.revenue_per_customer.loc[mar, 1] == 40.0
    assert pd.isna(tables.revenue_per_customer.loc[mar, 2])
    assert tables.cumulative_revenue_per_customer.loc[mar, 1] == 100.0
    assert pd.isna(tables.cumulative_revenue_per_customer.loc[mar, 2])


def test_partial_first_month_is_dropped_and_its_customers_are_outside(make_orders):
    orders = make_orders(
        [
            # January starts on the 15th, so it is not a complete month.
            {
                "invoice_no": "1",
                "customer_id": 1,
                "invoice_date": pd.Timestamp("2010-01-15"),
                "net_revenue": 100.0,
            },
            {
                "invoice_no": "2",
                "customer_id": 2,
                "invoice_date": pd.Timestamp("2010-02-10"),
                "net_revenue": 50.0,
            },
            {
                "invoice_no": "3",
                "customer_id": 2,
                "invoice_date": pd.Timestamp("2010-03-31"),
                "net_revenue": 50.0,
            },
        ]
    )
    tables = cohorts.build_cohorts(orders)

    assert tables.first_complete_month == pd.Timestamp("2010-02-01")
    assert tables.last_complete_month == pd.Timestamp("2010-03-01")
    assert tables.excluded_partial_months == (pd.Timestamp("2010-01-01"),)
    # Customer 1's only activity is in the dropped month, so they enter no cohort.
    assert tables.customers_outside_window == 1
    assert tables.cohort_sizes.to_dict() == {pd.Timestamp("2010-02-01"): 1}


def test_partial_final_month_is_dropped_not_reported_as_churn(make_orders):
    orders = make_orders(
        [
            {
                "invoice_no": "1",
                "customer_id": 1,
                "invoice_date": pd.Timestamp("2010-01-01"),
                "net_revenue": 100.0,
            },
            {
                "invoice_no": "2",
                "customer_id": 1,
                "invoice_date": pd.Timestamp("2010-02-10"),
                "net_revenue": 100.0,
            },
            # April is only partially covered: data stops on the 15th.
            {
                "invoice_no": "3",
                "customer_id": 1,
                "invoice_date": pd.Timestamp("2010-04-15"),
                "net_revenue": 100.0,
            },
        ]
    )
    tables = cohorts.build_cohorts(orders)

    assert tables.last_complete_month == pd.Timestamp("2010-03-01")
    assert tables.excluded_partial_months == (pd.Timestamp("2010-04-01"),)
    # April is not in the matrix at all: without dropping it, this cohort would look
    # like it churned in month 3.
    assert list(tables.retention.columns) == [0, 1, 2]
    assert tables.retention.loc[pd.Timestamp("2010-01-01")].tolist() == [1.0, 1.0, 0.0]


def test_left_censored_cohort_is_flagged(make_orders):
    orders = make_orders(
        [
            {"invoice_no": "1", "customer_id": 1, "invoice_date": pd.Timestamp("2010-01-01")},
            {"invoice_no": "2", "customer_id": 1, "invoice_date": pd.Timestamp("2010-02-01")},
        ]
    )
    tables = cohorts.build_cohorts(orders)

    assert tables.left_censored_cohort == tables.first_complete_month
    assert tables.left_censored_cohort == pd.Timestamp("2010-01-01")


def test_no_occasions_raises(make_orders):
    orders = make_orders(
        [
            {
                "invoice_no": "C1",
                "customer_id": 1,
                "net_revenue": -5.0,
                "gross_revenue": 0.0,
                "product_revenue": -5.0,
                "is_cancellation": True,
            },
        ]
    )
    with pytest.raises(ValueError, match="no purchase occasions"):
        cohorts.build_cohorts(orders)


def test_partial_spanning_no_complete_month_raises(make_orders):
    orders = make_orders(
        [
            {"invoice_no": "1", "customer_id": 1, "invoice_date": pd.Timestamp("2010-01-10")},
            {"invoice_no": "2", "customer_id": 1, "invoice_date": pd.Timestamp("2010-01-20")},
        ]
    )
    with pytest.raises(ValueError, match="does not cover any complete calendar month"):
        cohorts.build_cohorts(orders)
