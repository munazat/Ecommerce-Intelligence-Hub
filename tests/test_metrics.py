"""Metrics tests: every expected number is hand-computed from a tiny fixture.

Revenue bases, customer sets, and cancellation signs are the three places this module
can silently go wrong, so each has a test that would fail if the implementation flipped
to the plausible-but-wrong alternative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hub.analytics import metrics


def test_revenue_summary_hand_computed(make_orders):
    orders = make_orders(
        [
            {
                "invoice_no": "500001",
                "net_revenue": 50.0,
                "gross_revenue": 50.0,
                "product_revenue": 50.0,
                "product_count": 2,
            },
            {
                "invoice_no": "500002",
                "net_revenue": 15.0,
                "gross_revenue": 15.0,
                "product_revenue": 15.0,
                "product_count": 1,
            },
            {
                "invoice_no": "500003",
                "customer_id": None,
                "net_revenue": 18.0,
                "gross_revenue": 18.0,
                "product_revenue": 0.0,
                "product_count": 0,
            },
            {
                "invoice_no": "C500004",
                "net_revenue": -10.0,
                "gross_revenue": 0.0,
                "product_revenue": -10.0,
                "is_cancellation": True,
            },
            {
                "invoice_no": "A500005",
                "customer_id": None,
                "net_revenue": -500.0,
                "gross_revenue": 0.0,
                "product_revenue": 0.0,
                "product_count": 0,
                "is_adjustment": True,
            },
        ]
    )
    summary = metrics.revenue_summary(orders)

    assert summary.net_revenue == pytest.approx(-427.0)
    assert summary.gross_revenue == pytest.approx(83.0)
    assert summary.reversal_revenue == pytest.approx(-510.0)
    assert summary.order_count == 5
    assert summary.ordinary_order_count == 3
    assert summary.product_order_count == 2
    assert summary.product_revenue == pytest.approx(65.0)
    # (50 + 15) / 2, and the postage-only order is not a product order.
    assert summary.aov == pytest.approx(32.5)
    assert summary.median_product_order_value == pytest.approx(32.5)
    assert summary.identified_net_revenue == pytest.approx(55.0)
    assert summary.unattributable_net_revenue == pytest.approx(-482.0)


def test_arpu_uses_the_same_customer_set_on_both_sides(make_orders):
    orders = make_orders(
        [
            {"invoice_no": "1", "customer_id": 1001, "invoice_date": pd.Timestamp("2010-01-05")},
            {
                "invoice_no": "2",
                "customer_id": 1001,
                "invoice_date": pd.Timestamp("2010-01-20"),
                "net_revenue": 50.0,
            },
            {
                "invoice_no": "3",
                "customer_id": 1002,
                "invoice_date": pd.Timestamp("2010-01-06"),
                "net_revenue": 30.0,
            },
            {
                "invoice_no": "C4",
                "customer_id": 1002,
                "invoice_date": pd.Timestamp("2010-01-15"),
                "net_revenue": -20.0,
                "gross_revenue": 0.0,
                "product_revenue": -20.0,
                "is_cancellation": True,
            },
            # Customer 1003 only ever refunds; they are not an active buyer in the window.
            {
                "invoice_no": "C5",
                "customer_id": 1003,
                "invoice_date": pd.Timestamp("2010-01-07"),
                "net_revenue": -40.0,
                "gross_revenue": 0.0,
                "product_revenue": -40.0,
                "is_cancellation": True,
            },
        ]
    )
    value = metrics.arpu(orders, pd.Timestamp("2010-01-01"), pd.Timestamp("2010-02-01"))
    # (10 + 50 + 30 - 20) / 2 buyers. Counting 1003's refund without them as a buyer
    # would give (70 - 40) / 2 = 15, which is the asymmetry this test rules out.
    assert value == pytest.approx(35.0)


def test_purchase_occasions_collapse_same_day_invoices(make_orders):
    orders = make_orders(
        [
            {"invoice_no": "1", "invoice_date": pd.Timestamp("2010-01-05 09:00")},
            {"invoice_no": "2", "invoice_date": pd.Timestamp("2010-01-05 14:00")},
            {"invoice_no": "3", "invoice_date": pd.Timestamp("2010-01-06 09:00")},
            {
                "invoice_no": "C4",
                "invoice_date": pd.Timestamp("2010-01-06 10:00"),
                "is_cancellation": True,
                "net_revenue": -10.0,
            },
            {
                "invoice_no": "5",
                "invoice_date": pd.Timestamp("2010-01-07 09:00"),
                "customer_id": None,
            },
        ]
    )
    occasions = metrics.purchase_occasions(orders)

    assert len(occasions) == 2
    assert list(occasions["occasion_date"]) == [
        pd.Timestamp("2010-01-05"),
        pd.Timestamp("2010-01-06"),
    ]
    intervals = metrics.purchase_intervals(orders)
    assert list(intervals["gap_days"]) == [1]


def test_repeat_purchase_rate_hand_computed(make_orders):
    orders = make_orders(
        [
            {"invoice_no": "1", "customer_id": 1001, "invoice_date": pd.Timestamp("2010-01-01")},
            {"invoice_no": "2", "customer_id": 1001, "invoice_date": pd.Timestamp("2010-02-01")},
            {"invoice_no": "3", "customer_id": 1002, "invoice_date": pd.Timestamp("2010-01-02")},
            {"invoice_no": "4", "customer_id": 1003, "invoice_date": pd.Timestamp("2010-01-03")},
            {"invoice_no": "5", "customer_id": 1003, "invoice_date": pd.Timestamp("2010-01-20")},
        ]
    )
    result = metrics.repeat_purchase_rate(orders)

    assert (result.buyers, result.repeaters) == (3, 2)
    assert result.rate == pytest.approx(2 / 3)
    assert result.single_purchase_share == pytest.approx(1 / 3)


def test_interval_summary_and_window_coverage():
    intervals = pd.DataFrame({"customer_id": [1, 1, 2], "gap_days": [10, 10, 100]})
    summary = metrics.interval_summary(intervals)

    assert (summary.intervals, summary.customers_with_intervals) == (3, 2)
    assert summary.mean_days == pytest.approx(40.0)
    assert summary.median_days == pytest.approx(10.0)
    assert summary.p75_days == pytest.approx(55.0)
    assert summary.p90_days == pytest.approx(82.0)
    assert summary.max_days == pytest.approx(100.0)

    coverage = metrics.window_coverage(intervals, windows=(30, 100))
    assert list(coverage["window_days"]) == [30, 100]
    assert list(coverage["share_covered"]) == pytest.approx([2 / 3, 1.0])


def test_interval_summary_is_empty_safe():
    summary = metrics.interval_summary(pd.DataFrame({"customer_id": [], "gap_days": []}))
    assert summary.intervals == 0
    assert np.isnan(summary.mean_days)


def test_return_rates_use_signed_cancellation_sums(make_lines):
    """A C line that reverses a charge must reduce the refund total, not add to it."""
    lines = make_lines(
        [
            {"stock_code": "A", "quantity": 100, "revenue": 1000.0},
            {"stock_code": "A", "quantity": -20, "revenue": -200.0, "is_cancellation": True},
            # Positive quantity and revenue on a C invoice: a charge reversal, not a refund.
            {"stock_code": "M", "quantity": 1, "revenue": 10.0, "is_cancellation": True},
            {"stock_code": "B", "quantity": 1, "revenue": -500.0, "is_adjustment": True},
            {"stock_code": "A", "quantity": 50, "revenue": 500.0},
        ]
    )
    rates = metrics.return_rates(lines)

    assert rates.purchased_units == pytest.approx(150.0)
    assert rates.cancelled_units == pytest.approx(19.0)
    assert rates.unit_rate == pytest.approx(19.0 / 150.0)
    assert rates.gross_value == pytest.approx(1500.0)
    assert rates.cancelled_value == pytest.approx(190.0)
    assert rates.value_rate == pytest.approx(190.0 / 1500.0)
    # The adjustment line is neither an ordinary line nor a cancellation.
    assert (rates.cancelled_lines, rates.ordinary_lines) == (2, 2)
    assert rates.line_rate == pytest.approx(2 / 2)


def test_product_return_rates_flags_fully_returned_and_never_purchased(make_lines):
    lines = make_lines(
        [
            {"stock_code": "A", "quantity": 5, "revenue": 50.0},
            {"stock_code": "A", "quantity": -5, "revenue": -50.0, "is_cancellation": True},
            {"stock_code": "B", "quantity": 10, "revenue": 100.0},
            {"stock_code": "B", "quantity": -1, "revenue": -10.0, "is_cancellation": True},
            {"stock_code": "C", "quantity": -2, "revenue": -20.0, "is_cancellation": True},
            {"stock_code": "POST", "quantity": 1, "revenue": 18.0, "is_product": False},
        ]
    )
    rates = metrics.product_return_rates(lines)

    assert set(rates.index) == {"A", "B", "C"}
    assert rates.loc["A", "unit_return_rate"] == pytest.approx(1.0)
    assert rates.loc["A", "net_revenue"] == pytest.approx(0.0)
    assert rates.loc["B", "unit_return_rate"] == pytest.approx(0.1)
    assert rates.loc["B", "net_revenue"] == pytest.approx(90.0)
    # Never purchased, so there is no rate to state -- NaN, not a fabricated zero.
    assert np.isnan(rates.loc["C", "unit_return_rate"])
    assert rates.loc["C", "cancelled_units"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([10.0, 10.0, 10.0, 10.0], 0.0),
        ([0.0, 0.0, 0.0, 10.0], 0.75),
        ([1.0, 2.0, 3.0, 4.0], 0.25),
    ],
)
def test_gini_known_values(values, expected):
    assert metrics.gini(values) == pytest.approx(expected)


def test_gini_clips_negative_revenue_by_default():
    assert metrics.gini([-5.0, 5.0, 10.0]) == pytest.approx(0.4444444444)
    with pytest.raises(ValueError, match="non-negative"):
        metrics.gini([-5.0, 5.0, 10.0], clip_negative=False)


def test_gini_rejects_nan_and_empty():
    with pytest.raises(ValueError, match="NaN"):
        metrics.gini([1.0, float("nan")])
    with pytest.raises(ValueError, match="at least one"):
        metrics.gini([])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([8.0, 1.0, 1.0], 1),
        ([1.0, 1.0, 1.0, 1.0, 1.0], 4),
        ([10.0, 5.0, -2.0, 1.0], 2),
    ],
)
def test_pareto_share_known_values(values, expected):
    assert metrics.pareto_share(values).n_needed == expected


def test_pareto_share_rejects_unusable_totals():
    with pytest.raises(ValueError, match="positive total"):
        metrics.pareto_share([-1.0, -2.0])
    with pytest.raises(ValueError, match="NaN"):
        metrics.pareto_share([1.0, float("nan")])


def test_customer_concentration_reports_negative_customers(make_orders):
    orders = make_orders(
        [
            {"invoice_no": "1", "customer_id": 1001, "net_revenue": 100.0},
            {"invoice_no": "2", "customer_id": 1002, "net_revenue": 50.0},
            {
                "invoice_no": "C3",
                "customer_id": 1003,
                "net_revenue": -20.0,
                "is_cancellation": True,
            },
            {"invoice_no": "4", "customer_id": 1004, "net_revenue": 0.0},
            {"invoice_no": "5", "customer_id": None, "net_revenue": 999.0},
        ]
    )
    result = metrics.customer_concentration(orders)

    assert result.pareto.n_items == 4  # null customer excluded
    assert result.pareto.n_needed == 2
    assert result.negative_items == 1
    assert result.negative_total == pytest.approx(-20.0)
    # [100, 50, 0, -20] clipped to [100, 50, 0, 0].
    assert result.gini == pytest.approx(0.5833333333)


def test_churn_sensitivity_uses_the_snapshot_definition(make_orders):
    rows = [
        # C1 buys in Jan, Feb and June.
        {"invoice_no": "1", "customer_id": 1, "invoice_date": pd.Timestamp("2010-01-10")},
        {"invoice_no": "2", "customer_id": 1, "invoice_date": pd.Timestamp("2010-02-10")},
        {"invoice_no": "3", "customer_id": 1, "invoice_date": pd.Timestamp("2010-06-20")},
        # C2 buys once in January.
        {"invoice_no": "4", "customer_id": 2, "invoice_date": pd.Timestamp("2010-01-15")},
        # C3 buys in March and May.
        {"invoice_no": "5", "customer_id": 3, "invoice_date": pd.Timestamp("2010-03-05")},
        {"invoice_no": "6", "customer_id": 3, "invoice_date": pd.Timestamp("2010-05-05")},
    ]
    orders = make_orders(rows)

    table = metrics.churn_sensitivity(orders, windows=(60,), min_history_months=0)

    # Snapshots run Jan..Jun, but May's 60-day window ends on 30 June, past the last
    # observed purchase (20 June), so it is censored out.
    assert table["snapshot_date"].tolist() == [
        pd.Timestamp("2010-02-01"),
        pd.Timestamp("2010-03-01"),
        pd.Timestamp("2010-04-01"),
    ]
    # Eligibility is the 60-day lookback: C1+C2 in Feb and Mar, then C1+C3 in April
    # (C2's January purchase has aged out).
    assert table["churn_rate"].tolist() == pytest.approx([0.5, 1.0, 0.5])
    assert table["eligible_customers"].tolist() == [2, 2, 2]
    assert table["churned_customers"].tolist() == [1, 2, 1]


def test_churn_sensitivity_rejects_bad_input(make_orders):
    with pytest.raises(ValueError, match="positive"):
        metrics.churn_sensitivity(make_orders([{}]), windows=(0,))

    refunds_only = make_orders(
        [
            {"invoice_no": "C1", "customer_id": 1, "net_revenue": -5.0, "is_cancellation": True},
        ]
    )
    with pytest.raises(ValueError, match="no purchase occasions"):
        metrics.churn_sensitivity(refunds_only)
