"""Seasonality tests: the zero-filled calendar, Q4 availability, and STL integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hub.analytics import seasonality


def test_daily_revenue_fills_closed_days_with_zero(make_orders):
    orders = make_orders(
        [
            {
                "invoice_no": "1",
                "invoice_date": pd.Timestamp("2010-01-01 10:00"),
                "net_revenue": 10.0,
            },
            {
                "invoice_no": "2",
                "invoice_date": pd.Timestamp("2010-01-03 10:00"),
                "net_revenue": 5.0,
            },
            {
                "invoice_no": "3",
                "invoice_date": pd.Timestamp("2010-01-05 10:00"),
                "net_revenue": 7.0,
            },
        ]
    )
    daily = seasonality.daily_revenue(orders)

    assert daily.index.equals(pd.date_range("2010-01-01", "2010-01-05", freq="D", name="date"))
    assert daily.tolist() == [10.0, 0.0, 5.0, 0.0, 7.0]
    assert daily.name == "net_revenue"


def test_daily_revenue_rejects_empty_input(make_orders):
    empty = make_orders([])
    with pytest.raises(ValueError, match="at least one order"):
        seasonality.daily_revenue(empty)


def test_day_of_week_reports_closed_saturdays_as_zero():
    # Two full weeks: Saturday and Sunday are closed, weekdays earn 10 a day.
    index = pd.date_range("2010-01-04", "2010-01-17", freq="D", name="date")
    values = [0.0 if day.weekday() >= 5 else 10.0 for day in index]
    daily = pd.Series(values, index=index, name="net_revenue")

    table = seasonality.day_of_week_effect(daily)

    assert list(table.index) == list(seasonality.DAY_NAMES)
    assert table.loc["Saturday", "n_days"] == 2
    assert table.loc["Saturday", "zero_days"] == 2
    assert table.loc["Saturday", "mean_revenue"] == 0.0
    assert table.loc["Saturday", "index_vs_mean"] == 0.0
    assert table.loc["Monday", "mean_revenue"] == 10.0
    # Overall mean includes the closed days: (10 * 10) / 14.
    assert table.loc["Monday", "index_vs_mean"] == pytest.approx(10.0 / (100.0 / 14.0))


def test_hour_effect_ignores_reversals_and_shares_sum_to_one(make_orders):
    orders = make_orders(
        [
            {
                "invoice_no": "1",
                "invoice_date": pd.Timestamp("2010-01-05 09:15"),
                "net_revenue": 10.0,
            },
            {
                "invoice_no": "2",
                "invoice_date": pd.Timestamp("2010-01-05 09:45"),
                "net_revenue": 30.0,
            },
            {
                "invoice_no": "3",
                "invoice_date": pd.Timestamp("2010-01-05 15:00"),
                "net_revenue": 60.0,
            },
            {
                "invoice_no": "C4",
                "invoice_date": pd.Timestamp("2010-01-05 09:30"),
                "net_revenue": -100.0,
                "gross_revenue": 0.0,
                "product_revenue": -100.0,
                "is_cancellation": True,
            },
        ]
    )
    table = seasonality.hour_effect(orders)

    assert set(table.index) == {9, 15}
    assert table.loc[9, "orders"] == 2
    assert table.loc[9, "revenue"] == 40.0
    assert table.loc[15, "orders"] == 1
    assert table["order_share"].sum() == pytest.approx(1.0)
    assert table["revenue_share"].sum() == pytest.approx(1.0)


def test_q4_lift_requires_a_comparable_baseline():
    # 2009 starts in November (a partial Q4 with no baseline); 2010 has no Q4 at all.
    index = pd.date_range("2009-11-01", "2010-09-30", freq="D", name="date")
    daily = pd.Series(10.0, index=index, name="net_revenue")

    table = seasonality.q4_lift(daily)

    assert pd.isna(table.loc[2009, "lift"])
    assert not table.loc[2009, "q4_complete"]
    assert table.loc[2010, "q4_days"] == 0
    assert pd.isna(table.loc[2010, "lift"])


def test_q4_lift_computes_ratio_for_a_full_year():
    index = pd.date_range("2009-01-01", "2009-12-31", freq="D", name="date")
    q4 = index.quarter == 4
    daily = pd.Series(np.where(q4, 20.0, 10.0), index=index, name="net_revenue")

    table = seasonality.q4_lift(daily)

    assert table.loc[2009, "lift"] == pytest.approx(2.0)
    assert table.loc[2009, "q4_complete"]
    assert table.loc[2009, "q4_days"] == 92
    assert table.loc[2009, "non_q4_days"] == 273


def test_stl_reconstructs_the_series_and_detects_weekly_pattern():
    index = pd.date_range("2010-01-04", periods=84, freq="D", name="date")
    daily = pd.Series(
        100.0 + 20.0 * np.sin(2 * np.pi * index.dayofweek / 7), index=index, name="net_revenue"
    )

    result = seasonality.stl_decompose(daily)

    assert result.period == 7
    reconstructed = result.frame["trend"] + result.frame["seasonal"] + result.frame["resid"]
    assert np.allclose(reconstructed, daily)
    assert result.seasonal_strength > 0.9
    assert 0.0 <= result.trend_strength <= 1.0


def test_stl_rejects_gaps_and_short_series():
    gappy = pd.Series(
        [1.0, 2.0, np.nan] * 10,
        index=pd.date_range("2010-01-01", periods=30, freq="D", name="date"),
        name="net_revenue",
    )
    with pytest.raises(ValueError, match="gap-free"):
        seasonality.stl_decompose(gappy)

    short = pd.Series(
        1.0,
        index=pd.date_range("2010-01-01", periods=10, freq="D", name="date"),
        name="net_revenue",
    )
    with pytest.raises(ValueError, match="at least 14 observations"):
        seasonality.stl_decompose(short)
