"""Seasonality: a zero-filled daily series, weekday/hour effects, Q4 lift, and STL.

The zero-filling is the whole point. Grouping only days that had sales deletes every
closed day from the analysis, and this retailer's largest weekly effect is precisely a
closed day: 104 of 105 Saturdays have no transactions. A weekday table built on "days
with sales" would report seven roughly similar weekdays and miss that the shop does not
trade on Saturday.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from hub.analytics.metrics import ordinary_orders

DAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


@dataclass(frozen=True)
class STLResult:
    frame: pd.DataFrame
    seasonal_strength: float
    trend_strength: float
    period: int


def daily_revenue(orders: pd.DataFrame) -> pd.Series:
    """Net revenue per calendar day, reindexed onto a gap-free daily axis."""
    if orders.empty:
        raise ValueError("daily_revenue needs at least one order")
    by_day = (
        orders.assign(day=orders["invoice_date"].dt.normalize()).groupby("day")["net_revenue"].sum()
    )
    axis = pd.date_range(by_day.index.min(), by_day.index.max(), freq="D", name="date")
    return by_day.reindex(axis, fill_value=0.0).rename("net_revenue")


def day_of_week_effect(daily: pd.Series) -> pd.DataFrame:
    """Mean revenue per weekday across every calendar day, closed days included."""
    frame = daily.to_frame("net_revenue")
    frame["day_of_week"] = frame.index.dayofweek
    grouped = frame.groupby("day_of_week")["net_revenue"].agg(
        n_days="size",
        zero_days=lambda values: int((values == 0).sum()),
        total_revenue="sum",
        mean_revenue="mean",
    )
    grouped.index = pd.Index([DAY_NAMES[index] for index in grouped.index], name="day_name")
    overall = float(daily.mean())
    grouped["index_vs_mean"] = grouped["mean_revenue"] / overall if overall else np.nan
    return grouped


def hour_effect(orders: pd.DataFrame) -> pd.DataFrame:
    """Share of ordinary orders and of their revenue, by invoice hour."""
    ordinary = ordinary_orders(orders).copy()
    ordinary["hour"] = ordinary["invoice_date"].dt.hour
    grouped = ordinary.groupby("hour").agg(
        orders=("invoice_no", "nunique"),
        revenue=("net_revenue", "sum"),
    )
    grouped["order_share"] = grouped["orders"] / grouped["orders"].sum()
    grouped["revenue_share"] = grouped["revenue"] / grouped["revenue"].sum()
    grouped.index.name = "hour"
    return grouped


def q4_lift(daily: pd.Series) -> pd.DataFrame:
    """Q4 versus non-Q4 mean daily revenue, with unavailable years marked as such.

    `q4_complete` is False when the data does not cover the whole Oct-Dec window, which
    is the case for 2011 (the dataset ends on 9 December). The lift is still computed
    from the days that exist and must be read with that caveat.
    """
    frame = daily.to_frame("net_revenue")
    frame["year"] = frame.index.year
    frame["is_q4"] = frame.index.quarter == 4
    start, end = daily.index.min(), daily.index.max()

    rows = []
    for year, group in frame.groupby("year"):
        q4 = group.loc[group["is_q4"], "net_revenue"]
        other = group.loc[~group["is_q4"], "net_revenue"]
        lift = (
            float(q4.mean() / other.mean())
            if len(q4) and len(other) and other.mean() != 0
            else float("nan")
        )
        q4_complete = (start <= pd.Timestamp(year=year, month=10, day=1)) and (
            end >= pd.Timestamp(year=year, month=12, day=31)
        )
        rows.append(
            {
                "year": int(year),
                "q4_days": int(len(q4)),
                "non_q4_days": int(len(other)),
                "q4_mean": float(q4.mean()) if len(q4) else float("nan"),
                "non_q4_mean": float(other.mean()) if len(other) else float("nan"),
                "lift": lift,
                "q4_complete": bool(q4_complete),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def stl_decompose(daily: pd.Series, period: int = 7) -> STLResult:
    """Additive, robust STL decomposition and its seasonal/trend strength.

    Strength is `max(0, 1 - Var(residual) / Var(residual + component))`: the share of
    the component-plus-noise variation that the component explains.
    """
    if daily.isna().any():
        raise ValueError("STL needs a gap-free series; reindex with daily_revenue first")
    if len(daily) < 2 * period:
        raise ValueError(f"STL needs at least {2 * period} observations for period={period}")

    fitted = STL(daily, period=period, robust=True).fit()
    frame = pd.DataFrame(
        {
            "observed": daily,
            "trend": fitted.trend,
            "seasonal": fitted.seasonal,
            "resid": fitted.resid,
        }
    )
    return STLResult(
        frame=frame,
        seasonal_strength=_strength(fitted.resid, fitted.seasonal),
        trend_strength=_strength(fitted.resid, fitted.trend),
        period=period,
    )


def _strength(residual: np.ndarray, component: np.ndarray) -> float:
    residual_variance = float(np.var(residual))
    combined_variance = float(np.var(residual + component))
    if combined_variance == 0:
        return 0.0
    return float(max(0.0, 1.0 - residual_variance / combined_variance))
