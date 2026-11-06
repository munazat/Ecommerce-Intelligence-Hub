"""Reusable business metrics over the gold warehouse.

Every function here implements a definition in `docs/metrics_dictionary.md`; when the
two disagree, the dictionary wins and this file is the bug. The functions are pure:
they take DataFrames and return numbers or small frames, so they can be tested against
hand-computed fixtures without touching disk.

Two conventions run through the module and are easy to get wrong:

* **Behavioural metrics use purchase occasions** (one customer, one calendar day),
  never raw invoices. Several invoices minutes apart are one shopping visit.
* **Revenue metrics use net revenue** (cancellations subtract) unless a ratio needs a
  non-negative denominator, in which case gross revenue is used and named explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_TARGET = 0.8


@dataclass(frozen=True)
class RevenueSummary:
    net_revenue: float
    gross_revenue: float
    reversal_revenue: float
    order_count: int
    ordinary_order_count: int
    product_order_count: int
    product_revenue: float
    aov: float
    median_product_order_value: float
    identified_net_revenue: float
    unattributable_net_revenue: float


@dataclass(frozen=True)
class RepeatPurchase:
    buyers: int
    repeaters: int
    rate: float
    single_purchase_share: float


@dataclass(frozen=True)
class IntervalSummary:
    intervals: int
    customers_with_intervals: int
    mean_days: float
    median_days: float
    p75_days: float
    p90_days: float
    p95_days: float
    max_days: float


@dataclass(frozen=True)
class ReturnRates:
    unit_rate: float
    value_rate: float
    line_rate: float
    cancelled_units: float
    purchased_units: float
    cancelled_value: float
    gross_value: float
    cancelled_lines: int
    ordinary_lines: int


@dataclass(frozen=True)
class ParetoResult:
    n_items: int
    n_needed: int
    share_of_items: float
    target: float
    total: float


@dataclass(frozen=True)
class ConcentrationResult:
    pareto: ParetoResult
    gini: float
    negative_items: int
    negative_total: float


def is_reversal(df: pd.DataFrame) -> pd.Series:
    """Cancellations ('C') and bad-debt adjustments ('A') are reversals, not sales."""
    return df["is_cancellation"] | df["is_adjustment"]


def ordinary_orders(orders: pd.DataFrame) -> pd.DataFrame:
    return orders.loc[~is_reversal(orders)]


def purchase_occasions(orders: pd.DataFrame, *, require_customer: bool = True) -> pd.DataFrame:
    """One row per (customer, calendar day) with at least one non-reversal order."""
    frame = ordinary_orders(orders)
    if require_customer:
        frame = frame.loc[frame["customer_id"].notna()]
    frame = frame.assign(occasion_date=frame["invoice_date"].dt.normalize())
    columns = ["customer_id", "occasion_date"]
    return frame.loc[:, columns].drop_duplicates().reset_index(drop=True)


def revenue_summary(orders: pd.DataFrame) -> RevenueSummary:
    reversals = is_reversal(orders)
    ordinary = orders.loc[~reversals]
    product_orders = ordinary.loc[ordinary["product_count"] > 0]

    net = float(orders["net_revenue"].sum())
    gross = float(orders["gross_revenue"].sum())
    identified = float(orders.loc[orders["customer_id"].notna(), "net_revenue"].sum())
    product_revenue = float(product_orders["product_revenue"].sum())
    product_order_count = len(product_orders)

    return RevenueSummary(
        net_revenue=net,
        gross_revenue=gross,
        reversal_revenue=net - gross,
        order_count=len(orders),
        ordinary_order_count=len(ordinary),
        product_order_count=product_order_count,
        product_revenue=product_revenue,
        aov=product_revenue / product_order_count if product_order_count else float("nan"),
        median_product_order_value=(
            float(product_orders["product_revenue"].median())
            if product_order_count
            else float("nan")
        ),
        identified_net_revenue=identified,
        unattributable_net_revenue=net - identified,
    )


def arpu(orders: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Net revenue per identified customer with >=1 purchase occasion in [start, end).

    Both numerator and denominator use the same customer set, so a refund by a customer
    who did not buy in the window cannot drag ARPU down without also being counted.
    """
    windowed = orders.loc[(orders["invoice_date"] >= start) & (orders["invoice_date"] < end)]
    buyers = purchase_occasions(windowed)["customer_id"].unique()
    if len(buyers) == 0:
        return float("nan")
    attributable = windowed.loc[windowed["customer_id"].isin(buyers), "net_revenue"].sum()
    return float(attributable) / len(buyers)


def repeat_purchase_rate(
    orders: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> RepeatPurchase:
    windowed = orders
    if start is not None:
        windowed = windowed.loc[windowed["invoice_date"] >= start]
    if end is not None:
        windowed = windowed.loc[windowed["invoice_date"] < end]

    occasions = purchase_occasions(windowed)
    frequency = occasions.groupby("customer_id").size()
    buyers = int(len(frequency))
    repeaters = int((frequency >= 2).sum())
    rate = repeaters / buyers if buyers else float("nan")
    return RepeatPurchase(
        buyers=buyers,
        repeaters=repeaters,
        rate=rate,
        single_purchase_share=1.0 - rate if buyers else float("nan"),
    )


def purchase_intervals(orders: pd.DataFrame) -> pd.DataFrame:
    """Gap in days between a customer's consecutive purchase occasions.

    Only repeat buyers contribute; a customer with one occasion has no interval. That
    censoring shortens the observed distribution, which is why callers must report the
    single-purchase share next to any interval statistic.
    """
    occasions = purchase_occasions(orders).sort_values(["customer_id", "occasion_date"])
    occasions = occasions.assign(
        gap_days=occasions.groupby("customer_id")["occasion_date"].diff().dt.days
    )
    gaps = occasions.dropna(subset=["gap_days"]).loc[:, ["customer_id", "gap_days"]]
    gaps = gaps.assign(gap_days=gaps["gap_days"].astype("int64"))
    return gaps.reset_index(drop=True)


def interval_summary(intervals: pd.DataFrame) -> IntervalSummary:
    if intervals.empty:
        nan = float("nan")
        return IntervalSummary(0, 0, nan, nan, nan, nan, nan, nan)
    gaps = intervals["gap_days"].astype(float)
    return IntervalSummary(
        intervals=int(len(gaps)),
        customers_with_intervals=int(intervals["customer_id"].nunique()),
        mean_days=float(gaps.mean()),
        median_days=float(gaps.quantile(0.50)),
        p75_days=float(gaps.quantile(0.75)),
        p90_days=float(gaps.quantile(0.90)),
        p95_days=float(gaps.quantile(0.95)),
        max_days=float(gaps.max()),
    )


def window_coverage(
    intervals: pd.DataFrame, windows: tuple[int, ...] = (30, 60, 90, 120, 150, 180)
) -> pd.DataFrame:
    """Share of observed repeat intervals that fall inside each candidate churn window."""
    gaps = intervals["gap_days"]
    return pd.DataFrame(
        {
            "window_days": list(windows),
            "share_covered": [float((gaps <= w).mean()) for w in windows],
            "intervals": [int(len(gaps))] * len(windows),
        }
    )


def churn_sensitivity(
    orders: pd.DataFrame,
    windows: tuple[int, ...] = (60, 90, 120, 150),
    *,
    min_history_months: int = 2,
) -> pd.DataFrame:
    """Churn rate at each candidate window on the same monthly grid as customer_snapshots.

    A snapshot is eligible when at least `min_history_months` of data precede it and its
    window ends inside the observed data. A customer is eligible when they have at least
    one purchase occasion in [snapshot - W, snapshot), and churned when they have none
    inside [snapshot, snapshot + W). The lookback is set equal to W, mirroring the
    production choice of a 90-day label window with a 90-day activity check.
    """
    if any(window <= 0 for window in windows):
        raise ValueError("churn windows must be positive")
    occasions = purchase_occasions(orders)
    if occasions.empty:
        raise ValueError("no purchase occasions to measure churn from")

    dates = occasions["occasion_date"]
    last_date = orders["invoice_date"].max().normalize()
    first_snapshot = (dates.min().to_period("M") + min_history_months).to_timestamp()
    last_snapshot = last_date.to_period("M").to_timestamp()
    snapshots = pd.date_range(first_snapshot, last_snapshot, freq="MS")

    rows = []
    for window in windows:
        horizon = pd.Timedelta(days=int(window))
        lookback = horizon
        for snapshot in snapshots:
            if snapshot + horizon > last_date:
                continue
            eligible_mask = (dates >= snapshot - lookback) & (dates < snapshot)
            eligible = set(occasions.loc[eligible_mask, "customer_id"])
            if not eligible:
                continue
            active_after = set(
                occasions.loc[(dates >= snapshot) & (dates < snapshot + horizon), "customer_id"]
            )
            churned = len(eligible - active_after)
            rows.append(
                {
                    "window_days": int(window),
                    "snapshot_date": snapshot,
                    "eligible_customers": len(eligible),
                    "churned_customers": churned,
                    "churn_rate": churned / len(eligible),
                }
            )
    return pd.DataFrame(rows)


def return_rates(lines: pd.DataFrame) -> ReturnRates:
    reversals = is_reversal(lines)
    cancellations = lines["is_cancellation"]
    ordinary = lines.loc[~reversals]

    cancelled_units = float(-lines.loc[cancellations, "quantity"].sum())
    purchased_units = float(ordinary["quantity"].sum())
    cancelled_value = float(-lines.loc[cancellations, "revenue"].sum())
    gross_value = float(ordinary["revenue"].sum())
    cancelled_lines = int(cancellations.sum())
    ordinary_lines = int((~reversals).sum())

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else float("nan")

    return ReturnRates(
        unit_rate=ratio(cancelled_units, purchased_units),
        value_rate=ratio(cancelled_value, gross_value),
        line_rate=ratio(cancelled_lines, ordinary_lines),
        cancelled_units=cancelled_units,
        purchased_units=purchased_units,
        cancelled_value=cancelled_value,
        gross_value=gross_value,
        cancelled_lines=cancelled_lines,
        ordinary_lines=ordinary_lines,
    )


def product_return_rates(lines: pd.DataFrame) -> pd.DataFrame:
    """Unit, value and line return rates per product code.

    Products that only ever appear on cancellation lines keep NaN rates rather than a
    fabricated zero: there were no purchases to return.
    """
    product = lines.loc[lines["is_product"]].copy()
    ordinary = product.loc[~is_reversal(product)]
    cancelled = product.loc[product["is_cancellation"]]

    purchased = ordinary.groupby("stock_code").agg(
        purchased_units=("quantity", "sum"),
        gross_value=("revenue", "sum"),
        ordinary_lines=("stock_code", "size"),
    )
    cancelled_sums = cancelled.groupby("stock_code").agg(
        cancelled_units=("quantity", "sum"),
        cancelled_value=("revenue", "sum"),
        cancelled_lines=("stock_code", "size"),
    )
    # A C line reverses a charge, so its signed value is negated to state the refund.
    cancelled_sums["cancelled_units"] *= -1
    cancelled_sums["cancelled_value"] *= -1

    out = purchased.join(cancelled_sums, how="outer")
    out[["purchased_units", "gross_value", "ordinary_lines"]] = out[
        ["purchased_units", "gross_value", "ordinary_lines"]
    ].fillna(0.0)
    out[["cancelled_units", "cancelled_value", "cancelled_lines"]] = out[
        ["cancelled_units", "cancelled_value", "cancelled_lines"]
    ].fillna(0.0)

    out["unit_return_rate"] = out["cancelled_units"] / out["purchased_units"].replace(0, np.nan)
    out["value_return_rate"] = out["cancelled_value"] / out["gross_value"].replace(0, np.nan)
    out["line_return_rate"] = out["cancelled_lines"] / out["ordinary_lines"].replace(0, np.nan)
    out["net_revenue"] = out["gross_value"] - out["cancelled_value"]
    return out.sort_values("cancelled_value", ascending=False)


def pareto_share(values: pd.Series | np.ndarray, target: float = DEFAULT_TARGET) -> ParetoResult:
    """Smallest number of items whose cumulative share reaches `target`."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("pareto_share needs at least one value")
    if np.isnan(array).any():
        raise ValueError("pareto_share does not accept NaN values")
    total = float(array.sum())
    if total <= 0:
        raise ValueError(f"pareto_share needs a positive total, got {total}")

    ordered = np.sort(array)[::-1]
    cumulative = np.cumsum(ordered) / total
    # The crossing happens inside the positive prefix, so this search is monotone.
    n_needed = int(np.searchsorted(cumulative, target, side="left")) + 1
    return ParetoResult(
        n_items=int(array.size),
        n_needed=n_needed,
        share_of_items=n_needed / array.size,
        target=target,
        total=total,
    )


def gini(values: pd.Series | np.ndarray, *, clip_negative: bool = True) -> float:
    """Gini coefficient of a revenue distribution.

    Negative net revenue is clipped to zero by default: Gini is only defined for a
    non-negative distribution, and dropping those customers instead would silently
    shrink the population and understate concentration. The caller reports how many
    values were clipped.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("gini needs at least one value")
    if np.isnan(array).any():
        raise ValueError("gini does not accept NaN values")
    if (array < 0).any():
        if not clip_negative:
            raise ValueError("gini requires non-negative values when clip_negative=False")
        array = np.clip(array, 0.0, None)

    total = float(array.sum())
    if total <= 0:
        return 0.0
    ordered = np.sort(array)
    ranks = np.arange(1, ordered.size + 1)
    coefficient = 2.0 * float((ranks * ordered).sum()) / (ordered.size * total)
    coefficient -= (ordered.size + 1) / ordered.size
    return float(min(max(coefficient, 0.0), 1.0))


def customer_concentration(
    orders: pd.DataFrame, target: float = DEFAULT_TARGET
) -> ConcentrationResult:
    per_customer = (
        orders.loc[orders["customer_id"].notna()].groupby("customer_id")["net_revenue"].sum()
    )
    negative = per_customer.loc[per_customer < 0]
    return ConcentrationResult(
        pareto=pareto_share(per_customer, target),
        gini=gini(per_customer),
        negative_items=int(len(negative)),
        negative_total=float(negative.sum()),
    )


def product_concentration(
    products: pd.DataFrame, target: float = DEFAULT_TARGET
) -> ConcentrationResult:
    per_product = products.loc[products["is_product"]].set_index("stock_code")["net_revenue"]
    negative = per_product.loc[per_product < 0]
    return ConcentrationResult(
        pareto=pareto_share(per_product, target),
        gini=gini(per_product),
        negative_items=int(len(negative)),
        negative_total=float(negative.sum()),
    )
