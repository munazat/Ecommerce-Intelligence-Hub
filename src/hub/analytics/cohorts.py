"""Monthly acquisition cohorts: retention and revenue-retention matrices.

The two ways a cohort analysis lies are handled explicitly here:

* **Left censoring** — the first complete month contains customers who were already
  active before the data begins, so it is not a true acquisition cohort. It is flagged
  and excluded from headline claims.
* **Right censoring** — a cohort's month-k cell is only observable once that month has
  fully elapsed inside the data. Unobservable cells are NaN, never zero. A partial final
  month (e.g. data ending on the 9th) is dropped entirely rather than reported as a
  collapse in activity.

Month arithmetic is done on integer month ordinals (`year * 12 + month - 1`) rather than
Period offsets, which keeps the observable/non-observable test a simple integer
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hub.analytics.metrics import purchase_occasions


@dataclass(frozen=True)
class CohortTables:
    retention: pd.DataFrame
    revenue_per_customer: pd.DataFrame
    revenue_retention: pd.DataFrame
    cumulative_revenue_per_customer: pd.DataFrame
    cohort_sizes: pd.Series
    first_complete_month: pd.Timestamp
    last_complete_month: pd.Timestamp
    left_censored_cohort: pd.Timestamp
    excluded_partial_months: tuple[pd.Timestamp, ...]
    customers_outside_window: int


def _month_ordinal(dates: pd.Series) -> pd.Series:
    return dates.dt.year.astype("int64") * 12 + dates.dt.month.astype("int64") - 1


def _month_start(ordinal: int) -> pd.Timestamp:
    return pd.Timestamp(year=ordinal // 12, month=ordinal % 12 + 1, day=1)


def _month_end(ordinal: int) -> pd.Timestamp:
    return _month_start(ordinal + 1) - pd.Timedelta(days=1)


def complete_month_bounds(dates: pd.Series) -> tuple[int, int, tuple[pd.Timestamp, ...]]:
    """First and last month fully covered by the data, plus the partial months dropped."""
    start = dates.min().normalize()
    end = dates.max().normalize()
    first = int(_month_ordinal(pd.Series([start])).iloc[0])
    last = int(_month_ordinal(pd.Series([end])).iloc[0])

    excluded = []
    if start != _month_start(first):
        excluded.append(_month_start(first))
        first += 1
    if end != _month_end(last):
        excluded.append(_month_start(last))
        last -= 1
    if first > last:
        raise ValueError("data does not cover any complete calendar month")
    return first, last, tuple(excluded)


def build_cohorts(orders: pd.DataFrame) -> CohortTables:
    occasions = purchase_occasions(orders)
    if occasions.empty:
        raise ValueError("no purchase occasions to build cohorts from")

    occasions = occasions.assign(month_ord=_month_ordinal(occasions["occasion_date"]))
    first, last, excluded = complete_month_bounds(occasions["occasion_date"])
    columns = list(range(last - first + 1))

    first_purchase = occasions.groupby("customer_id")["month_ord"].min()
    outside = int(((first_purchase < first) | (first_purchase > last)).sum())
    in_window = first_purchase.loc[first_purchase.between(first, last)]
    cohort_sizes_ord = in_window.value_counts().sort_index()

    occasions = occasions.loc[occasions["month_ord"].between(first, last)].copy()
    occasions["cohort_ord"] = occasions["customer_id"].map(in_window)
    occasions = occasions.dropna(subset=["cohort_ord"])
    occasions["cohort_ord"] = occasions["cohort_ord"].astype("int64")
    occasions["months_since"] = occasions["month_ord"] - occasions["cohort_ord"]

    rows = list(cohort_sizes_ord.index)
    mask = pd.DataFrame(
        [[cohort + month <= last for month in columns] for cohort in rows],
        index=rows,
        columns=columns,
        dtype=bool,
    )

    active = (
        occasions.groupby(["cohort_ord", "months_since"])["customer_id"]
        .nunique()
        .unstack("months_since")
    )
    counts = active.reindex(index=rows, columns=columns).fillna(0.0)
    retention = counts.div(cohort_sizes_ord, axis=0).where(mask)

    revenue = _cohort_revenue(
        orders,
        in_window,
        first=first,
        last=last,
        rows=rows,
        columns=columns,
        mask=mask,
    )
    revenue_per_customer = revenue.div(cohort_sizes_ord, axis=0).where(mask)
    base = revenue_per_customer[0].replace(0.0, np.nan)
    revenue_retention = (
        revenue_per_customer.div(base, axis=0).where(mask).replace([np.inf, -np.inf], np.nan)
    )
    cumulative = revenue_per_customer.cumsum(axis=1, skipna=True)

    return CohortTables(
        retention=_label(retention),
        revenue_per_customer=_label(revenue_per_customer),
        revenue_retention=_label(revenue_retention),
        cumulative_revenue_per_customer=_label(cumulative),
        cohort_sizes=_label_sizes(cohort_sizes_ord),
        first_complete_month=_month_start(first),
        last_complete_month=_month_start(last),
        left_censored_cohort=_month_start(first),
        excluded_partial_months=excluded,
        customers_outside_window=outside,
    )


def _cohort_revenue(
    orders: pd.DataFrame,
    cohort_map: pd.Series,
    *,
    first: int,
    last: int,
    rows: list[int],
    columns: list[int],
    mask: pd.DataFrame,
) -> pd.DataFrame:
    frame = orders.loc[orders["customer_id"].notna()].copy()
    frame = frame.assign(month_ord=_month_ordinal(frame["invoice_date"]))
    frame = frame.loc[frame["month_ord"].between(first, last)]
    frame["cohort_ord"] = frame["customer_id"].map(cohort_map)
    frame = frame.dropna(subset=["cohort_ord"])
    frame["cohort_ord"] = frame["cohort_ord"].astype("int64")
    frame["months_since"] = frame["month_ord"] - frame["cohort_ord"]

    revenue = (
        frame.groupby(["cohort_ord", "months_since"])["net_revenue"].sum().unstack("months_since")
    )
    revenue = revenue.reindex(index=rows, columns=columns).fillna(0.0)
    return revenue.where(mask)


def _label(frame: pd.DataFrame) -> pd.DataFrame:
    labelled = frame.copy()
    labelled.index = pd.DatetimeIndex(
        [_month_start(int(ordinal)) for ordinal in labelled.index], name="cohort_month"
    )
    labelled.columns = pd.Index([int(month) for month in labelled.columns], name="months_since")
    return labelled


def _label_sizes(sizes: pd.Series) -> pd.Series:
    labelled = sizes.copy()
    labelled.index = pd.DatetimeIndex(
        [_month_start(int(ordinal)) for ordinal in labelled.index], name="cohort_month"
    )
    labelled.name = "customers"
    return labelled
