"""Run the Phase 3 analysis end to end and write tables, figures and a JSON summary.

This is the single place numbers are produced for the notebooks and the key-findings
document. It reads only the gold warehouse, so a rebuild of gold regenerates every
downstream artefact. All figures use the headless Agg backend; there is no display
dependency.

Outputs:
    reports/tables/*.csv    the numbers behind every chart
    reports/figures/*.png   the charts themselves
    reports/analysis.json   headline metrics, so prose can quote exact values

Usage:
    uv run python -m hub.analytics.report
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hub.analytics import baskets, cohorts, metrics, seasonality
from hub.analytics.data import GOLD_DIR, PROJECT_ROOT, load_gold

REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

CHURN_WINDOW = 90
CHURN_WINDOWS = (30, 60, 90, 120, 150, 180)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
    }
)


def _monthly_revenue(orders: pd.DataFrame) -> pd.DataFrame:
    frame = orders.assign(month=orders["invoice_date"].dt.to_period("M").dt.to_timestamp())
    grouped = frame.groupby("month").agg(
        net_revenue=("net_revenue", "sum"),
        gross_revenue=("gross_revenue", "sum"),
        product_revenue=("product_revenue", "sum"),
        orders=("invoice_no", "nunique"),
    )
    occasions = metrics.purchase_occasions(frame).assign(
        month=lambda df: df["occasion_date"].dt.to_period("M").dt.to_timestamp()
    )
    buyers = occasions.groupby("month")["customer_id"].nunique()
    grouped["buyers"] = buyers.reindex(grouped.index, fill_value=0)
    grouped.index.name = "month"

    first, last, _ = cohorts.complete_month_bounds(orders["invoice_date"])
    complete = [cohorts._month_start(ordinal) for ordinal in range(first, last + 1)]
    grouped["is_complete_month"] = grouped.index.isin(complete)
    return grouped.reset_index()


def _customer_pareto(orders: pd.DataFrame) -> pd.DataFrame:
    per_customer = (
        orders.loc[orders["customer_id"].notna()].groupby("customer_id")["net_revenue"].sum()
    )
    ordered = per_customer.sort_values(ascending=False).to_numpy(dtype=float)
    cumulative = np.cumsum(ordered) / ordered.sum()
    return pd.DataFrame(
        {
            "customers": np.arange(1, ordered.size + 1),
            "customer_share": np.arange(1, ordered.size + 1) / ordered.size,
            "cumulative_revenue_share": cumulative,
        }
    )


def _churn_tables(
    orders: pd.DataFrame, snapshots: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    complete = snapshots.loc[
        snapshots["is_label_window_complete"] & snapshots["was_active_in_prior_window"]
    ]
    by_snapshot = (
        complete.groupby("snapshot_date")
        .agg(customers=("customer_id", "size"), churned=("churned", "sum"))
        .reset_index()
    )
    by_snapshot["churn_rate"] = by_snapshot["churned"] / by_snapshot["customers"]

    sensitivity = metrics.churn_sensitivity(orders, windows=CHURN_WINDOWS)
    by_window = (
        sensitivity.groupby("window_days")
        .agg(
            snapshots=("snapshot_date", "size"),
            eligible_customers=("eligible_customers", "sum"),
            churned_customers=("churned_customers", "sum"),
            min_rate=("churn_rate", "min"),
            max_rate=("churn_rate", "max"),
        )
        .reset_index()
    )
    by_window["pooled_rate"] = by_window["churned_customers"] / by_window["eligible_customers"]

    pooled = float(complete["churned"].mean())
    summary = {
        "window_days": CHURN_WINDOW,
        "pooled_rate": pooled,
        "snapshots_used": int(by_snapshot["snapshot_date"].nunique()),
        "customer_rows_used": int(len(complete)),
        "by_window": {
            int(row.window_days): {
                "pooled_rate": float(row.pooled_rate),
                "min_rate": float(row.min_rate),
                "max_rate": float(row.max_rate),
                "snapshots": int(row.snapshots),
            }
            for row in by_window.itertuples()
        },
    }
    return by_snapshot, by_window, summary


def _product_tables(lines: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    returns = metrics.product_return_rates(lines)
    named = returns.join(products.set_index("stock_code")["description"], how="left")
    named.index.name = "stock_code"
    return_rates = named.sort_values("cancelled_value", ascending=False)

    concentration = metrics.product_concentration(products)
    product_mix = (
        products.loc[products["is_product"]]
        .sort_values("net_revenue", ascending=False)
        .loc[:, ["stock_code", "description", "net_revenue", "line_count", "units"]]
    )
    total = float(product_mix["net_revenue"].sum())
    product_mix["revenue_share"] = product_mix["net_revenue"] / total
    product_mix["cumulative_share"] = product_mix["revenue_share"].cumsum()

    summary = {
        "gini": concentration.gini,
        "negative_products": concentration.negative_items,
        "negative_products_revenue": concentration.negative_total,
        "pareto": asdict(concentration.pareto),
    }
    return return_rates, summary | {"mix": product_mix}


def _basket_summary(result: baskets.BasketResult, rules: pd.DataFrame) -> dict[str, Any]:
    material = rules.loc[rules["support_pair"] >= 0.005]
    top = material.iloc[0] if len(material) else None
    return {
        **asdict(result.summary),
        "rules": int(len(rules)),
        "rules_at_0_5pct_support": int(len(material)),
        "min_item_support": result.min_item_support,
        "min_pair_support": result.min_pair_support,
        "min_confidence": result.min_confidence,
        "top_rule_at_0_5pct_support": (
            {
                "antecedent": str(top["antecedent"]),
                "consequent": str(top["consequent"]),
                "support_pair": float(top["support_pair"]),
                "pair_baskets": int(top["pair_baskets"]),
                "confidence": float(top["confidence"]),
                "lift": float(top["lift"]),
            }
            if top is not None
            else None
        ),
    }


def _named_rules(rules: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    descriptions = products.set_index("stock_code")["description"]
    named = rules.copy()
    named["antecedent_description"] = named["antecedent"].map(descriptions)
    named["consequent_description"] = named["consequent"].map(descriptions)
    return named


def _seasonality_summary(
    daily: pd.Series,
    day_of_week: pd.DataFrame,
    q4: pd.DataFrame,
    stl: seasonality.STLResult,
) -> dict[str, Any]:
    zero_share = day_of_week["zero_days"] / day_of_week["n_days"]
    low_activity = day_of_week.loc[zero_share >= 0.9]
    best = day_of_week.drop(index=low_activity.index)["mean_revenue"].idxmax()
    return {
        "days": int(len(daily)),
        "zero_revenue_days": int((daily == 0).sum()),
        "best_weekday": str(best),
        "low_activity_weekdays": [
            {
                "weekday": str(name),
                "zero_days": int(row.zero_days),
                "days": int(row.n_days),
                "zero_share": float(row.zero_days / row.n_days),
                "mean_revenue": float(row.mean_revenue),
            }
            for name, row in low_activity.iterrows()
        ],
        "q4_lift": {
            int(year): {
                "lift": None if pd.isna(row.lift) else float(row.lift),
                "q4_complete": bool(row.q4_complete),
                "q4_days": int(row.q4_days),
            }
            for year, row in q4.iterrows()
        },
        "stl": {
            "period": stl.period,
            "seasonal_strength": stl.seasonal_strength,
            "trend_strength": stl.trend_strength,
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    return value.isoformat() if isinstance(value, pd.Timestamp) else value


def run_analysis(gold_dir: Path = GOLD_DIR, reports_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    tables_dir, figures_dir = reports_dir / "tables", reports_dir / "figures"
    for directory in (tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    gold = load_gold(gold_dir)
    orders = gold["fact_orders"]
    lines = gold["fact_order_lines"]
    customers = gold["dim_customer"]
    products = gold["dim_product"]
    snapshots = gold["customer_snapshots"]

    revenue = metrics.revenue_summary(orders)
    repeat = metrics.repeat_purchase_rate(orders)
    intervals = metrics.purchase_intervals(orders)
    interval_summary = metrics.interval_summary(intervals)
    coverage = metrics.window_coverage(intervals)
    returns = metrics.return_rates(lines)

    monthly = _monthly_revenue(orders)
    monthly.to_csv(tables_dir / "monthly_revenue.csv", index=False)
    pareto = _customer_pareto(orders)
    pareto.to_csv(tables_dir / "customer_pareto.csv", index=False)

    interval_summary_row = pd.DataFrame([asdict(interval_summary)])
    interval_summary_row.to_csv(tables_dir / "purchase_intervals.csv", index=False)
    coverage.to_csv(tables_dir / "churn_window_coverage.csv", index=False)

    churn_by_snapshot, churn_by_window, churn = _churn_tables(orders, snapshots)
    churn_by_snapshot.to_csv(tables_dir / "churn_by_snapshot.csv", index=False)
    churn_by_window.to_csv(tables_dir / "churn_window_sensitivity.csv", index=False)

    cohort = cohorts.build_cohorts(orders)
    cohort.retention.to_csv(tables_dir / "cohort_retention.csv")
    cohort.revenue_retention.to_csv(tables_dir / "cohort_revenue_retention.csv")
    cohort.cumulative_revenue_per_customer.to_csv(
        tables_dir / "cohort_cumulative_revenue_per_customer.csv"
    )
    cohort.cohort_sizes.to_csv(tables_dir / "cohort_sizes.csv")

    return_rates, product_info = _product_tables(lines, products)
    return_rates.to_csv(tables_dir / "product_return_rates.csv")
    product_info["mix"].to_csv(tables_dir / "product_revenue_mix.csv", index=False)

    customer_concentration = metrics.customer_concentration(orders)

    daily = seasonality.daily_revenue(orders)
    day_of_week = seasonality.day_of_week_effect(daily)
    hour = seasonality.hour_effect(orders)
    q4 = seasonality.q4_lift(daily)
    stl = seasonality.stl_decompose(daily)
    day_of_week.to_csv(tables_dir / "day_of_week.csv")
    hour.to_csv(tables_dir / "hour_effect.csv")
    q4.to_csv(tables_dir / "q4_lift.csv")
    stl.frame.to_csv(tables_dir / "stl_components.csv")

    basket_result = baskets.analyse_baskets(gold_dir / "fact_order_lines.parquet")
    named_rules = _named_rules(basket_result.rules, products)
    named_rules.to_csv(tables_dir / "association_rules.csv", index=False)

    _write_figures(
        monthly=monthly,
        cohort=cohort,
        pareto=pareto,
        customer_concentration=customer_concentration,
        day_of_week=day_of_week,
        hour=hour,
        q4=q4,
        stl=stl,
        churn_by_snapshot=churn_by_snapshot,
        churn_by_window=churn_by_window,
        return_rates=return_rates,
        rules=named_rules,
        figures_dir=figures_dir,
    )

    end = orders["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    analysis = {
        "data_window": {
            "first_order": orders["invoice_date"].min(),
            "last_order": orders["invoice_date"].max(),
            "orders": int(len(orders)),
            "lines": int(len(lines)),
            "customers": int(len(customers)),
            "products": int(products["is_product"].sum()),
            "non_product_codes": int((~products["is_product"]).sum()),
        },
        "revenue": asdict(revenue),
        "arpu": metrics.arpu(orders, orders["invoice_date"].min().normalize(), end),
        "repeat_purchase": asdict(repeat),
        "intervals": asdict(interval_summary),
        "window_coverage": coverage.to_dict(orient="records"),
        "churn": churn,
        "returns": asdict(returns),
        "customer_concentration": {
            **asdict(customer_concentration),
            "share_of_customers_for_80pct": customer_concentration.pareto.share_of_items,
        },
        "product_concentration": {
            "gini": product_info["gini"],
            "pareto": product_info["pareto"],
            "negative_products": product_info["negative_products"],
            "negative_products_revenue": product_info["negative_products_revenue"],
        },
        "cohorts": {
            "first_complete_month": cohort.first_complete_month,
            "last_complete_month": cohort.last_complete_month,
            "left_censored_cohort": cohort.left_censored_cohort,
            "excluded_partial_months": list(cohort.excluded_partial_months),
            "customers_outside_window": cohort.customers_outside_window,
            "cohort_sizes": cohort.cohort_sizes.to_dict(),
            "month_1_retention_by_cohort": (
                cohort.retention[1].dropna().to_dict() if 1 in cohort.retention.columns else {}
            ),
        },
        "seasonality": _seasonality_summary(daily, day_of_week, q4, stl),
        "baskets": _basket_summary(basket_result, named_rules),
    }
    analysis = _json_safe(analysis)
    (reports_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return analysis


def _write_figures(
    *,
    monthly: pd.DataFrame,
    cohort: cohorts.CohortTables,
    pareto: pd.DataFrame,
    customer_concentration: metrics.ConcentrationResult,
    day_of_week: pd.DataFrame,
    hour: pd.DataFrame,
    q4: pd.DataFrame,
    stl: seasonality.STLResult,
    churn_by_snapshot: pd.DataFrame,
    churn_by_window: pd.DataFrame,
    return_rates: pd.DataFrame,
    rules: pd.DataFrame,
    figures_dir: Path,
) -> None:
    _plot_monthly(monthly, figures_dir / "monthly_revenue.png")
    _plot_cohort_retention(cohort, figures_dir / "cohort_retention.png")
    _plot_cohort_curves(cohort, figures_dir / "cohort_revenue_retention.png")
    _plot_cohort_curves(
        cohort,
        figures_dir / "cohort_cumulative_revenue.png",
        frame=cohort.cumulative_revenue_per_customer,
        ylabel="Cumulative net revenue per customer (GBP)",
        title="Cumulative revenue per customer by acquisition cohort",
        y_zero=True,
    )
    _plot_pareto(pareto, customer_concentration, figures_dir / "customer_pareto.png")
    _plot_day_of_week(day_of_week, figures_dir / "day_of_week.png")
    _plot_hour(hour, figures_dir / "hour_effect.png")
    _plot_q4(q4, figures_dir / "q4_lift.png")
    _plot_stl(stl, figures_dir / "stl_decomposition.png")
    _plot_churn(churn_by_snapshot, churn_by_window, figures_dir / "churn.png")
    _plot_product_returns(return_rates, figures_dir / "product_returns.png")
    _plot_rules(rules, figures_dir / "basket_rules.png")


def _plot_monthly(monthly: pd.DataFrame, path: Path) -> None:
    complete = monthly.loc[monthly["is_complete_month"]]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(complete["month"], complete["gross_revenue"] / 1e3, label="Gross", color="#7c8db5")
    ax.plot(complete["month"], complete["net_revenue"] / 1e3, label="Net", color="#1f4e79")
    ax.set_title("Monthly revenue (complete months only)")
    ax.set_ylabel("GBP thousands")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _cohort_selection(cohort: cohorts.CohortTables) -> pd.Index:
    # Skip the left-censored first cohort and sample every third month so the chart
    # stays readable without cherry-picking.
    return cohort.revenue_retention.index[1::3]


def _plot_cohort_retention(cohort: cohorts.CohortTables, path: Path) -> None:
    values = np.ma.masked_invalid(cohort.retention.to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(8.5, 6))
    image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(cohort.retention.columns)))
    ax.set_xticklabels(cohort.retention.columns)
    ax.set_yticks(range(len(cohort.retention.index)))
    ax.set_yticklabels([label.strftime("%Y-%m") for label in cohort.retention.index])
    ax.set_xlabel("Months since acquisition")
    ax.set_ylabel("Acquisition cohort")
    ax.set_title("Monthly retention: share of cohort buying again (NaN = not yet observable)")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="share active", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_cohort_curves(
    cohort: cohorts.CohortTables,
    path: Path,
    *,
    frame: pd.DataFrame | None = None,
    ylabel: str = "Revenue retention (month 0 = 1.0)",
    title: str = "Revenue retention by acquisition cohort",
    y_zero: bool = False,
) -> None:
    data = cohort.revenue_retention if frame is None else frame
    fig, ax = plt.subplots(figsize=(9, 4.7))
    for label in _cohort_selection(cohort):
        series = data.loc[label].dropna()
        ax.plot(
            series.index, series.to_numpy(), marker="o", markersize=3, label=label.strftime("%Y-%m")
        )
    ax.set_title(title)
    ax.set_xlabel("Months since acquisition")
    ax.set_ylabel(ylabel)
    if y_zero:
        ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncols=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_pareto(
    pareto: pd.DataFrame, concentration: metrics.ConcentrationResult, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        pareto["customer_share"] * 100, pareto["cumulative_revenue_share"] * 100, color="#1f4e79"
    )
    crossing = concentration.pareto.share_of_items * 100
    ax.axvline(crossing, color="#c0504d", linestyle="--", linewidth=1)
    ax.axhline(80, color="#c0504d", linestyle="--", linewidth=1)
    ax.annotate(
        f"top {crossing:.1f}% of customers = 80% of revenue",
        xy=(crossing, 80),
        xytext=(crossing + 4, 68),
        fontsize=8,
        color="#c0504d",
    )
    ax.set_title(f"Customer revenue concentration (Gini {concentration.gini:.3f})")
    ax.set_xlabel("Share of customers, highest revenue first (%)")
    ax.set_ylabel("Cumulative share of net revenue (%)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_day_of_week(day_of_week: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = ["#c0504d" if name == "Saturday" else "#1f4e79" for name in day_of_week.index]
    ax.bar(day_of_week.index, day_of_week["mean_revenue"], color=colors)
    for index, row in enumerate(day_of_week.itertuples()):
        if row.zero_days / row.n_days >= 0.5:
            ax.annotate(
                f"closed\n{row.zero_days}/{row.n_days} days",
                xy=(index, 0),
                xytext=(index, row.mean_revenue + max(day_of_week["mean_revenue"]) * 0.05),
                ha="center",
                fontsize=7,
                color="#c0504d",
            )
    ax.set_title("Mean daily net revenue by weekday (zero-revenue days included)")
    ax.set_ylabel("GBP per calendar day")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_hour(hour: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(hour.index, hour["revenue_share"] * 100, color="#1f4e79")
    ax.set_title("Share of ordinary revenue by invoice hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("% of revenue")
    ax.set_xticks(range(int(hour.index.min()), int(hour.index.max()) + 1))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_q4(q4: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(q4))
    width = 0.36
    ax.bar(x - width / 2, q4["non_q4_mean"] / 1e3, width, label="Non-Q4", color="#7c8db5")
    ax.bar(x + width / 2, q4["q4_mean"] / 1e3, width, label="Q4", color="#c0504d")
    for index, row in enumerate(q4.itertuples()):
        label = "lift n/a" if pd.isna(row.lift) else f"{row.lift:.2f}x"
        if not row.q4_complete:
            label += "\n(partial)"
        ax.annotate(
            label,
            xy=(index, max(row.q4_mean, row.non_q4_mean) / 1e3),
            xytext=(index, max(row.q4_mean, row.non_q4_mean) / 1e3 + 0.4),
            ha="center",
            fontsize=7,
        )
    ax.set_xticks(x, [str(year) for year in q4.index])
    ax.set_title("Q4 vs non-Q4 mean daily revenue")
    ax.set_ylabel("GBP thousands per day")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_stl(stl: seasonality.STLResult, path: Path) -> None:
    frame = stl.frame
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7), sharex=True)
    axes[0].plot(frame.index, frame["observed"] / 1e3, color="#7c8db5", linewidth=0.5)
    axes[0].plot(frame.index, frame["trend"] / 1e3, color="#1f4e79", linewidth=1.5)
    axes[0].set_ylabel("GBP thousands")
    axes[0].set_title(
        f"STL decomposition (seasonal strength {stl.seasonal_strength:.2f}, "
        f"trend strength {stl.trend_strength:.2f})"
    )
    axes[1].plot(frame.index, frame["seasonal"], color="#c0504d", linewidth=0.6)
    axes[1].set_ylabel("Seasonal (GBP)")
    axes[2].plot(frame.index, frame["resid"], color="#4f6228", linewidth=0.5)
    axes[2].set_ylabel("Residual (GBP)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_churn(by_snapshot: pd.DataFrame, by_window: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].fill_between(
        by_window["window_days"],
        by_window["min_rate"] * 100,
        by_window["max_rate"] * 100,
        color="#7c8db5",
        alpha=0.25,
        label="min-max across snapshots",
    )
    axes[0].plot(
        by_window["window_days"], by_window["pooled_rate"] * 100, marker="o", color="#1f4e79"
    )
    axes[0].axvline(CHURN_WINDOW, color="#c0504d", linestyle="--", linewidth=1)
    axes[0].set_title("Churn sensitivity to window length")
    axes[0].set_xlabel("Window W (days)")
    axes[0].set_ylabel("Churn rate (%)")
    axes[0].legend(frameon=False, fontsize=7)

    axes[1].plot(
        by_snapshot["snapshot_date"],
        by_snapshot["churn_rate"] * 100,
        marker="o",
        markersize=3,
        color="#c0504d",
    )
    axes[1].set_title(f"90-day churn by snapshot ({len(by_snapshot)} complete windows)")
    axes[1].set_ylabel("Churn rate (%)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_product_returns(return_rates: pd.DataFrame, path: Path) -> None:
    top = return_rates.loc[return_rates["cancelled_value"] > 0].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = [
        f"{code}  {str(description)[:28]}" if pd.notna(description) else str(code)
        for code, description in zip(top.index, top["description"], strict=True)
    ]
    ax.barh(labels, top["cancelled_value"], color="#c0504d")
    for index, (value, rate) in enumerate(
        zip(top["cancelled_value"], top["value_return_rate"], strict=True)
    ):
        text = f" GBP {value:,.0f}" + ("" if pd.isna(rate) else f" | {rate:.1%} of value")
        ax.annotate(text, xy=(value, index), xytext=(value, index), va="center", fontsize=7)
    ax.set_title("Largest sources of cancelled value (top 10 products)")
    ax.set_xlabel("Cancelled value (GBP)")
    ax.set_xlim(0, float(top["cancelled_value"].max()) * 1.65)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _short(description: object, limit: int = 18) -> str:
    if pd.isna(description):
        return ""
    text = str(description)
    return text if len(text) <= limit else text[: limit - 2] + ".."


def _plot_rules(rules: pd.DataFrame, path: Path) -> None:
    material = rules.loc[rules["support_pair"] >= 0.005].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = [
        f"{a} {_short(desc_a)} -> {b} {_short(desc_b)}"
        for a, b, desc_a, desc_b in zip(
            material["antecedent"],
            material["consequent"],
            material["antecedent_description"],
            material["consequent_description"],
            strict=True,
        )
    ]
    ax.barh(labels, material["lift"], color="#1f4e79")
    for index, (lift, support) in enumerate(
        zip(material["lift"], material["support_pair"], strict=True)
    ):
        ax.annotate(
            f"{lift:.2f}x | {support:.1%} of baskets",
            xy=(lift, index),
            xytext=(lift, index),
            va="center",
            fontsize=7,
        )
    ax.set_title("Strongest product associations (pairs in >= 0.5% of baskets)")
    ax.set_xlabel("Lift")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, default=GOLD_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    analysis = run_analysis(args.gold_dir, args.reports_dir)
    revenue = analysis["revenue"]
    print(
        f"[analysis] net revenue GBP {revenue['net_revenue']:,.2f} over "
        f"{revenue['order_count']:,} orders; "
        f"top {analysis['customer_concentration']['share_of_customers_for_80pct']:.1%} of "
        f"customers drive 80% of revenue"
    )
    print(
        f"[analysis] 90-day churn {analysis['churn']['pooled_rate']:.1%}; "
        f"{analysis['baskets']['rules_emitted']:,} association rules; "
        f"wrote {args.reports_dir}"
    )


if __name__ == "__main__":
    main()
