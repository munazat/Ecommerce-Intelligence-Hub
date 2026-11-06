"""Run Phase 4 customer segmentation end to end.

Outputs:
    reports/tables/segmentation_*.csv   selection, stability and segment evidence
    reports/figures/segment_*.png       model selection, stability, profile and strategy charts
    reports/segments.json               every number quoted in the notebook and the docs

The statistical optimum is not the shipped solution. k=4 wins silhouette and
Davies-Bouldin, but it merges two populations that need opposite treatments: 1,949
cooling mid-value accounts (19.6% of revenue) and 1,623 lapsed one-time buyers (3.9% of
revenue). k=5 separates them and isolates a returns-heavy account group for remediation,
at a documented silhouette cost. The full comparison is still written out, so the
judgement can be checked rather than trusted.

Usage:
    uv run python -m hub.analytics.segment_report
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from hub.analytics.data import GOLD_DIR, PROJECT_ROOT, load_table
from hub.models import segmentation as seg

REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

CHOSEN_ALGORITHM = "kmeans"
CHOSEN_K = 5
N_BOOTSTRAP = 100
SELECTION_NOTE = (
    "k=4 is the statistical optimum (silhouette 0.335, Davies-Bouldin 1.051), but its "
    "2,571-customer lapsed block holds 1,605 of the 1,623 lapsed one-time buyers "
    "alongside 966 of the 1,949 cooling mid-value accounts, and the remaining 857 "
    "cooling accounts sit in the high-value cluster: the suppression list and the "
    "reactivation opportunity are not separable. k=5 separates them and isolates a "
    "returns-heavy account group, costing 0.054 silhouette. k=6 raises Davies-Bouldin "
    "by 14% and splits the cooling base at a boundary with no operational meaning, so "
    "the search stops at 5."
)

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

COLORS = (
    "#1f4e79",
    "#c0504d",
    "#4f6228",
    "#7c8db5",
    "#bf8f00",
    "#5f6b7a",
    "#9467bd",
    "#2ca02c",
)


def _json_safe(value: Any) -> Any:  # noqa: PLR0911 - one guard per JSON edge case
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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _alternative_composition(
    scaled: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
    algorithm: str,
    k: int,
) -> pd.DataFrame:
    """Show which shipped segments the statistical optimum would merge, by name."""
    alternative = seg.fit_model(scaled, algorithm, k)
    table = pd.crosstab(
        pd.Series(alternative.labels, name="alternative_cluster"),
        pd.Series(labels, name="segment").map(names),
    )
    table["total"] = table.sum(axis=1)
    return table


def _composition_evidence(
    scaled: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
    statistical_algorithm: str,
    statistical_k: int,
    *,
    tables_dir: Path,
) -> pd.DataFrame | None:
    if (statistical_algorithm, statistical_k) == (CHOSEN_ALGORITHM, CHOSEN_K):
        return None
    composition = _alternative_composition(
        scaled, labels, names, statistical_algorithm, statistical_k
    )
    composition.to_csv(tables_dir / "statistical_choice_composition.csv")
    return composition


def _rule_based_profile(frame: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    categories = scores["rfm_segment"].astype("category")
    names = dict(enumerate(categories.cat.categories))
    codes = categories.cat.codes.to_numpy()
    return seg.profile_segments(frame, codes, names)


def _approach_summary(profile: pd.DataFrame) -> dict[str, Any]:
    churn = profile["churn_rate_among_active"].dropna()
    forward = profile["future_revenue_per_customer"]
    return {
        "segments": int(len(profile)),
        "smallest_segment_share": float(profile["share_of_customers"].min()),
        "largest_segment_share": float(profile["share_of_customers"].max()),
        "churn_rate_range": [float(churn.min()), float(churn.max())],
        "churn_rate_spread": float(churn.max() - churn.min()),
        "forward_revenue_per_customer_range": [float(forward.min()), float(forward.max())],
        "revenue_share_range": [
            float(profile["gross_revenue_share"].min()),
            float(profile["gross_revenue_share"].max()),
        ],
    }


def _segment_zscores(profile: pd.DataFrame, prepared: seg.PreparedMatrix) -> pd.DataFrame:
    medians = profile.loc[:, [f"median_{column}" for column in seg.FEATURES]].copy()
    medians.columns = list(seg.FEATURES)
    for column in seg.LOG_FEATURES:
        medians[column] = np.log1p(medians[column])
    population = prepared.transformed
    return (medians - population.mean()) / population.std(ddof=0)


def _percentile_of_median(profile: pd.DataFrame, prepared: seg.PreparedMatrix) -> pd.DataFrame:
    medians = profile.loc[:, [f"median_{column}" for column in seg.FEATURES]].copy()
    medians.columns = list(seg.FEATURES)
    for column in seg.LOG_FEATURES:
        medians[column] = np.log1p(medians[column])
    population = prepared.transformed
    # Mid-rank handles the zero-inflated features honestly: a segment with the same return
    # rate as the 56% of customers who never refunded is at the middle of that mass, not
    # at the 0th percentile (which strict "<" would claim).
    return medians.apply(
        lambda column: column.map(
            lambda value: (
                (population[column.name] < value).mean()
                + 0.5 * (population[column.name] == value).mean()
            )
        )
    )


def _plot_model_selection(comparison: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for index, algorithm in enumerate(seg.ALGORITHMS):
        subset = comparison.loc[comparison["algorithm"] == algorithm]
        axes[0].plot(
            subset["k"], subset["silhouette"], marker="o", color=COLORS[index], label=algorithm
        )
        axes[1].plot(
            subset["k"],
            subset["davies_bouldin"],
            marker="o",
            color=COLORS[index],
            label=algorithm,
        )
    best = comparison.sort_values("silhouette", ascending=False).iloc[0]
    axes[0].scatter([best["k"]], [best["silhouette"]], marker="*", s=150, color="#c0504d", zorder=5)
    axes[0].annotate(
        "statistical optimum",
        xy=(best["k"], best["silhouette"]),
        xytext=(best["k"] + 0.25, best["silhouette"] - 0.012),
        fontsize=7,
        color="#c0504d",
    )
    for axis in axes:
        axis.axvline(CHOSEN_K, color="#c0504d", linestyle="--", linewidth=1)
        axis.set_xlabel("Clusters (k)")
        axis.legend(frameon=False, fontsize=7)
    axes[0].set_title("Silhouette by k (higher is better)")
    axes[0].set_ylabel("Silhouette")
    axes[1].set_title("Davies-Bouldin by k (lower is better)")
    axes[1].set_ylabel("Davies-Bouldin")
    fig.suptitle("Model selection: dashed line is the shipped k=5, not the statistical optimum")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_stability(stability: seg.StabilityResult, path: Path) -> None:
    scores = np.asarray(stability.scores)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(scores, bins=np.linspace(-1.0, 1.0, 41), color="#1f4e79", edgecolor="white")
    axes[0].axvline(0.5, color="#c0504d", linestyle="--", linewidth=1)
    axes[0].set_title(f"Bootstrap agreement with the full fit ({stability.n_bootstrap} refits)")
    axes[0].set_xlabel("Adjusted Rand Index")
    axes[0].set_ylabel("Refits")

    axes[1].bar(
        ["mean", "median", "10th pct", "worst"],
        [stability.mean, stability.median, stability.p10, stability.minimum],
        color=["#1f4e79", "#7c8db5", "#bf8f00", "#c0504d"],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"{stability.unstable_runs} of {stability.n_bootstrap} refits below ARI 0.5")
    axes[1].set_ylabel("Adjusted Rand Index")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_radar(zscores: pd.DataFrame, path: Path) -> None:
    labels = [seg.FEATURE_LABELS[column] for column in seg.FEATURES]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    limit = 2.5

    fig, ax = plt.subplots(figsize=(8.4, 7.6), subplot_kw={"polar": True})
    for index, (segment, row) in enumerate(zscores.iterrows()):
        values = row.loc[list(seg.FEATURES)].clip(-limit, limit).tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.6, color=COLORS[index % len(COLORS)], label=segment)
        ax.fill(angles, values, alpha=0.05, color=COLORS[index % len(COLORS)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis="x", pad=14)
    ax.set_ylim(-limit, limit)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_yticklabels(["-2", "-1", "0", "+1", "+2"], fontsize=7)
    ax.set_title(
        "Segment shape (z-score of the segment median vs the population)\n"
        f"values clipped at +/-{limit:g} for readability",
        pad=26,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_parallel(percentiles: pd.DataFrame, path: Path) -> None:
    labels = [seg.FEATURE_LABELS[column] for column in seg.FEATURES]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for index, (segment, row) in enumerate(percentiles.iterrows()):
        values = row.loc[list(seg.FEATURES)].to_numpy(dtype=float)
        ax.plot(
            x, values, marker="o", markersize=4, color=COLORS[index % len(COLORS)], label=segment
        )
        ax.annotate(
            segment,
            xy=(x[-1], values[-1]),
            xytext=(x[-1] + 0.08, values[-1]),
            fontsize=7,
            color=COLORS[index % len(COLORS)],
            va="center",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Population percentile of the segment median")
    ax.set_title("Parallel coordinates: where each segment sits in the customer distribution")
    ax.set_xlim(-0.2, len(labels) + 0.7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_segment_map(profile: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sizes = profile["customers"] / profile["customers"].max() * 900 + 60
    scatter = ax.scatter(
        profile["median_recency_days"],
        profile["median_gross_revenue"],
        s=sizes,
        c=profile["churn_rate_among_active"],
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.8,
        edgecolor="white",
        linewidth=1.2,
    )
    for segment, row in profile.iterrows():
        ax.annotate(
            f"{segment}\n{int(row['customers']):,} customers",
            xy=(row["median_recency_days"], row["median_gross_revenue"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Median days since last purchase (higher = more lapsed)")
    ax.set_ylabel("Median lifetime gross revenue, GBP (log scale)")
    ax.set_title("Segment map: bubble size is customer count, colour is 90-day churn among active")
    fig.colorbar(scatter, ax=ax, label="Churn among active customers", shrink=0.85)
    ax.margins(x=0.18, y=0.18)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_strategy(strategy: pd.DataFrame, path: Path) -> None:
    ordered = strategy.sort_values("net_estimate")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = ["#c0504d" if value < 0 else "#1f4e79" for value in ordered["net_estimate"]]
    ax.barh(ordered.index, ordered["net_estimate"], color=colors)
    for index, row in enumerate(ordered.itertuples()):
        ax.annotate(
            f" GBP {row.net_estimate:,.0f} | uplift {row.assumed_uplift:.1%}",
            xy=(row.net_estimate, index),
            xytext=(row.net_estimate, index),
            va="center",
            fontsize=7,
        )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Illustrative net value after assumed campaign cost (GBP)")
    ax.set_title("Strategy matrix: expected value by segment (assumptions in the table)")
    ax.margins(x=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_rule_agreement(crosstab: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5))
    values = crosstab.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=values.max())
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_xticklabels(crosstab.columns, rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_yticklabels(crosstab.index, fontsize=7)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if values[row, column] > 0:
                ax.text(
                    column,
                    row,
                    f"{values[row, column]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if values[row, column] > values.max() * 0.6 else "#333333",
                )
    ax.set_xlabel("Rule-based RFM segment")
    ax.set_ylabel("ML segment")
    ax.set_title("Where the two approaches agree: share of each ML segment by rule-based segment")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Share of ML segment", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_segmentation(gold_dir: Path = GOLD_DIR, reports_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    tables_dir, figures_dir = reports_dir / "tables", reports_dir / "figures"
    for directory in (tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    snapshots = load_table("customer_snapshots", gold_dir)
    frame = seg.build_segment_frame(snapshots)
    prepared = seg.prepare_matrix(frame)

    comparison = seg.compare_models(prepared.scaled)
    comparison.to_csv(tables_dir / "segmentation_model_comparison.csv", index=False)
    statistical_algorithm, statistical_k = seg.select_model(comparison)

    fit = seg.fit_model(prepared.scaled, CHOSEN_ALGORITHM, CHOSEN_K)
    stability = seg.bootstrap_stability(
        prepared.scaled,
        CHOSEN_ALGORITHM,
        CHOSEN_K,
        n_bootstrap=N_BOOTSTRAP,
    )
    pd.DataFrame(
        {
            "bootstrap": np.arange(1, stability.n_bootstrap + 1),
            "adjusted_rand_index": stability.scores,
        }
    ).to_csv(tables_dir / "segmentation_stability.csv", index=False)

    skew = pd.DataFrame(
        {
            "feature": list(seg.FEATURES),
            "skew_before": prepared.skew_before.reindex(seg.FEATURES).to_numpy(),
            "skew_after": prepared.skew_after.reindex(seg.FEATURES).to_numpy(),
        }
    )
    skew.to_csv(tables_dir / "segmentation_skew.csv", index=False)

    raw_profile = seg.profile_segments(frame, fit.labels)
    names = seg.name_segments(raw_profile, frame)
    profile = seg.profile_segments(frame, fit.labels, names)
    profile.to_csv(tables_dir / "segment_profiles.csv")

    composition = _composition_evidence(
        prepared.scaled,
        fit.labels,
        names,
        statistical_algorithm,
        statistical_k,
        tables_dir=tables_dir,
    )

    strategy = seg.strategy_matrix(profile)
    strategy.to_csv(tables_dir / "segment_strategy.csv")

    assignments = frame.loc[
        :,
        [
            "customer_id",
            "recency_days",
            "frequency",
            "gross_revenue",
            "tenure_days",
            "return_rate",
            "distinct_products",
            "was_active_in_prior_window",
            "churned",
            "future_revenue",
        ],
    ].copy()
    assignments["cluster"] = fit.labels
    assignments["segment"] = assignments["cluster"].map(names)
    assignments.to_csv(tables_dir / "segment_assignments.csv", index=False)

    scores = seg.rfm_scores(frame)
    rule_profile = _rule_based_profile(frame, scores)
    rule_profile.to_csv(tables_dir / "rfm_rule_profiles.csv")

    rule_categories = scores["rfm_segment"].astype("category")
    rule_codes = rule_categories.cat.codes.to_numpy()
    agreement = float(adjusted_rand_score(fit.labels, rule_codes))
    ml_segments = pd.Series(
        assignments["segment"].to_numpy(), index=scores.index, name="ml_segment"
    )
    crosstab = pd.crosstab(ml_segments, scores["rfm_segment"], normalize="index")
    crosstab.to_csv(tables_dir / "rfm_vs_clusters.csv")

    zscores = _segment_zscores(profile, prepared)
    percentiles = _percentile_of_median(profile, prepared)
    zscores.to_csv(tables_dir / "segment_zscores.csv")
    percentiles.to_csv(tables_dir / "segment_percentiles.csv")

    _plot_model_selection(comparison, figures_dir / "segmentation_model_selection.png")
    _plot_stability(stability, figures_dir / "segmentation_stability.png")
    _plot_radar(zscores, figures_dir / "segment_radar.png")
    _plot_parallel(percentiles, figures_dir / "segment_parallel_coordinates.png")
    _plot_segment_map(profile, figures_dir / "segment_map.png")
    _plot_strategy(strategy, figures_dir / "segment_strategy.png")
    _plot_rule_agreement(crosstab, figures_dir / "rfm_vs_clusters.png")

    statistical = comparison.loc[
        (comparison["algorithm"] == statistical_algorithm) & (comparison["k"] == statistical_k)
    ].iloc[0]
    chosen = comparison.loc[
        (comparison["algorithm"] == CHOSEN_ALGORITHM) & (comparison["k"] == CHOSEN_K)
    ].iloc[0]

    result = {
        "snapshot": {
            "date": frame.attrs["snapshot_date"],
            "label_horizon_days": 90,
            "customers": int(len(frame)),
            "active_customers": int(frame["was_active_in_prior_window"].sum()),
            "lapsed_customers": int((~frame["was_active_in_prior_window"]).sum()),
            "refund_only_excluded": frame.attrs["refund_only_excluded"],
            "single_purchase_customers": frame.attrs["single_purchase_customers"],
            "gross_revenue": float(frame["gross_revenue"].sum()),
            "future_revenue": float(frame["future_revenue"].sum()),
        },
        "features": {
            "used": list(seg.FEATURES),
            "log1p": list(seg.LOG_FEATURES),
            "skew": skew.to_dict(orient="records"),
            "naming_thresholds": seg.population_thresholds(frame),
            "note": (
                "Box-Cox is undefined at the exact zeros in recency, frequency, tenure, "
                "return rate and product counts; log1p stabilises the rest. Return rate "
                "is bounded in [0, 1] and is only standardised."
            ),
        },
        "model_comparison": comparison.to_dict(orient="records"),
        "statistical_choice": {
            "algorithm": statistical_algorithm,
            "k": statistical_k,
            "silhouette": float(statistical["silhouette"]),
            "davies_bouldin": float(statistical["davies_bouldin"]),
        },
        "statistical_choice_composition": (
            composition.reset_index().to_dict(orient="records") if composition is not None else None
        ),
        "chosen_model": {
            "algorithm": CHOSEN_ALGORITHM,
            "k": CHOSEN_K,
            "silhouette": float(chosen["silhouette"]),
            "davies_bouldin": float(chosen["davies_bouldin"]),
            "calinski_harabasz": float(chosen["calinski_harabasz"]),
            "min_cluster_share": float(chosen["min_cluster_share"]),
            "selection_note": SELECTION_NOTE,
        },
        "stability": {
            "n_bootstrap": stability.n_bootstrap,
            "mean_ari": stability.mean,
            "median_ari": stability.median,
            "p10_ari": stability.p10,
            "min_ari": stability.minimum,
            "unstable_runs": stability.unstable_runs,
            "scores": list(stability.scores),
        },
        "segments": profile.reset_index().to_dict(orient="records"),
        "strategy": strategy.reset_index().to_dict(orient="records"),
        "rfm_baseline": {
            "agreement_ari": agreement,
            "summary": _approach_summary(rule_profile),
            "profiles": rule_profile.reset_index().to_dict(orient="records"),
        },
        "ml_summary": _approach_summary(profile),
    }
    result = _json_safe(result)
    (reports_dir / "segments.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, default=GOLD_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    result = run_segmentation(args.gold_dir, args.reports_dir)
    chosen = result["chosen_model"]
    stability = result["stability"]
    print(
        f"[segmentation] {result['snapshot']['customers']:,} customers at "
        f"{result['snapshot']['date'][:10]}; {chosen['k']} clusters "
        f"({chosen['algorithm']}), silhouette {chosen['silhouette']:.3f}, "
        f"bootstrap ARI {stability['median_ari']:.3f}"
    )
    for segment in result["segments"]:
        print(
            f"  {segment['segment']:<26} {segment['customers']:>5,} customers "
            f"({segment['share_of_customers']:>5.1%}), "
            f"{segment['gross_revenue_share']:>5.1%} of lifetime revenue, "
            f"churn {segment['churn_rate_among_active']:.1%} among active"
        )
    print(f"[segmentation] wrote {args.reports_dir}")


if __name__ == "__main__":
    main()
