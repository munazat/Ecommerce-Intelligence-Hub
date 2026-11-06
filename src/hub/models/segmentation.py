"""Customer segmentation on the gold point-in-time snapshot.

Segmentation uses one frozen snapshot per customer and never pools rows: a customer
appears once, and every feature is history-only by construction of
``customer_snapshots``. The population is every customer who has actually bought —
refund-only accounts are excluded because frequency, monetary value and product
diversity are all zero for them and they would form an artefact cluster rather than a
market segment.

Churn is reported only among customers active in the 90 days before the snapshot.
``churned`` is undefined for lapsed customers (they were already inactive), so quoting a
pooled churn rate per segment would invent a label the metrics dictionary does not
define. Lapsed segments are described by reactivation and forward revenue instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
ALGORITHMS = ("kmeans", "gmm", "ward")

FEATURES = (
    "recency_days",
    "frequency",
    "gross_revenue",
    "tenure_days",
    "avg_days_between_orders",
    "return_rate",
    "distinct_products",
)

# Everything except return_rate is a count or a duration with exact zeros, where Box-Cox
# is undefined and log1p is the standard stabiliser. Return rate is already bounded in
# [0, 1], so it is only standardised.
LOG_FEATURES = tuple(column for column in FEATURES if column != "return_rate")

FEATURE_LABELS = {
    "recency_days": "Recency (days)",
    "frequency": "Purchase invoices",
    "gross_revenue": "Lifetime gross revenue",
    "tenure_days": "Tenure (days)",
    "avg_days_between_orders": "Avg days between orders",
    "return_rate": "Return rate",
    "distinct_products": "Distinct products",
}


@dataclass(frozen=True)
class PreparedMatrix:
    """The transformed feature space plus the skew evidence used to justify it."""

    frame: pd.DataFrame
    transformed: pd.DataFrame
    scaled: np.ndarray
    skew_before: pd.Series
    skew_after: pd.Series


@dataclass(frozen=True)
class ModelFit:
    algorithm: str
    k: int
    labels: np.ndarray
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    bic: float | None
    min_cluster_share: float


@dataclass(frozen=True)
class StabilityResult:
    algorithm: str
    k: int
    n_bootstrap: int
    scores: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores))

    @property
    def median(self) -> float:
        return float(np.median(self.scores))

    @property
    def p10(self) -> float:
        return float(np.percentile(self.scores, 10))

    @property
    def minimum(self) -> float:
        return float(np.min(self.scores))

    @property
    def unstable_runs(self) -> int:
        return int(np.sum(np.asarray(self.scores) < 0.5))


def latest_complete_snapshot(snapshots: pd.DataFrame) -> pd.Timestamp:
    """The most recent snapshot whose 90-day label window is fully observed."""
    complete = snapshots.loc[snapshots["is_label_window_complete"], "snapshot_date"]
    if complete.empty:
        raise ValueError("No snapshot has a complete label window.")
    return pd.Timestamp(complete.max())


def build_segment_frame(
    snapshots: pd.DataFrame, snapshot_date: pd.Timestamp | str | None = None
) -> pd.DataFrame:
    """One row per buying customer at the chosen snapshot.

    Single-purchase customers have no observed inter-purchase interval. Imputing it with
    tenure is the least surprising choice: both are elapsed time since the first purchase,
    and the ``inter_purchase_imputed`` flag keeps the assumption visible downstream.
    """
    chosen = (
        pd.Timestamp(snapshot_date)
        if snapshot_date is not None
        else latest_complete_snapshot(snapshots)
    )
    frame = snapshots.loc[snapshots["snapshot_date"] == chosen].copy()
    if frame.empty:
        raise ValueError(f"No customers at snapshot {chosen.date()}.")

    n_before = len(frame)
    frame = frame.loc[frame["frequency"] > 0].copy()
    refund_only = n_before - len(frame)

    imputed = frame["avg_days_between_orders"].isna()
    frame["inter_purchase_imputed"] = imputed
    frame["avg_days_between_orders"] = frame["avg_days_between_orders"].fillna(frame["tenure_days"])
    frame = frame.reset_index(drop=True)
    frame.attrs["snapshot_date"] = chosen
    frame.attrs["refund_only_excluded"] = int(refund_only)
    frame.attrs["single_purchase_customers"] = int(imputed.sum())
    return frame


def prepare_matrix(frame: pd.DataFrame) -> PreparedMatrix:
    """Log-stabilise the heavy-tailed features and standardise every column."""
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing segmentation features: {', '.join(missing)}")

    raw = frame.loc[:, list(FEATURES)].astype(float)
    if raw.isna().any().any():
        raise ValueError("Segmentation features contain nulls after imputation.")

    transformed = raw.copy()
    for column in LOG_FEATURES:
        if (transformed[column] < 0).any():
            raise ValueError(f"{column} contains negative values and cannot be log1p-transformed")
        transformed[column] = np.log1p(transformed[column])

    scaled = StandardScaler().fit_transform(transformed.to_numpy())
    return PreparedMatrix(
        frame=frame,
        transformed=transformed,
        scaled=scaled,
        skew_before=raw.skew(),
        skew_after=transformed.skew(),
    )


def _fit_labels(
    scaled: np.ndarray, algorithm: str, k: int, random_state: int
) -> tuple[np.ndarray, float | None]:
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        return model.fit_predict(scaled), None
    if algorithm == "gmm":
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=5,
            random_state=random_state,
        )
        labels = model.fit_predict(scaled)
        return labels, float(model.bic(scaled))
    if algorithm == "ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return model.fit_predict(scaled), None
    raise ValueError(f"Unknown algorithm {algorithm!r}; choose from {ALGORITHMS}.")


def fit_model(
    scaled: np.ndarray,
    algorithm: str,
    k: int,
    *,
    random_state: int = RANDOM_STATE,
) -> ModelFit:
    if k < 2:
        raise ValueError("k must be at least 2.")
    if k >= len(scaled):
        raise ValueError("k must be smaller than the number of customers.")

    labels, bic = _fit_labels(scaled, algorithm, k, random_state)
    counts = np.bincount(labels)
    return ModelFit(
        algorithm=algorithm,
        k=k,
        labels=labels,
        silhouette=float(silhouette_score(scaled, labels)),
        davies_bouldin=float(davies_bouldin_score(scaled, labels)),
        calinski_harabasz=float(calinski_harabasz_score(scaled, labels)),
        bic=bic,
        min_cluster_share=float(counts.min() / counts.sum()),
    )


def compare_models(
    scaled: np.ndarray,
    *,
    k_values: range | tuple[int, ...] = range(2, 9),
    algorithms: tuple[str, ...] = ALGORITHMS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Fit every algorithm at every k and return the selection evidence, not just the winner."""
    rows: list[dict[str, Any]] = []
    for algorithm in algorithms:
        for k in k_values:
            fit = fit_model(scaled, algorithm, k, random_state=random_state)
            rows.append(
                {
                    "algorithm": fit.algorithm,
                    "k": fit.k,
                    "silhouette": fit.silhouette,
                    "davies_bouldin": fit.davies_bouldin,
                    "calinski_harabasz": fit.calinski_harabasz,
                    "bic": fit.bic,
                    "min_cluster_share": fit.min_cluster_share,
                }
            )
    return pd.DataFrame(rows)


def select_model(
    comparison: pd.DataFrame,
    *,
    min_k: int = 3,
    min_cluster_share: float = 0.02,
) -> tuple[str, int]:
    """Pick the best silhouette among actionably sized solutions.

    A two-cluster split often wins on silhouette, but "active vs lapsed" is one business
    fact, not a segmentation strategy. The constraint is explicit and the full curve is
    still reported, including the k that was rejected.
    """
    eligible = comparison.loc[
        (comparison["k"] >= min_k) & (comparison["min_cluster_share"] >= min_cluster_share)
    ]
    if eligible.empty:
        raise ValueError("No candidate satisfies the minimum k and cluster-share constraints.")

    preference = {name: rank for rank, name in enumerate(ALGORITHMS)}
    ranked = eligible.assign(
        algorithm_rank=eligible["algorithm"].map(preference),
    ).sort_values(
        ["silhouette", "k", "algorithm_rank"],
        ascending=[False, True, True],
    )
    best = ranked.iloc[0]
    return str(best["algorithm"]), int(best["k"])


def bootstrap_stability(
    scaled: np.ndarray,
    algorithm: str,
    k: int,
    *,
    n_bootstrap: int = 100,
    random_state: int = RANDOM_STATE,
) -> StabilityResult:
    """Refit on bootstrap samples and score agreement with the full fit by ARI.

    Resampled indices are de-duplicated so every algorithm, including Ward (which has no
    out-of-sample predict), is scored on the same footing.
    """
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")

    base_labels, _ = _fit_labels(scaled, algorithm, k, random_state)
    rng = np.random.default_rng(random_state)
    n_rows = len(scaled)
    scores: list[float] = []
    for _ in range(n_bootstrap):
        indices = np.unique(rng.integers(0, n_rows, n_rows))
        labels, _ = _fit_labels(scaled[indices], algorithm, k, random_state)
        scores.append(float(adjusted_rand_score(base_labels[indices], labels)))
    return StabilityResult(algorithm=algorithm, k=k, n_bootstrap=n_bootstrap, scores=tuple(scores))


def _quantile_score(values: pd.Series, quantiles: int, *, higher_is_better: bool) -> pd.Series:
    # Ranks break ties deterministically, so qcut always produces balanced quintiles.
    ranks = values.rank(method="first")
    bins = pd.qcut(ranks, q=quantiles, labels=False, duplicates="drop").astype(int) + 1
    score = bins if higher_is_better else (bins.max() + 1 - bins)
    return score.astype(int)


def rfm_scores(frame: pd.DataFrame, *, quantiles: int = 5) -> pd.DataFrame:
    """Classic quintile RFM, the rule-based baseline the clustering has to beat."""
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2.")

    scores = pd.DataFrame(index=frame.index)
    scores["customer_id"] = frame["customer_id"]
    scores["r_score"] = _quantile_score(frame["recency_days"], quantiles, higher_is_better=False)
    scores["f_score"] = _quantile_score(frame["frequency"], quantiles, higher_is_better=True)
    scores["m_score"] = _quantile_score(frame["gross_revenue"], quantiles, higher_is_better=True)
    scores["rfm_score"] = scores[["r_score", "f_score", "m_score"]].sum(axis=1)
    scores["rfm_segment"] = scores.apply(_rule_segment, axis=1)
    return scores


def _rule_segment(row: pd.Series) -> str:  # noqa: PLR0911 - decision cascade
    r, f, m = int(row["r_score"]), int(row["f_score"]), int(row["m_score"])
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 4 and f <= 2:
        return "New / promising"
    if r >= 3 and f >= 3 and m >= 3:
        return "Loyal"
    if r <= 2 and f >= 4 and m >= 4:
        return "At risk — high value"
    if r <= 2 and f >= 3:
        return "At risk"
    if r <= 2:
        return "Hibernating"
    if f <= 2:
        return "Occasional"
    return "Core"


def population_thresholds(frame: pd.DataFrame) -> dict[str, float]:
    """Quartile boundaries the naming rules read, exposed so reports can quote them."""
    return {
        "recency_p25": float(frame["recency_days"].quantile(0.25)),
        "recency_p75": float(frame["recency_days"].quantile(0.75)),
        "frequency_p75": float(frame["frequency"].quantile(0.75)),
        "gross_p25": float(frame["gross_revenue"].quantile(0.25)),
        "gross_p75": float(frame["gross_revenue"].quantile(0.75)),
        "tenure_p25": float(frame["tenure_days"].quantile(0.25)),
    }


def name_segments(profile: pd.DataFrame, frame: pd.DataFrame) -> dict[int, str]:
    """Name clusters from their measured profile, not from the brief's wish list.

    The rules are deliberately simple and inspectable: every name can be traced to a
    median crossing a population quartile. Duplicate archetypes are disambiguated by
    relative value so the name mapping stays one-to-one.
    """
    thresholds = population_thresholds(frame)
    names: dict[int, str] = {}
    for row in profile.itertuples():
        high_value = row.median_gross_revenue >= thresholds["gross_p75"]
        low_value = row.median_gross_revenue <= thresholds["gross_p25"]
        value_band = "high" if high_value else "low" if low_value else "mid"
        frequent = row.median_frequency >= thresholds["frequency_p75"]
        recent = row.median_recency_days <= thresholds["recency_p25"]
        stale = row.median_recency_days >= thresholds["recency_p75"]
        new = row.median_tenure_days <= thresholds["tenure_p25"]
        active = row.active_share >= 0.5
        cooling = row.active_share >= 0.15

        if row.median_return_rate >= 0.25:
            name = "Returns-heavy accounts"
        elif not active:
            name = f"Cooling {value_band} value" if cooling else f"Lapsed {value_band} value"
        elif frequent and high_value and recent:
            name = "Champions"
        elif frequent and high_value:
            name = "High-value regulars"
        elif frequent and recent:
            name = "Loyal regulars"
        elif new and not frequent:
            name = "New / one-time"
        elif stale:
            name = "Cooling active"
        elif high_value:
            name = "High-value occasional"
        elif low_value:
            name = "Low-value active"
        else:
            name = "Steady core"
        names[int(row.cluster)] = name

    return _deduplicate_names(names, profile)


def _deduplicate_names(names: dict[int, str], profile: pd.DataFrame) -> dict[int, str]:
    by_value = (
        profile.assign(cluster=pd.to_numeric(profile.index))
        .sort_values("median_gross_revenue", ascending=False)["cluster"]
        .astype(int)
        .tolist()
    )
    taken: dict[str, int] = {}
    for cluster in by_value:
        name = names[cluster]
        if name not in taken:
            taken[name] = 1
            continue
        taken[name] += 1
        suffix = "higher value" if taken[name] == 2 else f"value rank {taken[name]}"
        names[cluster] = f"{name} — {suffix}"
    return names


def profile_segments(
    frame: pd.DataFrame,
    labels: np.ndarray,
    names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Per-segment size, value, activity and churn evidence."""
    labelled = frame.copy()
    labelled["cluster"] = labels
    labelled["segment"] = (
        labelled["cluster"].map(names) if names is not None else labelled["cluster"]
    )
    labelled["churned_active"] = labelled["churned"] & labelled["was_active_in_prior_window"]

    grouped = labelled.groupby("segment", sort=False)
    total_gross = float(labelled["gross_revenue"].sum())
    total_future = float(labelled["future_revenue"].sum())

    profile = pd.DataFrame(
        {
            "cluster": grouped["cluster"].first(),
            "customers": grouped.size(),
            "share_of_customers": grouped.size() / len(labelled),
            "active_customers": grouped["was_active_in_prior_window"].sum(),
            "active_share": grouped["was_active_in_prior_window"].mean(),
            "churned_active_customers": grouped["churned_active"].sum(),
            "gross_revenue": grouped["gross_revenue"].sum(),
            "gross_revenue_share": grouped["gross_revenue"].sum() / total_gross,
            "future_revenue": grouped["future_revenue"].sum(),
            "future_revenue_share": grouped["future_revenue"].sum() / total_future,
            "future_revenue_per_customer": grouped["future_revenue"].mean(),
            "future_active_rate": grouped["is_active"].mean(),
            "return_rate_mean": grouped["return_rate"].mean(),
        }
    )
    profile["churn_rate_among_active"] = profile["churned_active_customers"] / profile[
        "active_customers"
    ].replace(0, np.nan)
    for column in FEATURES:
        profile[f"median_{column}"] = grouped[column].median()
    profile["median_net_revenue"] = grouped["net_revenue"].median()
    profile["median_future_revenue"] = grouped["future_revenue"].median()
    profile.index.name = "segment"
    return profile.sort_values("gross_revenue", ascending=False)


@dataclass(frozen=True)
class Strategy:
    action: str
    trigger: str
    channel: str
    uplift: float
    uplift_basis: str
    cost_per_customer: float


# Every number here is an assumption, not a measurement. They are stated in the report so
# the expected-impact column can be challenged instead of being taken as a forecast.
STRATEGIES: tuple[tuple[str, Strategy], ...] = (
    (
        "champions",
        Strategy(
            action="Protect the account: named contact, early access, no blanket discount.",
            trigger="Service issue, order gap above 1.5x usual interval.",
            channel="Account management",
            uplift=0.10,
            uplift_basis="Assumed 10% lift on realised 90-day revenue",
            cost_per_customer=25.0,
        ),
    ),
    (
        "high-value regulars",
        Strategy(
            action="Grow the account: category expansion and priority service, no discount.",
            trigger="Order gap above 1.25x usual interval.",
            channel="Email + account management",
            uplift=0.08,
            uplift_basis="Assumed 8% lift on realised 90-day revenue",
            cost_per_customer=10.0,
        ),
    ),
    (
        "loyal regulars",
        Strategy(
            action="Grow basket: cross-sell adjacent categories and a tiered loyalty reward.",
            trigger="After every second purchase.",
            channel="Email",
            uplift=0.05,
            uplift_basis="Assumed 5% lift on realised 90-day revenue",
            cost_per_customer=2.0,
        ),
    ),
    (
        "high-value occasional",
        Strategy(
            action="Increase cadence: replenishment reminder at 0.8x the customer's interval.",
            trigger="At 0.8x the customer's average days between orders.",
            channel="Email",
            uplift=0.05,
            uplift_basis="Assumed 5% lift on realised 90-day revenue",
            cost_per_customer=1.0,
        ),
    ),
    (
        "steady core",
        Strategy(
            action="Nurture: category education and seasonal prompts, no incentive.",
            trigger="Quarterly.",
            channel="Email",
            uplift=0.03,
            uplift_basis="Assumed 3% lift on realised 90-day revenue",
            cost_per_customer=0.5,
        ),
    ),
    (
        "new / one-time",
        Strategy(
            action="Convert: second-purchase incentive inside 30 days.",
            trigger="14 days after the first purchase.",
            channel="Email + voucher",
            uplift=0.05,
            uplift_basis="Assumed 5% conversion of one-time buyers into a second purchase",
            cost_per_customer=3.0,
        ),
    ),
    (
        "cooling",
        Strategy(
            action="Reactivate before lapse: personalised reminder and a time-limited offer.",
            trigger="Recency above 1.25x the customer's usual interval.",
            channel="Email",
            uplift=0.04,
            uplift_basis="Assumed 4% lift on realised 90-day revenue",
            cost_per_customer=1.0,
        ),
    ),
    (
        "returns-heavy accounts",
        Strategy(
            action="Remediate first: audit refund reasons and fix the product or service issue.",
            trigger="Return rate at or above 25%.",
            channel="Service recovery",
            uplift=0.0,
            uplift_basis="No campaign uplift is assumed until the return pattern is explained",
            cost_per_customer=5.0,
        ),
    ),
    (
        "low-value active",
        Strategy(
            action="Low-cost automation only; do not discount a low-margin base.",
            trigger="Seasonal.",
            channel="Automated email",
            uplift=0.01,
            uplift_basis="Assumed 1% lift on realised 90-day revenue",
            cost_per_customer=0.2,
        ),
    ),
    (
        "lapsed high value",
        Strategy(
            action="Win back with human outreach and an account-level offer.",
            trigger="90 days after the last purchase.",
            channel="Phone / account management",
            uplift=0.03,
            uplift_basis="Assumed 3% reactivation, valued at realised forward revenue",
            cost_per_customer=15.0,
        ),
    ),
    (
        "lapsed mid value",
        Strategy(
            action="Win back with an automated sequence and a modest incentive.",
            trigger="90 days after the last purchase.",
            channel="Email",
            uplift=0.015,
            uplift_basis="Assumed 1.5% reactivation, valued at realised forward revenue",
            cost_per_customer=1.0,
        ),
    ),
    (
        "lapsed low value",
        Strategy(
            action="Suppress from paid channels; email only, no incentive.",
            trigger="Quarterly.",
            channel="Email",
            uplift=0.005,
            uplift_basis="Assumed 0.5% reactivation, valued at realised forward revenue",
            cost_per_customer=0.05,
        ),
    ),
)


def _strategy_for(segment: str) -> Strategy:
    lower = segment.lower()
    for key, strategy in STRATEGIES:
        if lower.startswith(key):
            return strategy
    raise KeyError(f"No strategy mapped to segment {segment!r}.")


def strategy_matrix(profile: pd.DataFrame) -> pd.DataFrame:
    """One actionable row per segment, with the assumption behind every pound."""
    rows: list[dict[str, Any]] = []
    for segment, row in profile.iterrows():
        strategy = _strategy_for(str(segment))
        incremental = float(row["future_revenue"]) * strategy.uplift
        cost = float(row["customers"]) * strategy.cost_per_customer
        rows.append(
            {
                "segment": segment,
                "customers": int(row["customers"]),
                "action": strategy.action,
                "trigger": strategy.trigger,
                "channel": strategy.channel,
                "assumed_uplift": strategy.uplift,
                "uplift_basis": strategy.uplift_basis,
                "incremental_revenue_estimate": incremental,
                "cost_per_customer": strategy.cost_per_customer,
                "campaign_cost_estimate": cost,
                "net_estimate": incremental - cost,
                "customers_per_pound": (
                    float(row["customers"]) / strategy.cost_per_customer
                    if strategy.cost_per_customer > 0
                    else np.inf
                ),
            }
        )
    frame = pd.DataFrame(rows).set_index("segment")
    return frame.sort_values("net_estimate", ascending=False)
