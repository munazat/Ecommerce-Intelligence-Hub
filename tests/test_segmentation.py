"""Segmentation tests: hand-computed RFM rules, population handling and clustering sanity.

The clustering metrics are not asserted against a golden number -- they depend on the
random start and would make the suite brittle. What is asserted is the thing that can
silently go wrong: the population (refund-only accounts must not enter), the imputation,
the direction of every RFM score, churn being measured only among active customers, and
one-to-one segment names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hub.analytics.segment_report import _alternative_composition
from hub.models import segmentation as seg

SNAPSHOT_COLUMNS = {
    "snapshot_date": pd.Timestamp("2011-09-01"),
    "customer_id": 1001,
    "label_horizon_days": 90,
    "first_invoice_date": pd.Timestamp("2010-01-01"),
    "last_invoice_date": pd.Timestamp("2011-08-01"),
    "recency_days": 31,
    "tenure_days": 600,
    "invoice_count": 5,
    "frequency": 5,
    "monetary": 100.0,
    "line_count": 20,
    "distinct_products": 30,
    "active_days": 5,
    "units": 50.0,
    "net_revenue": 500.0,
    "gross_revenue": 500.0,
    "cancellation_line_count": 0.0,
    "return_rate": 0.0,
    "avg_days_between_orders": 60.0,
    "was_active_in_prior_window": True,
    "future_invoices": 1,
    "future_orders": 1,
    "future_lines": 2,
    "future_units": 3.0,
    "future_revenue": 300.0,
    "is_active": True,
    "churned": False,
    "is_label_window_complete": True,
}


def _snapshots(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame([{**SNAPSHOT_COLUMNS, **row} for row in rows])
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    for column in ("first_invoice_date", "last_invoice_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def _blobs(
    centres: list[tuple[float, ...]], per_centre: int = 60, spread: float = 0.25, seed: int = 7
):
    rng = np.random.default_rng(seed)
    blocks = [rng.normal(centre, spread, size=(per_centre, len(centre))) for centre in centres]
    return np.vstack(blocks)


def test_latest_complete_snapshot_skips_censored_windows():
    snapshots = _snapshots(
        [
            {"snapshot_date": "2011-08-01"},
            {"snapshot_date": "2011-09-01"},
            {"snapshot_date": "2011-10-01", "is_label_window_complete": False},
            {"snapshot_date": "2011-11-01", "is_label_window_complete": False},
        ]
    )
    assert seg.latest_complete_snapshot(snapshots) == pd.Timestamp("2011-09-01")


def test_latest_complete_snapshot_rejects_all_censored():
    snapshots = _snapshots([{"snapshot_date": "2011-10-01", "is_label_window_complete": False}])
    with pytest.raises(ValueError, match="complete label window"):
        seg.latest_complete_snapshot(snapshots)


def test_build_segment_frame_excludes_refund_only_and_imputes_interval():
    snapshots = _snapshots(
        [
            {"customer_id": 1, "frequency": 3, "avg_days_between_orders": 20.0},
            {"customer_id": 2, "frequency": 1, "tenure_days": 100, "avg_days_between_orders": None},
            {"customer_id": 3, "frequency": 0, "gross_revenue": 0.0, "net_revenue": -50.0},
        ]
    )
    frame = seg.build_segment_frame(snapshots)

    assert frame["customer_id"].tolist() == [1, 2]
    assert frame.attrs["refund_only_excluded"] == 1
    assert frame.attrs["single_purchase_customers"] == 1
    assert bool(frame.loc[frame["customer_id"] == 2, "inter_purchase_imputed"].iloc[0]) is True
    # No interval was observed, so tenure is the only defensible elapsed-time proxy.
    assert frame.loc[frame["customer_id"] == 2, "avg_days_between_orders"].iloc[0] == 100
    assert frame.loc[:, list(seg.FEATURES)].notna().all().all()


def test_build_segment_frame_rejects_unknown_snapshot():
    snapshots = _snapshots([{"customer_id": 1}])
    with pytest.raises(ValueError, match="No customers at snapshot"):
        seg.build_segment_frame(snapshots, "2012-01-01")


def test_prepare_matrix_stabilises_skew_and_standardises():
    frame = _snapshots(
        [
            {
                "customer_id": 1,
                "recency_days": 1,
                "frequency": 1,
                "gross_revenue": 1.0,
                "tenure_days": 10,
                "avg_days_between_orders": 1.0,
                "return_rate": 0.0,
                "distinct_products": 1,
            },
            {
                "customer_id": 2,
                "recency_days": 10,
                "frequency": 5,
                "gross_revenue": 100.0,
                "tenure_days": 100,
                "avg_days_between_orders": 10.0,
                "return_rate": 0.0,
                "distinct_products": 10,
            },
            {
                "customer_id": 3,
                "recency_days": 20,
                "frequency": 10,
                "gross_revenue": 1000.0,
                "tenure_days": 200,
                "avg_days_between_orders": 20.0,
                "return_rate": 0.1,
                "distinct_products": 100,
            },
            {
                "customer_id": 4,
                "recency_days": 30,
                "frequency": 50,
                "gross_revenue": 100000.0,
                "tenure_days": 300,
                "avg_days_between_orders": 30.0,
                "return_rate": 0.0,
                "distinct_products": 1000,
            },
        ]
    )
    frame = seg.build_segment_frame(frame, "2011-09-01")
    prepared = seg.prepare_matrix(frame)

    assert prepared.scaled.shape == (4, len(seg.FEATURES))
    assert np.isfinite(prepared.scaled).all()
    assert np.allclose(prepared.scaled.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(prepared.scaled.std(axis=0), 1.0, atol=1e-10)
    # A four-order-of-magnitude revenue range must be less skewed on the log scale.
    assert prepared.skew_after["gross_revenue"] < prepared.skew_before["gross_revenue"]
    # Return rate is deliberately not log-transformed, so its skew is untouched.
    assert prepared.skew_after["return_rate"] == pytest.approx(prepared.skew_before["return_rate"])


def test_prepare_matrix_rejects_negative_log_features():
    frame = _snapshots([{"customer_id": 1, "gross_revenue": -10.0}])
    frame = seg.build_segment_frame(frame, "2011-09-01")
    with pytest.raises(ValueError, match="negative values"):
        seg.prepare_matrix(frame)


def test_fit_model_returns_requested_clusters_and_finite_metrics():
    scaled = _blobs([(0, 0), (4, 0), (0, 4), (4, 4)], per_centre=40)
    fit = seg.fit_model(scaled, "kmeans", 4)

    assert len(np.unique(fit.labels)) == 4
    assert fit.min_cluster_share == pytest.approx(0.25)
    assert np.isfinite([fit.silhouette, fit.davies_bouldin, fit.calinski_harabasz]).all()
    assert fit.bic is None

    gmm = seg.fit_model(scaled, "gmm", 4)
    assert gmm.bic is not None and np.isfinite(gmm.bic)


def test_fit_model_rejects_impossible_k():
    scaled = _blobs([(0, 0)], per_centre=10)
    with pytest.raises(ValueError, match="at least 2"):
        seg.fit_model(scaled, "kmeans", 1)
    with pytest.raises(ValueError, match="smaller than"):
        seg.fit_model(scaled, "kmeans", 10)


def test_select_model_enforces_minimum_k_and_cluster_share():
    comparison = pd.DataFrame(
        [
            {
                "algorithm": "gmm",
                "k": 2,
                "silhouette": 0.90,
                "davies_bouldin": 0.5,
                "calinski_harabasz": 100.0,
                "bic": 1.0,
                "min_cluster_share": 0.45,
            },
            {
                "algorithm": "kmeans",
                "k": 3,
                "silhouette": 0.50,
                "davies_bouldin": 1.0,
                "calinski_harabasz": 90.0,
                "bic": np.nan,
                "min_cluster_share": 0.01,
            },
            {
                "algorithm": "ward",
                "k": 4,
                "silhouette": 0.55,
                "davies_bouldin": 0.9,
                "calinski_harabasz": 80.0,
                "bic": np.nan,
                "min_cluster_share": 0.10,
            },
            {
                "algorithm": "kmeans",
                "k": 4,
                "silhouette": 0.60,
                "davies_bouldin": 0.8,
                "calinski_harabasz": 85.0,
                "bic": np.nan,
                "min_cluster_share": 0.05,
            },
        ]
    )
    # k=2 wins on silhouette but is excluded as non-actionable; k=3 is too small a cluster.
    assert seg.select_model(comparison) == ("kmeans", 4)


def test_bootstrap_stability_is_high_when_blobs_are_separated():
    scaled = _blobs([(0, 0), (5, 0), (0, 5), (5, 5)], per_centre=60, spread=0.2)
    stability = seg.bootstrap_stability(scaled, "kmeans", 4, n_bootstrap=8)

    assert stability.n_bootstrap == 8
    assert stability.mean > 0.9
    assert stability.unstable_runs == 0
    assert stability.minimum >= 0.0
    with pytest.raises(ValueError, match="at least 2"):
        seg.bootstrap_stability(scaled, "kmeans", 4, n_bootstrap=1)


def test_rfm_scores_rank_direction_is_hand_computed():
    frame = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "recency_days": [10, 20, 30, 40, 50],
            "frequency": [10, 8, 6, 4, 2],
            "gross_revenue": [1000.0, 800.0, 600.0, 400.0, 200.0],
        }
    )
    scores = seg.rfm_scores(frame, quantiles=5)

    # Lowest recency and highest frequency/value earn a 5.
    assert scores["r_score"].tolist() == [5, 4, 3, 2, 1]
    assert scores["f_score"].tolist() == [5, 4, 3, 2, 1]
    assert scores["m_score"].tolist() == [5, 4, 3, 2, 1]
    assert scores["rfm_score"].tolist() == [15, 12, 9, 6, 3]
    assert scores["rfm_segment"].tolist() == [
        "Champions",
        "Champions",
        "Loyal",
        "Hibernating",
        "Hibernating",
    ]


def test_rfm_scores_rejects_bad_quantiles():
    frame = pd.DataFrame(
        {"customer_id": [1], "recency_days": [1], "frequency": [1], "gross_revenue": [1.0]}
    )
    with pytest.raises(ValueError, match="at least 2"):
        seg.rfm_scores(frame, quantiles=1)


def test_profile_segments_measures_churn_only_among_active_customers():
    frame = _snapshots(
        [
            {
                "customer_id": 1,
                "frequency": 5,
                "gross_revenue": 1000.0,
                "was_active_in_prior_window": True,
                "churned": True,
                "is_active": False,
                "future_revenue": 0.0,
            },
            {
                "customer_id": 2,
                "frequency": 4,
                "gross_revenue": 800.0,
                "was_active_in_prior_window": True,
                "churned": False,
                "is_active": True,
                "future_revenue": 400.0,
            },
            {
                "customer_id": 3,
                "frequency": 1,
                "gross_revenue": 200.0,
                "was_active_in_prior_window": False,
                "churned": True,
                "is_active": False,
                "future_revenue": 0.0,
            },
            {
                "customer_id": 4,
                "frequency": 2,
                "gross_revenue": 300.0,
                "was_active_in_prior_window": False,
                "churned": True,
                "is_active": False,
                "future_revenue": 100.0,
            },
        ]
    )
    frame = seg.build_segment_frame(frame, "2011-09-01")
    profile = seg.profile_segments(frame, np.array([0, 0, 1, 1]))

    assert profile["share_of_customers"].sum() == pytest.approx(1.0)
    assert profile["gross_revenue_share"].sum() == pytest.approx(1.0)
    assert profile["future_revenue_share"].sum() == pytest.approx(1.0)
    # Cluster 0: one of two active customers churned. Cluster 1 has no active customers,
    # so its churn rate is undefined rather than 100%.
    assert profile.loc[0, "churn_rate_among_active"] == pytest.approx(0.5)
    assert np.isnan(profile.loc[1, "churn_rate_among_active"])


def test_names_are_unique_and_cover_every_cluster():
    rows = []
    customer = 1
    for recency, frequency, revenue, active, returns in (
        (20, 12, 5000.0, True, 0.0),
        (180, 4, 1000.0, False, 0.0),
        (330, 1, 250.0, False, 0.0),
        (250, 2, 400.0, False, 0.4),
    ):
        for _ in range(10):
            rows.append(
                {
                    "customer_id": customer,
                    "recency_days": recency,
                    "frequency": frequency,
                    "gross_revenue": revenue,
                    "tenure_days": 300,
                    "avg_days_between_orders": max(recency, 1),
                    "return_rate": returns,
                    "distinct_products": frequency * 5,
                    "was_active_in_prior_window": active,
                    "churned": not active,
                    "is_active": active,
                    "future_revenue": 500.0 if active else 0.0,
                }
            )
            customer += 1
    frame = seg.build_segment_frame(_snapshots(rows), "2011-09-01")
    labels = np.repeat(np.arange(4), 10)

    raw = seg.profile_segments(frame, labels)
    names = seg.name_segments(raw, frame)
    named = seg.profile_segments(frame, labels, names)

    assert sorted(names) == [0, 1, 2, 3]
    assert len(set(names.values())) == 4
    assert named.loc[names[0], "cluster"] == 0
    assert "Returns-heavy accounts" in names.values()


def test_strategy_matrix_values_every_segment_and_states_assumptions():
    profile = pd.DataFrame(
        {
            "cluster": [0, 1],
            "customers": [100, 10],
            "future_revenue": [10000.0, 1000.0],
        },
        index=pd.Index(["Champions", "Returns-heavy accounts"], name="segment"),
    )
    strategy = seg.strategy_matrix(profile)

    champions = strategy.loc["Champions"]
    assert champions["incremental_revenue_estimate"] == pytest.approx(1000.0)
    assert champions["campaign_cost_estimate"] == pytest.approx(2500.0)
    assert champions["net_estimate"] == pytest.approx(-1500.0)
    assert "assumed" in champions["uplift_basis"].lower()
    # Returns-heavy accounts are remediation, not a campaign: no uplift is claimed.
    returns = strategy.loc["Returns-heavy accounts"]
    assert returns["assumed_uplift"] == 0.0
    assert returns["incremental_revenue_estimate"] == 0.0


def test_alternative_composition_accounts_for_every_customer():
    rng = np.random.default_rng(7)
    scaled = np.vstack(
        [
            rng.normal(0, 0.2, (40, 3)),
            rng.normal(5, 0.2, (40, 3)),
            rng.normal(10, 0.2, (40, 3)),
        ]
    )
    labels = np.repeat([0, 1, 2], 40)
    names = {0: "Low", 1: "Mid", 2: "High"}

    table = _alternative_composition(scaled, labels, names, "kmeans", 2)

    assert table["total"].sum() == 120
    assert set(table.columns[:-1]) == {"Low", "Mid", "High"}
    assert table["total"].to_numpy().tolist() == (
        table.drop(columns="total").sum(axis=1).to_numpy().tolist()
    )
