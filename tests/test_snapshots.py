"""Point-in-time snapshot tests, with the leakage guarantee as the centrepiece.

The fixture spans 2010-01-05 .. 2010-08-01, which yields six monthly snapshots
(2010-03-01 .. 2010-08-01) for two customers -- twelve rows, every value hand-computable.

    cust 1001   500001   2010-01-05 09:00   85048  qty  2 @ 10.00  =  20.00
    cust 1001   500002   2010-01-20 09:00   85049  qty  1 @ 30.00  =  30.00
    cust 1001   500003   2010-03-10 09:00   85048  qty  1 @ 10.00  =  10.00
    cust 1001   C500004  2010-08-01 09:00   85048  qty -1 @ 10.00  = -10.00  cancellation
    cust 1002   500005   2010-01-10 09:00   85049  qty  3 @  5.00  =  15.00
"""

from __future__ import annotations

import pandas as pd
import pytest

from hub.features.snapshots import build_customer_snapshots
from hub.ingestion.gold import build_gold

FEATURE_COLUMNS = [
    "first_invoice_date",
    "last_invoice_date",
    "recency_days",
    "tenure_days",
    "invoice_count",
    "frequency",
    "monetary",
    "line_count",
    "distinct_products",
    "active_days",
    "units",
    "net_revenue",
    "gross_revenue",
    "cancellation_line_count",
    "return_rate",
    "avg_days_between_orders",
    "was_active_in_prior_window",
]

LABEL_COLUMNS = [
    "future_invoices",
    "future_orders",
    "future_lines",
    "future_units",
    "future_revenue",
    "is_active",
    "churned",
]

EXPECTED_SNAPSHOTS = [pd.Timestamp(f"2010-{m:02d}-01") for m in range(3, 9)]

SNAPSHOT_ROWS = [
    {"invoice_no": "500001", "stock_code": "85048", "quantity": 2, "unit_price": 10.0},
    {
        "invoice_no": "500002",
        "stock_code": "85049",
        "quantity": 1,
        "unit_price": 30.0,
        "invoice_date": pd.Timestamp("2010-01-20 09:00:00"),
    },
    {
        "invoice_no": "500003",
        "quantity": 1,
        "invoice_date": pd.Timestamp("2010-03-10 09:00:00"),
    },
    {
        "invoice_no": "C500004",
        "quantity": -1,
        "is_cancellation": True,
        "invoice_date": pd.Timestamp("2010-08-01 09:00:00"),
    },
    {
        "invoice_no": "500005",
        "stock_code": "85049",
        "customer_id": 1002,
        "quantity": 3,
        "unit_price": 5.0,
        "invoice_date": pd.Timestamp("2010-01-10 09:00:00"),
    },
]


def _build(tmp_path, write_silver, rows, subdir):
    """Run silver -> gold -> snapshots on a fixture and return the snapshot table."""
    silver_dir = tmp_path / subdir / "silver"
    gold_dir = tmp_path / subdir / "gold"
    write_silver(silver_dir, rows)
    build_gold(silver_dir=silver_dir, gold_dir=gold_dir)
    result = build_customer_snapshots(gold_dir=gold_dir)
    return pd.read_parquet(result.path)


@pytest.fixture
def snapshots(tmp_path, write_silver):
    return _build(tmp_path, write_silver, SNAPSHOT_ROWS, "base")


def _row(snapshots, customer_id, snapshot_date):
    key = pd.Timestamp(snapshot_date)
    match = snapshots[
        (snapshots["customer_id"] == customer_id) & (snapshots["snapshot_date"] == key)
    ]
    assert len(match) == 1, f"expected exactly one row for {customer_id} @ {key}"
    return match.iloc[0]


def _snapshot_dates(snapshots):
    return pd.to_datetime(sorted(snapshots["snapshot_date"].unique())).tolist()


def test_grain_and_snapshot_grid(snapshots):
    assert len(snapshots) == 12
    assert not snapshots.duplicated(subset=["customer_id", "snapshot_date"]).any()
    assert _snapshot_dates(snapshots) == EXPECTED_SNAPSHOTS
    assert set(snapshots["customer_id"]) == {1001, 1002}
    assert (snapshots["label_horizon_days"] == 90).all()
    # Guards against a column being dropped from the SQL without any test noticing.
    missing = [c for c in [*FEATURE_COLUMNS, *LABEL_COLUMNS] if c not in snapshots.columns]
    assert not missing, f"snapshots are missing columns: {missing}"


def test_features_match_hand_computed_values(snapshots):
    """Customer 1001 at the first snapshot: two purchases, both in January."""
    s = _row(snapshots, 1001, "2010-03-01")
    assert s["recency_days"] == 40  # 2010-01-20 -> 2010-03-01
    assert s["tenure_days"] == 55  # 2010-01-05 -> 2010-03-01
    assert s["invoice_count"] == 2
    assert s["frequency"] == 2
    assert s["monetary"] == pytest.approx(25.0)  # 50.00 gross / 2 purchases
    assert s["gross_revenue"] == pytest.approx(50.0)
    assert s["units"] == 3
    assert s["distinct_products"] == 2
    assert s["active_days"] == 2
    assert s["avg_days_between_orders"] == pytest.approx(15.0)
    assert s["return_rate"] == pytest.approx(0.0)
    assert s["was_active_in_prior_window"]

    # The 2010-03-10 purchase is in this snapshot's future and must not appear above.
    assert s["last_invoice_date"] == pd.Timestamp("2010-01-20 09:00:00")


def test_single_purchase_customer_has_no_inter_purchase_interval(snapshots):
    s = _row(snapshots, 1002, "2010-03-01")
    assert s["frequency"] == 1
    assert pd.isna(s["avg_days_between_orders"])  # not 0 -- there is no interval
    assert s["recency_days"] == 50
    assert s["tenure_days"] == 50  # first and last purchase are the same day
    assert s["monetary"] == pytest.approx(15.0)


def test_labels_see_only_the_90_day_future_window(snapshots):
    # 2010-03-10 falls inside [2010-03-01, 2010-05-30).
    march = _row(snapshots, 1001, "2010-03-01")
    assert march["future_orders"] == 1
    assert march["future_revenue"] == pytest.approx(10.0)
    assert march["is_active"]
    assert not march["churned"]

    # ...and outside [2010-04-01, 2010-06-30), so the same customer is churned a month on.
    april = _row(snapshots, 1001, "2010-04-01")
    assert april["future_orders"] == 0
    assert april["future_revenue"] == pytest.approx(0.0)
    assert april["churned"]

    # A customer who never buys again is churned at every snapshot.
    assert all(_row(snapshots, 1002, d)["churned"] for d in EXPECTED_SNAPSHOTS)


def test_cancellation_inside_the_window_is_not_activity(snapshots):
    """A return is not a purchase. Counting it would hide exactly the churn we predict."""
    s = _row(snapshots, 1001, "2010-08-01")
    assert s["future_invoices"] == 1  # the C500004 invoice does exist in the window
    assert s["future_orders"] == 0  # ...but it is a cancellation, not an order
    assert s["future_revenue"] == pytest.approx(-10.0)
    assert not s["is_active"]
    assert s["churned"]
    # 90 days before 2010-08-01 is 2010-05-03; the last real purchase was 2010-03-10.
    assert not s["was_active_in_prior_window"]


def test_right_censored_label_windows_are_flagged_not_dropped(snapshots):
    complete = snapshots[snapshots["is_label_window_complete"]]
    censored = snapshots[~snapshots["is_label_window_complete"]]

    assert _snapshot_dates(complete) == EXPECTED_SNAPSHOTS[:3]
    assert _snapshot_dates(censored) == EXPECTED_SNAPSHOTS[3:]
    assert len(complete) == len(censored) == 6


def test_features_ignore_data_added_after_the_snapshot(tmp_path, write_silver):
    """The leakage test.

    A purchase is inserted on 2010-06-15 for customer 1002, who had bought exactly once
    before. For every snapshot on or before that date the features must be identical,
    because features are only allowed to read rows strictly before snapshot_date. If any
    feature moves, the snapshot saw the future and every model trained on it is worthless.

    The test also asserts the features *do* change for later snapshots and the labels *do*
    change for windows covering 2010-06-15 -- otherwise it would pass vacuously.
    """
    base = _build(tmp_path, write_silver, SNAPSHOT_ROWS, "leak_base")
    inserted = {
        "invoice_no": "500099",
        "stock_code": "85048",
        "customer_id": 1002,
        "quantity": 100,
        "unit_price": 999.0,
        "invoice_date": pd.Timestamp("2010-06-15 09:00:00"),
    }
    mutated = _build(tmp_path, write_silver, [*SNAPSHOT_ROWS, inserted], "leak_mutated")

    cutoff = pd.Timestamp("2010-06-15")

    def features_at_or_before(snap):
        s = snap[snap["snapshot_date"] <= cutoff].sort_values(["customer_id", "snapshot_date"])
        return s[["customer_id", "snapshot_date", *FEATURE_COLUMNS]].reset_index(drop=True)

    pd.testing.assert_frame_equal(features_at_or_before(base), features_at_or_before(mutated))

    # Non-vacuity, part 1: once the inserted purchase IS history, features must move.
    late_base = _row(base, 1002, "2010-07-01")
    late_mutated = _row(mutated, 1002, "2010-07-01")
    assert late_base["frequency"] == 1
    assert late_mutated["frequency"] == 2
    assert late_mutated["net_revenue"] == pytest.approx(15.0 + 100 * 999.0)

    # Non-vacuity, part 2: labels are *supposed* to see 2010-06-15, and they do.
    assert _row(base, 1002, "2010-05-01")["churned"]
    assert not _row(mutated, 1002, "2010-05-01")["churned"]
    assert _row(mutated, 1002, "2010-05-01")["future_revenue"] == pytest.approx(99900.0)


def test_snapshot_boundary_is_strict(tmp_path, write_silver):
    """A row at exactly snapshot midnight belongs to the label window, never to history."""
    rows = [
        {"invoice_no": "500001", "quantity": 1, "unit_price": 10.0},
        {
            "invoice_no": "500009",
            "quantity": 1,
            "unit_price": 100.0,
            "invoice_date": pd.Timestamp("2010-03-01 00:00:00"),
        },
    ]
    snap = _build(tmp_path, write_silver, rows, "boundary")
    assert _snapshot_dates(snap) == [pd.Timestamp("2010-03-01")]

    s = _row(snap, 1001, "2010-03-01")
    # History side: only the January row.
    assert s["frequency"] == 1
    assert s["line_count"] == 1
    assert s["gross_revenue"] == pytest.approx(10.0)
    assert s["recency_days"] == 55
    # Label side: the boundary row, at exactly snapshot midnight.
    assert s["future_orders"] == 1
    assert s["future_revenue"] == pytest.approx(100.0)
    assert s["is_active"]


def test_snapshots_fail_loudly_without_gold(tmp_path):
    with pytest.raises(FileNotFoundError, match="hub.ingestion.gold"):
        build_customer_snapshots(gold_dir=tmp_path / "empty")
