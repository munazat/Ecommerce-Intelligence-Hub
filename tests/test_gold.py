"""Gold layer tests: grain, reconciliation, and the returns-handling guarantee.

The fixture below is small enough to hand-compute every expected number from, which is
the point -- a reconciliation test that only checks "the two sides agree" would pass even
if both sides were wrong in the same way.

Fixture inventory (net revenue in GBP):
    500001  cust 1001  2010-01-05 09:00  85048  qty 2 @ 10.00   =  20.00
    500001  cust 1001  2010-01-05 09:04  85049  qty 1 @ 30.00   =  30.00
    500002  cust 1001  2010-01-20 10:00  85048  qty 1 @ 10.00   =  10.00
    C500003 cust 1001  2010-02-15 11:00  85048  qty -1 @ 10.00  = -10.00  cancellation
    500004  cust ----  2010-01-25 12:00  POST   qty 1 @ 18.00   =  18.00  non-product
    A500005 cust ----  2010-03-01 13:00  B      qty 1 @ -500.00 = -500.00 adjustment
    500006  cust 1002  2010-01-10 09:30  85049  qty 3 @ 5.00    =  15.00
    ------------------------------------------------------------------------
    total net revenue                                          = -417.00
"""

from __future__ import annotations

import pandas as pd
import pytest

from hub.ingestion.gold import build_gold

FIXTURE_ROWS = [
    {"invoice_no": "500001", "stock_code": "85048", "quantity": 2, "unit_price": 10.0},
    {
        "invoice_no": "500001",
        "stock_code": "85049",
        "quantity": 1,
        "unit_price": 30.0,
        "invoice_date": pd.Timestamp("2010-01-05 09:04:00"),
    },
    {
        "invoice_no": "500002",
        "quantity": 1,
        "invoice_date": pd.Timestamp("2010-01-20 10:00:00"),
    },
    {
        "invoice_no": "C500003",
        "quantity": -1,
        "is_cancellation": True,
        "invoice_date": pd.Timestamp("2010-02-15 11:00:00"),
    },
    {
        "invoice_no": "500004",
        "stock_code": "POST",
        "description": "POSTAGE",
        "customer_id": None,
        "unit_price": 18.0,
        "is_product": False,
        "invoice_date": pd.Timestamp("2010-01-25 12:00:00"),
    },
    {
        "invoice_no": "A500005",
        "stock_code": "B",
        "description": "Adjust bad debt",
        "customer_id": None,
        "unit_price": -500.0,
        "is_product": False,
        "is_adjustment": True,
        "invoice_date": pd.Timestamp("2010-03-01 13:00:00"),
    },
    {
        "invoice_no": "500006",
        "stock_code": "85049",
        "customer_id": 1002,
        "quantity": 3,
        "unit_price": 5.0,
        "invoice_date": pd.Timestamp("2010-01-10 09:30:00"),
    },
]


@pytest.fixture
def gold(tmp_path, write_silver):
    write_silver(tmp_path / "silver", FIXTURE_ROWS)
    gold_dir = tmp_path / "gold"
    build_gold(silver_dir=tmp_path / "silver", gold_dir=gold_dir)
    return {
        name: pd.read_parquet(gold_dir / f"{name}.parquet")
        for name in [
            "fact_order_lines",
            "fact_orders",
            "dim_customer",
            "dim_product",
            "dim_date",
        ]
    }


def test_row_counts_and_grains(gold):
    assert len(gold["fact_order_lines"]) == 7
    assert len(gold["fact_orders"]) == 6
    assert gold["fact_orders"]["invoice_no"].is_unique
    assert gold["fact_order_lines"]["line_key"].is_unique
    assert gold["dim_customer"]["customer_id"].is_unique
    assert gold["dim_product"]["stock_code"].is_unique
    assert gold["dim_date"]["date_key"].is_unique
    assert sorted(gold["dim_product"]["stock_code"]) == ["85048", "85049", "B", "POST"]


def test_revenue_reconciles_across_all_layers(gold):
    """The roadmap's required check: customer-level revenue ties to order-level revenue.

    It only ties once the unattributable (null Customer ID) revenue is added back -- that
    term is the documented silver policy for missing customer ids, not a fudge factor.
    """
    lines = gold["fact_order_lines"]["revenue"].sum()
    orders = gold["fact_orders"]["net_revenue"].sum()
    customers = gold["dim_customer"]["net_revenue"].sum()
    unattributable = (
        gold["fact_order_lines"]
        .loc[gold["fact_order_lines"]["customer_id"].isna(), "revenue"]
        .sum()
    )

    assert lines == pytest.approx(-417.0)
    assert orders == pytest.approx(lines)
    assert customers + unattributable == pytest.approx(lines)
    assert unattributable == pytest.approx(18.0 - 500.0)

    # Units must tie the same way, or the fact table is dropping rows somewhere.
    assert gold["fact_orders"]["line_count"].sum() == len(gold["fact_order_lines"])
    assert gold["fact_orders"]["units"].sum() == gold["fact_order_lines"]["quantity"].sum()


def test_no_negative_revenue_after_returns_handling(gold):
    """Reversals are supposed to be negative; everything else must not be."""
    orders = gold["fact_orders"]
    ordinary = orders[~(orders["is_cancellation"] | orders["is_adjustment"])]
    reversals = orders[orders["is_cancellation"] | orders["is_adjustment"]]

    assert (ordinary["net_revenue"] >= 0).all()
    assert len(ordinary) == 4
    # The guarantee is meaningful only because the negatives are quarantined, not deleted.
    assert (reversals["net_revenue"] < 0).all()
    assert len(reversals) == 2
    # And gross revenue excludes reversals entirely, so it never goes negative either.
    assert reversals["gross_revenue"].abs().sum() == 0


def test_order_date_is_the_basket_start(gold):
    """Invoice 500001 has two line timestamps 4 minutes apart; the order takes the first."""
    order = gold["fact_orders"].set_index("invoice_no").loc["500001"]
    assert order["invoice_date"] == pd.Timestamp("2010-01-05 09:00:00")
    assert order["line_count"] == 2
    assert order["net_revenue"] == pytest.approx(50.0)
    # Per-line timestamps survive in the atomic fact table.
    line_times = gold["fact_order_lines"].loc[
        gold["fact_order_lines"]["invoice_no"] == "500001", "invoice_date"
    ]
    assert sorted(line_times) == [
        pd.Timestamp("2010-01-05 09:00:00"),
        pd.Timestamp("2010-01-05 09:04:00"),
    ]


def test_customer_aggregates_match_hand_computed_values(gold):
    c = gold["dim_customer"].set_index("customer_id")
    assert c.loc[1001, "invoice_count"] == 3
    assert c.loc[1001, "purchase_count"] == 2  # the cancellation is not a purchase
    assert c.loc[1001, "net_revenue"] == pytest.approx(50.0)
    assert c.loc[1001, "line_count"] == 4
    assert c.loc[1001, "distinct_products"] == 2
    assert c.loc[1001, "units"] == 3  # 2 + 1 + 1 - 1
    assert c.loc[1001, "cancellation_line_count"] == 1
    assert c.loc[1002, "invoice_count"] == 1
    assert c.loc[1002, "net_revenue"] == pytest.approx(15.0)


def test_missing_customer_ids_are_excluded_from_dim_but_kept_in_facts(gold):
    """22.8% of real rows have no Customer ID. They must not vanish from revenue."""
    assert set(gold["dim_customer"]["customer_id"]) == {1001, 1002}
    unattributed = gold["fact_orders"][gold["fact_orders"]["customer_id"].isna()]
    assert sorted(unattributed["invoice_no"]) == ["500004", "A500005"]
    assert unattributed["net_revenue"].sum() == pytest.approx(-482.0)


def test_product_and_non_product_revenue_split(gold):
    orders = gold["fact_orders"]
    postage = orders.set_index("invoice_no").loc["500004"]
    assert postage["product_revenue"] == pytest.approx(0.0)
    assert postage["non_product_revenue"] == pytest.approx(18.0)
    assert postage["product_count"] == 0


def test_dim_date_is_contiguous_with_no_missing_days(gold):
    dates = gold["dim_date"]
    assert len(dates) == 56  # 2010-01-05 .. 2010-03-01 inclusive
    assert dates["date_day"].min() == pd.Timestamp("2010-01-05")
    assert dates["date_day"].max() == pd.Timestamp("2010-03-01")
    # No gaps: a continuous axis is what lets a zero-revenue day read as zero demand.
    assert len(dates) == (dates["date_day"].max() - dates["date_day"].min()).days + 1
    # ISO calendar correctness: 2010-01-05 is ISO year 2010, week 1, Tuesday.
    first = dates.iloc[0]
    assert (first["iso_year"], first["iso_week"], first["iso_day_of_week"]) == (2010, 1, 2)
    assert not first["is_weekend"]
    # 2010-01-09 is the first Saturday in range and must be flagged.
    saturday = dates[dates["date_day"] == pd.Timestamp("2010-01-09")].iloc[0]
    assert saturday["is_weekend"]
    assert saturday["day_name"] == "Saturday"


def test_build_gold_fails_loudly_without_silver(tmp_path):
    with pytest.raises(FileNotFoundError, match="hub.ingestion.silver"):
        build_gold(silver_dir=tmp_path / "empty", gold_dir=tmp_path / "gold")
