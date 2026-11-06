"""Market-basket tests on a hand-built five-basket warehouse.

Baskets: AB, AB, AC, DE, BC. Non-product and cancellation lines are present but must
never enter a basket, and both rule directions must be emitted.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hub.analytics.baskets import analyse_baskets
from hub.ingestion.gold import build_gold

FIXTURE_ROWS = [
    # AB
    {"invoice_no": "100", "stock_code": "A", "quantity": 1},
    {"invoice_no": "100", "stock_code": "B", "quantity": 1},
    # AB, plus postage that is not merchandise
    {"invoice_no": "101", "stock_code": "A", "quantity": 1},
    {"invoice_no": "101", "stock_code": "B", "quantity": 1},
    {
        "invoice_no": "101",
        "stock_code": "POST",
        "quantity": 1,
        "unit_price": 5.0,
        "description": "POSTAGE",
        "is_product": False,
    },
    # AC
    {"invoice_no": "102", "stock_code": "A", "quantity": 1},
    {"invoice_no": "102", "stock_code": "C", "quantity": 1},
    # DE
    {"invoice_no": "103", "stock_code": "D", "quantity": 1},
    {"invoice_no": "103", "stock_code": "E", "quantity": 1},
    # BC
    {"invoice_no": "104", "stock_code": "B", "quantity": 1},
    {"invoice_no": "104", "stock_code": "C", "quantity": 1},
    # Cancelled merchandise must not count as a co-purchase.
    {"invoice_no": "C105", "stock_code": "F", "quantity": -1, "is_cancellation": True},
    {"invoice_no": "C105", "stock_code": "A", "quantity": -1, "is_cancellation": True},
]


@pytest.fixture
def lines_path(tmp_path, write_silver):
    write_silver(tmp_path / "silver", FIXTURE_ROWS)
    build_gold(silver_dir=tmp_path / "silver", gold_dir=tmp_path / "gold")
    return tmp_path / "gold" / "fact_order_lines.parquet"


def test_summary_counts_only_ordinary_product_baskets(lines_path):
    result = analyse_baskets(lines_path)
    summary = result.summary

    assert summary.baskets == 5
    assert summary.items == 5
    assert summary.eligible_items == 5
    assert summary.avg_items == pytest.approx(2.0)
    assert summary.median_items == pytest.approx(2.0)
    assert summary.max_items == 2


def test_rules_include_both_directions_with_asymmetric_confidence(lines_path):
    rules = analyse_baskets(lines_path).rules

    assert len(rules) == 8  # four unordered pairs, both directions
    assert set(rules["antecedent"]) <= {"A", "B", "C", "D", "E"}
    assert "POST" not in set(rules["antecedent"]) | set(rules["consequent"])
    assert "F" not in set(rules["antecedent"]) | set(rules["consequent"])

    def rule(left: str, right: str) -> pd.Series:
        return rules.loc[(rules["antecedent"] == left) & (rules["consequent"] == right)].iloc[0]

    ab = rule("A", "B")
    assert ab["pair_baskets"] == 2
    assert ab["support_pair"] == pytest.approx(0.4)
    assert ab["support_antecedent"] == pytest.approx(0.6)
    assert ab["support_consequent"] == pytest.approx(0.6)
    assert ab["confidence"] == pytest.approx(2 / 3)
    assert ab["lift"] == pytest.approx(10 / 9)

    # A -> C is weaker than C -> A: three baskets hold A, only two hold C.
    assert rule("A", "C")["confidence"] == pytest.approx(1 / 3)
    assert rule("C", "A")["confidence"] == pytest.approx(1 / 2)

    # D and E always occur together, so the lift is 1 / 0.2.
    de = rule("D", "E")
    assert de["lift"] == pytest.approx(5.0)
    assert de["confidence"] == pytest.approx(1.0)
    assert rules.iloc[0]["lift"] == pytest.approx(5.0)


def test_confidence_threshold_filters_rules(lines_path):
    rules = analyse_baskets(lines_path, min_confidence=0.9).rules

    assert len(rules) == 2
    assert set(zip(rules["antecedent"], rules["consequent"], strict=True)) == {
        ("D", "E"),
        ("E", "D"),
    }


def test_item_support_threshold_removes_rare_items(lines_path):
    rules = analyse_baskets(lines_path, min_item_support=0.5).rules

    # Only A and B clear 50% of baskets, so only the A/B pair survives.
    assert len(rules) == 2
    assert set(rules["antecedent"]) == {"A", "B"}


def test_top_n_limits_rows(lines_path):
    rules = analyse_baskets(lines_path, top_n=3).rules
    assert len(rules) == 3


def test_invalid_thresholds_raise(lines_path):
    for kwargs in (
        {"min_item_support": 0.0},
        {"min_pair_support": 1.0},
        {"min_confidence": -0.1},
    ):
        with pytest.raises(ValueError, match="must be in"):
            analyse_baskets(lines_path, **kwargs)
