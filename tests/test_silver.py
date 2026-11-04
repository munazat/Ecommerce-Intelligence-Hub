from datetime import UTC, datetime

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from hub.ingestion.silver import clean_online_retail_ii
from hub.validation.schemas import silver_online_retail_ii_schema


def _write_bronze_fixture(bronze_dir):
    bronze_dir.mkdir(parents=True)
    rows = pd.DataFrame(
        {
            "Invoice": ["489434", "489434", "489435", "C489436", "489437", "489438", "A506401"],
            "StockCode": ["85048", "85048", "POST", "22041", "gift_0001_20", "21730 ", "B"],
            "Description": [
                "15CM CHRISTMAS GLASS BALL",
                "15CM CHRISTMAS GLASS BALL",
                "POSTAGE",
                'RECORD FRAME 7" SINGLE SIZE',
                "Dotcomgiftshop Gift Voucher £20.00",
                "GLASS STAR FROSTED",
                "Adjust bad debt",
            ],
            "Quantity": [12, 12, 1, 500, 1, 4, 1],
            "InvoiceDate": pd.to_datetime(
                [
                    "2009-12-01 07:45:00",
                    "2009-12-01 07:45:00",
                    "2009-12-01 07:46:00",
                    "2010-01-05 10:00:00",
                    "2010-02-01 09:00:00",
                    "2010-02-02 09:00:00",
                    "2010-04-29 13:36:00",
                ]
            ),
            "Price": [6.95, 6.95, 18.0, 0.0, 17.39, 4.25, -53594.36],
            "Customer ID": [13085, 13085, None, 14527, 14527, 14528, None],
            "Country": ["United Kingdom"] * 7,
            "_source_sheet": ["Year 2009-2010"] * 4 + ["Year 2010-2011"] * 2 + ["Year 2009-2010"],
            "_source_file": ["online_retail_II.xlsx"] * 7,
            "_ingested_at": [datetime.now(UTC).isoformat()] * 7,
        }
    )
    rows.to_parquet(bronze_dir / "online_retail_ii.parquet", index=False)


def test_clean_online_retail_ii(tmp_path):
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    _write_bronze_fixture(bronze_dir)

    result = clean_online_retail_ii(bronze_dir=bronze_dir, silver_dir=silver_dir)

    assert result.rows_in == 7
    assert result.duplicates_dropped == 1  # the two identical 85048 rows
    assert result.rows_out == 6

    df = pd.read_parquet(result.path)

    # Cancellation flag matches the 'C' prefix exactly.
    assert df.set_index("invoice_no")["is_cancellation"].to_dict()["C489436"] is True
    assert df.set_index("invoice_no")["is_cancellation"].to_dict()["489434"] is False

    # Adjustment flag matches the 'A' prefix exactly, and the two never overlap.
    is_adjustment_by_invoice = df.set_index("invoice_no")["is_adjustment"].to_dict()
    assert is_adjustment_by_invoice["A506401"] is True
    assert is_adjustment_by_invoice["489434"] is False
    assert is_adjustment_by_invoice["C489436"] is False
    assert not (df["is_cancellation"] & df["is_adjustment"]).any()

    # POST and the gift voucher are flagged non-product; real products stay flagged True.
    is_product_by_code = df.set_index("stock_code")["is_product"].to_dict()
    assert is_product_by_code["POST"] is False
    assert is_product_by_code["gift_0001_20"] is False
    assert is_product_by_code["B"] is False  # bad-debt adjustment, not merchandise
    assert is_product_by_code["85048"] is True
    # StockCode was whitespace-trimmed, not left as "21730 ".
    assert "21730" in is_product_by_code
    assert is_product_by_code["21730"] is True

    # Zero-price row flagged, missing customer_id preserved as null (not dropped).
    zero_price_row = df[df["stock_code"] == "22041"].iloc[0]
    assert zero_price_row["is_zero_price"]
    assert bool(zero_price_row["is_quantity_outlier"])  # Quantity 500 > threshold 200

    post_row = df[df["stock_code"] == "POST"].iloc[0]
    assert pd.isna(post_row["customer_id"])

    # revenue = quantity * unit_price, computed correctly including for cancellations.
    cancel_row = df[df["invoice_no"] == "C489436"].iloc[0]
    assert cancel_row["revenue"] == pytest.approx(500 * 0.0)

    normal_row = df[df["stock_code"] == "gift_0001_20"].iloc[0]
    assert normal_row["revenue"] == pytest.approx(1 * 17.39)


def test_clean_online_retail_ii_missing_bronze_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        clean_online_retail_ii(bronze_dir=tmp_path / "empty", silver_dir=tmp_path / "silver")


def _valid_silver_row() -> dict:
    return {
        "invoice_no": "489434",
        "stock_code": "85048",
        "description": "15CM CHRISTMAS GLASS BALL",
        "quantity": 12,
        "invoice_date": pd.Timestamp("2010-01-01"),
        "unit_price": 6.95,
        "customer_id": 13085,
        "country": "United Kingdom",
        "is_cancellation": False,
        "is_adjustment": False,
        "is_product": True,
        "is_zero_price": False,
        "is_quantity_outlier": False,
        "is_price_outlier": False,
        "revenue": 12 * 6.95,
        "source_sheet": "Year 2009-2010",
        "cleaned_at": datetime.now(UTC).isoformat(),
    }


def test_schema_accepts_valid_row():
    df = pd.DataFrame([_valid_silver_row()])
    validated = silver_online_retail_ii_schema.validate(df, lazy=True)
    assert len(validated) == 1


def test_schema_fails_loudly_when_cancellation_flag_disagrees_with_invoice_no():
    """The pipeline must fail loudly, not silently, on a broken invariant."""
    row = _valid_silver_row()
    row["invoice_no"] = "C489434"  # a cancellation invoice...
    row["is_cancellation"] = False  # ...mislabeled as not a cancellation.
    df = pd.DataFrame([row])

    with pytest.raises(SchemaErrors):
        silver_online_retail_ii_schema.validate(df, lazy=True)


def test_schema_fails_loudly_when_adjustment_flag_disagrees_with_invoice_no():
    row = _valid_silver_row()
    row["invoice_no"] = "A506401"  # a bad-debt adjustment invoice...
    row["is_adjustment"] = False  # ...mislabeled as an ordinary invoice.
    df = pd.DataFrame([row])

    with pytest.raises(SchemaErrors):
        silver_online_retail_ii_schema.validate(df, lazy=True)


def test_schema_fails_loudly_on_revenue_mismatch():
    row = _valid_silver_row()
    row["revenue"] = 999999.0  # doesn't match quantity * unit_price
    df = pd.DataFrame([row])

    with pytest.raises(SchemaErrors):
        silver_online_retail_ii_schema.validate(df, lazy=True)


def test_schema_fails_loudly_on_zero_quantity():
    row = _valid_silver_row()
    row["quantity"] = 0
    row["revenue"] = 0.0
    df = pd.DataFrame([row])

    with pytest.raises(SchemaErrors):
        silver_online_retail_ii_schema.validate(df, lazy=True)
