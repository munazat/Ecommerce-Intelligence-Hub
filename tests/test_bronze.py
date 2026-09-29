import pandas as pd
import pytest

from hub.ingestion.bronze import load_olist_to_bronze, load_online_retail_ii_to_bronze


@pytest.fixture
def raw_online_retail_ii(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    src = raw_dir / "online_retail_II.xlsx"

    sheet_2009 = pd.DataFrame(
        {
            "Invoice": ["489434", "489435"],
            "StockCode": ["85048", "POST"],
            "Description": ["15CM CHRISTMAS GLASS BALL", "POSTAGE"],
            "Quantity": [12, 1],
            "InvoiceDate": pd.to_datetime(["2009-12-01 07:45:00", "2009-12-01 07:46:00"]),
            "Price": [6.95, 18.0],
            "Customer ID": [13085, None],
            "Country": ["United Kingdom", "United Kingdom"],
        }
    )
    sheet_2010 = pd.DataFrame(
        {
            "Invoice": ["C536379", "536378"],
            "StockCode": ["D", "21730"],
            "Description": ["Discount", "GLASS STAR FROSTED"],
            "Quantity": [-1, 4],
            "InvoiceDate": pd.to_datetime(["2010-12-01 09:41:00", "2010-12-01 09:41:00"]),
            "Price": [27.5, 4.25],
            "Customer ID": [14527, 14527],
            "Country": ["United Kingdom", "United Kingdom"],
        }
    )
    with pd.ExcelWriter(src) as writer:
        sheet_2009.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        sheet_2010.to_excel(writer, sheet_name="Year 2010-2011", index=False)

    return raw_dir


def test_load_online_retail_ii_to_bronze(raw_online_retail_ii, tmp_path):
    bronze_dir = tmp_path / "bronze"

    result = load_online_retail_ii_to_bronze(raw_dir=raw_online_retail_ii, bronze_dir=bronze_dir)

    assert result.rows == 4
    assert result.path == bronze_dir / "online_retail_ii.parquet"
    assert result.path.exists()

    df = pd.read_parquet(result.path)
    assert len(df) == 4
    # Bronze must not drop or rename any raw business column.
    assert {
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    } <= set(df.columns)
    # Ingestion metadata columns are present.
    assert {"_source_sheet", "_source_file", "_ingested_at"} <= set(df.columns)
    assert set(df["_source_sheet"]) == {"Year 2009-2010", "Year 2010-2011"}
    # Alphanumeric identifiers preserved exactly, not coerced to numbers.
    assert "C536379" in df["Invoice"].to_numpy()
    assert "POST" in df["StockCode"].to_numpy()
    # Missing Customer ID preserved as null, not dropped or filled.
    assert df["Customer ID"].isna().sum() == 1


def test_load_online_retail_ii_to_bronze_missing_raw_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_online_retail_ii_to_bronze(raw_dir=tmp_path / "empty", bronze_dir=tmp_path / "bronze")


@pytest.fixture
def raw_olist(tmp_path):
    raw_dir = tmp_path / "raw" / "olist"
    raw_dir.mkdir(parents=True)
    (raw_dir / "olist_customers_dataset.csv").write_text(
        "customer_id,customer_unique_id,customer_city\nc1,u1,sao paulo\nc2,u2,rio\n"
    )
    (raw_dir / "olist_orders_dataset.csv").write_text(
        "order_id,customer_id,order_status\no1,c1,delivered\n"
    )
    # olist_products_dataset.csv deliberately absent to test partial-dataset tolerance.
    return raw_dir.parent


def test_load_olist_to_bronze_is_tolerant_of_missing_tables(raw_olist, tmp_path):
    bronze_dir = tmp_path / "bronze"

    results = load_olist_to_bronze(raw_dir=raw_olist, bronze_dir=bronze_dir)

    tables = {r.table: r for r in results}
    assert set(tables) == {"olist_customers_dataset", "olist_orders_dataset"}
    assert tables["olist_customers_dataset"].rows == 2
    assert tables["olist_orders_dataset"].rows == 1

    customers = pd.read_parquet(bronze_dir / "olist" / "olist_customers_dataset.parquet")
    assert {"_source_file", "_ingested_at"} <= set(customers.columns)


def test_load_olist_to_bronze_missing_raw_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_olist_to_bronze(raw_dir=tmp_path / "empty", bronze_dir=tmp_path / "bronze")
