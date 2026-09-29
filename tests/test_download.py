import hashlib

from hub.ingestion.download import _sha256, download_olist, download_online_retail_ii


def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_bytes(b"some bytes to hash")
    assert _sha256(f) == hashlib.sha256(b"some bytes to hash").hexdigest()


def test_download_online_retail_ii_skips_when_already_present(tmp_path):
    """Must not touch the network if the file already exists (no --force)."""
    existing = tmp_path / "online_retail_II.xlsx"
    existing.write_bytes(b"fake xlsx bytes")

    result = download_online_retail_ii(dest_dir=tmp_path)

    assert result.dataset == "online_retail_ii"
    assert result.files == [existing]
    assert "already present" in result.downloaded_at


def test_download_olist_skips_when_already_present(tmp_path):
    """Must not touch the Kaggle API if CSVs already exist (no --force)."""
    olist_dir = tmp_path / "olist"
    olist_dir.mkdir()
    csv_a = olist_dir / "orders.csv"
    csv_b = olist_dir / "customers.csv"
    csv_a.write_text("id\n1\n")
    csv_b.write_text("id\n1\n")

    result = download_olist(dest_dir=tmp_path)

    assert result.dataset == "olist"
    assert sorted(p.name for p in result.files) == ["customers.csv", "orders.csv"]
    assert "already present" in result.downloaded_at
