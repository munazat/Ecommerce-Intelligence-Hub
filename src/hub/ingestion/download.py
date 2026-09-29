"""Reproducible raw-data acquisition. No manual downloads.

Usage:
    uv run python -m hub.ingestion.download --dataset online_retail_ii
    uv run python -m hub.ingestion.download --dataset olist
    uv run python -m hub.ingestion.download --dataset all
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

ONLINE_RETAIL_II_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
OLIST_KAGGLE_DATASET = "olistbr/brazilian-ecommerce"


@dataclass
class DownloadResult:
    dataset: str
    files: list[Path]
    source_url: str
    downloaded_at: str
    sha256: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(dest_dir: Path, result: DownloadResult) -> Path:
    manifest_path = dest_dir / f"{result.dataset}.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": result.dataset,
                "source_url": result.source_url,
                "downloaded_at": result.downloaded_at,
                "files": [str(p.relative_to(dest_dir)) for p in result.files],
                "sha256": result.sha256,
            },
            indent=2,
        )
    )
    return manifest_path


def download_online_retail_ii(dest_dir: Path = RAW_DIR, *, force: bool = False) -> DownloadResult:
    """Download the UCI Online Retail II dataset (~1.07M rows, Dec 2009-Dec 2011).

    Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
    Ships as a single .xlsx with two sheets: "Year 2009-2010" and "Year 2010-2011".
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_xlsx = dest_dir / "online_retail_II.xlsx"

    if target_xlsx.exists() and not force:
        return DownloadResult(
            dataset="online_retail_ii",
            files=[target_xlsx],
            source_url=ONLINE_RETAIL_II_URL,
            downloaded_at="already present (use --force to re-download)",
            sha256={target_xlsx.name: _sha256(target_xlsx)},
        )

    zip_path = dest_dir / "_online_retail_ii_download.zip"
    with requests.get(ONLINE_RETAIL_II_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()

    result = DownloadResult(
        dataset="online_retail_ii",
        files=[target_xlsx],
        source_url=ONLINE_RETAIL_II_URL,
        downloaded_at=datetime.now(UTC).isoformat(),
        sha256={target_xlsx.name: _sha256(target_xlsx)},
    )
    _write_manifest(dest_dir, result)
    return result


def download_olist(dest_dir: Path = RAW_DIR, *, force: bool = False) -> DownloadResult:
    """Download the Olist Brazilian e-commerce dataset (9 CSVs) via the Kaggle API.

    Requires a Kaggle account + API token (kaggle.json) — see the error message below
    if that isn't set up yet; this is a manual, one-time step that can't be scripted
    around it (Kaggle requires an authenticated account to accept the dataset's terms).
    """
    olist_dir = dest_dir / "olist"

    existing = sorted(olist_dir.glob("*.csv")) if olist_dir.exists() else []
    if existing and not force:
        return DownloadResult(
            dataset="olist",
            files=existing,
            source_url=f"kaggle:{OLIST_KAGGLE_DATASET}",
            downloaded_at="already present (use --force to re-download)",
            sha256={p.name: _sha256(p) for p in existing},
        )

    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

    api = KaggleApi()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            api.authenticate()
    except SystemExit as exc:
        raise RuntimeError(
            "Kaggle API authentication required. One-time manual setup, pick one:\n"
            "  A) Run `uv run kaggle auth login` and follow the OAuth flow in your browser.\n"
            "  B) Generate a token at https://www.kaggle.com/settings/api ('Create New Token')\n"
            "     then either:\n"
            "       - set env var KAGGLE_API_TOKEN=<token>, or\n"
            "       - save the token to C:\\Users\\<you>\\.kaggle\\access_token\n"
            "  Re-run this command afterwards."
        ) from exc

    olist_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(OLIST_KAGGLE_DATASET, path=str(olist_dir), unzip=True)

    files = sorted(olist_dir.glob("*.csv"))
    result = DownloadResult(
        dataset="olist",
        files=files,
        source_url=f"kaggle:{OLIST_KAGGLE_DATASET}",
        downloaded_at=datetime.now(UTC).isoformat(),
        sha256={p.name: _sha256(p) for p in files},
    )
    _write_manifest(dest_dir, result)
    return result


DATASETS = {
    "online_retail_ii": download_online_retail_ii,
    "olist": download_olist,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=[*DATASETS.keys(), "all"], default="all", help="Which dataset to fetch"
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    targets = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    for name in targets:
        print(f"[{name}] downloading...")
        result = DATASETS[name](force=args.force)
        print(f"[{name}] {len(result.files)} file(s) at {result.files[0].parent}")


if __name__ == "__main__":
    main()
