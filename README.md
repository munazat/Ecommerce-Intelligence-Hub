# E-Commerce Intelligence Hub

End-to-end customer analytics platform: ingests raw e-commerce transaction data, validates and models it through a layered warehouse, predicts customer lifetime value and churn with probabilistic and ML models, forecasts revenue with backtested time-series models, and turns predictions into dollar-optimized retention decisions — served via an API, a dashboard, and auto-generated reports.

**Status:** the data engineering and analysis layers are complete for the primary dataset — reproducible download, bronze, silver with enforced data contracts, a SQL/DuckDB gold warehouse, a point-in-time feature store built specifically so no downstream model can leak, and a tested metrics/EDA layer covering revenue, retention, churn, cohorts, seasonality, product concentration and market baskets. Segmentation, CLV/churn modelling, forecasting, the decision layer, the API and the dashboard are not started.

Phase 3 results are summarised in [`docs/key_findings.md`](docs/key_findings.md) and explored in three executed notebooks under [`notebooks/`](notebooks/).

## Quick start

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management (no `make` dependency — plain `uv run` commands work everywhere, including Windows).

```
uv sync --dev          # create .venv and install dependencies
uv run pytest          # run tests
uv run ruff check .    # lint
uv run ruff format .   # format
uv run pre-commit run --all-files   # run all pre-commit hooks
```

## Reproducing the warehouse

Nothing under `data/` is committed except `.gitkeep`, so a fresh clone rebuilds every layer from the original source files:

```
uv run python -m hub.ingestion.download --dataset online_retail_ii
uv run python -m hub.ingestion.bronze   --dataset online_retail_ii
uv run python -m hub.ingestion.silver
uv run python -m hub.ingestion.gold
uv run python -m hub.features.snapshots
```

Olist, the secondary dataset, also needs a one-time Kaggle API token; `hub.ingestion.download` prints the exact setup steps when it is missing. The whole chain above runs in under a minute on ~1M rows.

| Layer | Table | Rows |
|---|---|---|
| bronze | `online_retail_ii` — both source sheets stacked, ingestion metadata only, nothing else touched | 1,067,371 |
| silver | `online_retail_ii` — cleaned, typed, flagged; 34,335 exact duplicates dropped | 1,033,036 |
| gold | `fact_order_lines` | 1,033,036 |
| gold | `fact_orders` | 53,628 |
| gold | `dim_customer` | 5,942 |
| gold | `dim_product` | 5,304 |
| gold | `dim_date` | 739 |
| gold | `customer_snapshots` — 23 monthly snapshots, 76,261 rows with a complete label window | 93,398 |

Revenue ties out exactly at all three grains: £18,855,533.70 at line level, at order level, and at customer level once the unattributable null-`Customer ID` revenue (£2,565,542.41, 22.8% of rows) is added back. Every layer is validated against a pandera contract and fails loudly rather than writing a table that quietly double-counts.

### How leakage is prevented

`customer_snapshots` freezes the clock. Each row is one customer at one monthly snapshot date: every feature is computed only from rows strictly *before* that date, every label only from the 90 days *on or after* it, and no row contributes to both sides. Snapshots whose label window runs past the end of the data are kept but flagged `is_label_window_complete = FALSE`, so right-censored labels can never be mistaken for churn.

This is enforced by test, not by convention: `tests/test_snapshots.py` rebuilds the table with a purchase inserted after a snapshot and asserts the feature columns come back identical, then asserts they *do* change once that purchase is genuinely history — so the test cannot pass vacuously.

### Documented decisions

The reasoning behind each layer lives next to the code that implements it, because that is where it stays accurate:

- `src/hub/ingestion/silver.py` — nine numbered cleaning decisions with measured row-count and revenue impact, including why cancellations and bad-debt adjustments are flagged rather than dropped, and why wholesale quantities are flagged instead of winsorized.
- `sql/gold/*.sql` — one header comment per model recording the grain and the data quirks that forced it, such as the 83 invoices whose line timestamps drift by up to nine minutes, and the 13 customers who ship to more than one country.
- `docs/case_study.md`, `docs/metrics_dictionary.md` — the stakeholder, the three business questions, and every metric defined before it is computed.

## Reproducing the analysis

```
uv run python -m hub.analytics.report   # analysis.json, tables and figures (~30s)
uv run pytest -q                        # 74 tests
```

`hub.analytics.report` is the single source of truth for the Phase 3 numbers. The
notebooks load its tables instead of re-deriving them, so the narrative cannot drift
from the data. It writes `reports/analysis.json`, regenerable tables under
`reports/tables/`, and 12 figures under `reports/figures/`.

| Notebook | Focus |
|---|---|
| [`01_business_overview.ipynb`](notebooks/01_business_overview.ipynb) | revenue, order value, seasonality, trading hours, customer concentration |
| [`02_customer_retention.ipynb`](notebooks/02_customer_retention.ipynb) | purchase intervals, churn definition and sensitivity, cohort retention and value |
| [`03_product_and_baskets.ipynb`](notebooks/03_product_and_baskets.ipynb) | product concentration, returns, market-basket association rules |

The quantified one-page summary is [`docs/key_findings.md`](docs/key_findings.md).

Full README (architecture, headline results, model comparison, honest limitations) will be written once the platform has something to show.
