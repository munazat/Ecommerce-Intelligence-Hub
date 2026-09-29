# E-Commerce Intelligence Hub

End-to-end customer analytics platform: ingests raw e-commerce transaction data, validates and models it through a layered warehouse, predicts customer lifetime value and churn with probabilistic and ML models, forecasts revenue with backtested time-series models, and turns predictions into dollar-optimized retention decisions — served via an API, a dashboard, and auto-generated reports.

**Status:** early build-out. The foundation phase is in place — project tooling, continuous integration, the package skeleton, the case study brief and metrics dictionary, and a reproducible dataset download script.

## Quick start

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management (no `make` dependency — plain `uv run` commands work everywhere, including Windows).

```
uv sync --dev          # create .venv and install dependencies
uv run pytest          # run tests
uv run ruff check .    # lint
uv run ruff format .   # format
uv run pre-commit run --all-files   # run all pre-commit hooks
```

Full README (architecture, results, quick start with real data, model comparison) will be written once the platform has something to show — see the roadmap for the plan.
