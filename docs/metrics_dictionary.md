# Metrics Dictionary

Every metric used anywhere in this project — code, notebooks, dashboard, reports — must
be defined here **before** it's computed. If a metric in the code doesn't match its
definition here, the code is wrong, not the dictionary.

Scope note: definitions below apply to the primary dataset (Online Retail II) unless
stated otherwise. See [`case_study.md`](case_study.md) for why Olist is excluded from
customer-level metrics.

---

## Revenue & order metrics

### Revenue
`quantity * unit_price`, summed at whatever grain is requested (order, customer, day,
segment). **Net of cancellations**: any invoice whose number starts with `C` reverses a
prior order and its line-level revenue is subtracted, not dropped — so cancelled orders
still net to ~zero rather than silently disappearing from totals.

Non-product stock codes (`POST`, `M`, `BANK CHARGES`, `DOT`, `D` for discounts, etc.) are
**excluded from product-level analysis** but a documented decision (made in Phase 5,
silver layer) governs whether they're included in top-line revenue — because things like
postage are real cash the business collected.

### Order
One distinct `InvoiceNo`. An order with only non-product lines (e.g. a `BANK CHARGES`
adjustment) is not counted as a product order for AOV purposes.

### AOV — Average Order Value
`total revenue / number of distinct orders`, over a stated time window. Computed at
product-order grain (see above).

### ARPU — Average Revenue Per User
`total revenue / number of distinct customers with a non-null Customer ID`, over a
stated time window. **Always state the window** (e.g. "ARPU, trailing 90 days") — ARPU
with no window attached is meaningless and will not appear undated anywhere in this
project.

Customers with a null `Customer ID` (~20–25% of rows in Online Retail II) are excluded
from every customer-level metric (ARPU, CLV, churn, segmentation) but retained in
order-level and revenue totals. This is a documented policy, not an oversight — see the
silver-layer cleaning notes.

---

## Customer value & retention metrics

### Repeat Purchase Rate
Of customers with at least one order in a cohort/window, the fraction who placed **more
than one** order in that same window (or, for cohort analysis, in any period after
acquisition). Always stated with its window and cohort definition attached.

### Purchase interval
Time in days between a customer's consecutive orders. The **distribution** of this
value (not just its mean) is what determines the churn window (see below) — a long-tailed
distribution means a single fixed window is a simplification worth stating explicitly.

### Tenure
Days between a customer's first observed order and the snapshot date used for a given
feature/label computation.

### Cohort
Customers grouped by the calendar month of their **first ever order**. A customer
belongs to exactly one cohort for the lifetime of the analysis.

### Retention rate (cohort)
For cohort *C* acquired in month 0, the retention rate at month *k* is:
`(# customers in cohort C who placed >=1 order in month k) / (# customers in cohort C)`.
Rendered as a triangular heatmap (month-0 cohort down the rows, months-since-acquisition
across the columns).

### Revenue retention (cohort)
Same cohort structure, but the numerator is `revenue from cohort C in month k` instead of
customer count — shows whether the *customers who remain* are also worth as much, which
plain logo retention hides.

### Churn (customer-level, non-contractual)
**Not directly observed** — there is no cancellation event in this data, so "churned"
is a modeling choice, not a fact. Working definition, to be finalized empirically in
Phase 7 (Metrics & EDA) from the purchase-interval distribution:

> A customer **active** as of snapshot date *t* is **churned** if they place zero orders
> in the window `[t, t + W]`, where `W` is the churn window (placeholder: **90 days**,
> matching the horizon in the business questions in `case_study.md`).

"Active as of *t*" itself requires a definition (e.g. placed >=1 order in the `L` days
before *t*) — this is finalized alongside `W` in Phase 7, not assumed here.

### CLV — Customer Lifetime Value
Two distinct things, never conflated:

- **Historical CLV**: total realized revenue from a customer, over their full observed
  history to date. A backward-looking fact, used for profiling/segmentation.
- **Predicted CLV**: expected *future* revenue over a stated horizon (default: 90 days,
  matching the business questions), produced by a model (BG/NBD + Gamma-Gamma in Phase
  10, or the ML regression target in Phase 11). Always reported with its horizon and
  which model produced it.

### P(alive)
Output of the BG/NBD model: the model's estimated probability that a customer has not
yet permanently stopped purchasing, as of the snapshot date. Distinct from churn
probability (which is defined over a fixed forward window) — P(alive) has no fixed
horizon by construction.

---

## Concentration metrics

### Pareto share
"Top X% of customers generate Y% of revenue" — X is usually fixed at 20% by convention;
Y is computed, not assumed.

### Gini coefficient
Standard Gini coefficient of the customer-level revenue distribution (0 = perfectly
equal, 1 = maximally concentrated). Computed over historical CLV per customer within a
stated window.

---

## Model evaluation metrics

### Classification (churn)
- **PR-AUC**: area under the precision-recall curve — preferred over ROC-AUC here
  because churn is imbalanced.
- **Calibration / Brier score**: how well predicted probabilities match observed
  frequencies. A model isn't allowed to feed the decision layer (Phase 14) until it's
  calibrated (isotonic/Platt), because uncalibrated scores can't be multiplied by
  dollars.
- **Top-decile lift**: `(actual churn rate in the model's riskiest 10%) / (overall churn
  rate)`. The headline number for "is this model useful."

### Regression (CLV)
- **MAE / RMSE**: standard, reported alongside the label's own scale (zero-inflated
  next-90-day revenue) since RMSE on a zero-inflated target is easy to misread.
  **Rank correlation** (Spearman): whether the model orders customers correctly even if
  absolute dollar predictions are off — often the more decision-relevant number.

### Forecasting
- **MASE** (Mean Absolute Scaled Error): primary metric, scale-free, comparable across
  models and against the seasonal-naive baseline (MASE < 1 means "better than
  seasonal-naive").
- **sMAPE**: secondary, reported for comparability with external benchmarks.
- **Prediction interval coverage**: fraction of backtest folds where the actual value
  fell inside the model's stated interval. A "95% interval" that covers 60% of actuals is
  reported as broken, not rounded up.

---

## Change log

Any redefinition of a metric here (especially the churn window `W`, finalized in Phase
7) must be a dated entry in this section, not a silent edit — models trained under one
definition and evaluated under another is exactly the kind of bug this dictionary exists
to prevent.

- **2026-09-29** — initial version. Churn window left as placeholder (90 days), pending
  empirical purchase-interval analysis in Phase 7.
