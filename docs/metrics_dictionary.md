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
**excluded from product-level analysis** but a documented decision (made in Phase 2,
silver layer) governs whether they're included in top-line revenue — because things like
postage are real cash the business collected.

### Revenue basis: gross vs net

Two sums are reported throughout and are never interchanged:

- **Gross revenue** — sum of line revenue over lines that are neither a cancellation
  (`C` invoice) nor a bad-debt adjustment (`A` invoice). Non-negative by construction
  (every ordinary line has `quantity > 0` and `unit_price >= 0`). Used for "sales"
  language and wherever a ratio needs a non-negative denominator (e.g. Gini).
- **Net revenue** — sum of line revenue over **every** line, so cancellations and
  adjustments subtract. This is the true cash effect and the only number allowed to be
  called "revenue" in a headline total.

Any metric below states which basis it uses. A metric that does not state its basis is
under-specified and must not be reported.

### Order
One distinct `InvoiceNo`. An order with only non-product lines (e.g. a `BANK CHARGES`
adjustment) is not counted as a product order for AOV purposes.

### Purchase occasion
One customer on one calendar day with at least one **non-reversal** order (neither `C`
nor `A`). Used for every behavioural metric: frequency, repeat rate, inter-purchase
intervals, cohort activity, and churn.

Why a day rather than an invoice: 3,008 customer-days carry more than one invoice,
amounting to 3,863 extra invoices (7.2% of all orders). Those are one shopping visit
recorded as several baskets, not separate returns to the shop. Counting them as separate
purchases would create zero-length "intervals" and inflate repeat rates. Revenue is
still summed at line/order grain — collapsing to a day applies to *behaviour* only.

A cancellation is **not** a purchase occasion: a refund is a reversal of an earlier
visit, not a new one. It is still counted in net revenue, in the month it occurs.

### AOV — Average Order Value
`product revenue / number of distinct product orders`, over a stated time window, using
**ordinary orders only** (no `C`, no `A`) that contain at least one product line. Product
revenue excludes postage and fee codes, so AOV measures merchandise basket size rather
than shipping mix. Postage-inclusive revenue is reported separately if ever needed, and
labelled as such.

### Median order value
Median product revenue across the same ordinary product orders used for AOV. Reported
alongside every AOV because the basket distribution is heavily right-skewed by wholesale
orders: the mean says what an order is worth on average, the median says what a typical
order looks like, and quoting only the mean overstates the typical gift-shop basket.

### ARPU — Average Revenue Per User
`net revenue attributable to identified customers / number of identified customers with
at least one purchase occasion in the window`, over a stated time window.

- Numerator: net revenue (cancellations subtract) from orders carrying a non-null
  `Customer ID`, **restricted to the same customers counted in the denominator**, in the
  window. Using the same customer set on both sides stops a refund by a customer who did
  not otherwise buy in the window from dragging ARPU down.
- Denominator: distinct customers with ≥1 purchase occasion in the window — not all
  customers who ever existed, which would make ARPU move when old customers stop
  ordering rather than when spend changes.
- Null-`Customer ID` revenue is excluded from both sides; it is unattributable.

**Always state the window** (e.g. "ARPU, trailing 90 days") — ARPU with no window
attached is meaningless and will not appear undated anywhere in this project.

Customers with a null `Customer ID` (~20–25% of rows in Online Retail II) are excluded
from every customer-level metric (ARPU, CLV, churn, segmentation) but retained in
order-level and revenue totals. This is a documented policy, not an oversight — see the
silver-layer cleaning notes.

---

## Customer value & retention metrics

### Repeat Purchase Rate
Of customers with at least one **purchase occasion** in a cohort/window, the fraction
with **more than one** purchase occasion (distinct days, per the definition above) in
that same window. For cohort analysis the second occasion may fall in any later period.

Reported as `repeat rate = repeaters / buyers`. Single-purchase customers are included
in the denominator — excluding them would define the metric as "share of repeaters who
repeat", which is circular.

### Purchase interval
Time in days between a customer's consecutive **purchase occasions**. The **distribution**
of this value (not just its mean) is what determines the churn window (see below) — a
long-tailed distribution means a single fixed window is a simplification worth stating
explicitly.

Interval statistics are reported over repeat buyers only, because a customer with one
occasion has no interval. That is a censoring bias, not a property of the population: a
first-time buyer who never returns contributes no observation, so the observed interval
distribution is **shorter** than the true one. The share of customers with only one
occasion is therefore reported alongside every interval figure.

### Tenure
Days between a customer's first observed order and the snapshot date used for a given
feature/label computation.

### Return rate
Three variants exist because they answer different questions. Each is named explicitly
in any output; "return rate" alone is not a valid label.

- **Unit return rate** — `cancelled units / purchased units`, where cancelled units are
  the absolute value of quantity on `C` lines and purchased units are quantity on
  ordinary (`C`- and `A`-free) lines. The default for product and customer diagnostics.
- **Value return rate** — `|revenue on C lines| / gross revenue`. What refunds cost as a
  share of what was sold.
- **Line return rate** — `C lines / ordinary lines`. Reported for comparability with
  `dim_product.cancellation_rate`, which is a *share of all lines* (including
  cancellations in its denominator) and is **not** the same number.

A cancellation can reference an order placed before the dataset begins, so at very fine
granularity (one customer, one month) the ratio can legitimately exceed 1. Any value
above 1 is reported as-is rather than clipped; clipping would hide the data-boundary
effect.

### Cohort (left- and right-censoring)
Customers grouped by the calendar month of their **first observed purchase occasion**.
Two boundaries are handled explicitly rather than left to look like behaviour:

- **Left censoring.** The dataset starts 2009-12-01, so the 2009-12 cohort contains
  every customer who was already active before the window opened. It is flagged
  `is_left_censored = TRUE` and excluded from headline retention claims, because its
  "new customers" were not observed being acquired.
- **Right censoring.** A cohort's month-*k* retention is only observable when month
  *c* + *k* has fully elapsed within the data. Unobservable cells are **NaN, never 0** —
  rendering them as zero would invent a collapse in retention that the data cannot show.

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
is a modeling choice, not a fact. **Finalized in Phase 3 from the empirical
purchase-interval distribution** (see the change log for the evidence):

> A customer **active** as of snapshot date *t* is **churned** if they place zero
> purchase occasions in the window `[t, t + W]`, with **W = 90 days**.

- **Active at *t*** means at least one purchase occasion in `[t − 90, t)` — the same
  length as the label window, so the definition is symmetric. It is implemented as
  `was_active_in_prior_window` in `customer_snapshots.sql`.
- **W = 90 days** captures 80.7% of all observed repeat-purchase intervals. The observed
  distribution is median 31 days, p75 70, p90 147, p95 221.
- The label means **"did not transact in the next quarter"**, not "permanently lapsed".
  19.3% of observed repeat purchases arrive after 90 days, and the interval distribution
  is itself right-censored (28.8% of customers have only one purchase occasion and
  contribute no interval), so the true share is higher. The decision layer must not treat
  the label as proof that a customer is gone.
- **Sensitivity** (pooled across monthly snapshots, right-censored snapshots excluded,
  lookback `L = W`): churn rate among active customers is **61.5% at W = 30, 48.6% at
  60, 42.9% at 90, 39.7% at 120, 37.9% at 150, and 36.7% at 180** (snapshot counts 22,
  21, 20, 19, 18, 17 respectively, because a longer window censors more of the recent
  grid). The curve is steep between 30 and 90 days and flattens afterwards, so the
  choice inside the 90–180 range changes the level by ~6 points but not the ordering.
  90 is retained because it sits just past the flattening point and matches the quarterly
  budget cycle and the 90-day revenue question in `case_study.md`.
- **Every snapshot in the comparison is left- and right-censored consistently:** a
  snapshot is only scored when `[t, t + W)` lies entirely inside the observed data, and
  eligibility uses the same `W`-day lookback. Comparing windows with different censoring
  rules was the source of the earlier, higher sensitivity figures (see the change log).
- Rows whose label window extends past the last observed transaction date are retained
  but flagged `is_label_window_complete = FALSE`. They are **right-censored** and must
  be excluded from training and evaluation — otherwise a customer whose window is only
  half-observed is scored as churned by default.
- **Reversals are not activity.** Cancellation (`C`) and adjustment (`A`) lines are
  excluded from both sides of the definition: `was_active_in_prior_window` requires a
  non-reversal exception, and `future_orders` counts non-reversal invoices only. Counting
  a refund as a "purchase occasion" would make a churned customer look retained.

### CLV — Customer Lifetime Value
Two distinct things, never conflated:

- **Historical CLV**: total realized revenue from a customer, over their full observed
  history to date. A backward-looking fact, used for profiling/segmentation.
- **Predicted CLV**: expected *future* revenue over a stated horizon (default: 90 days,
  matching the business questions), produced by a model (BG/NBD + Gamma-Gamma in Phase
  5A, or the ML regression target in Phase 5B). Always reported with its horizon and
  which model produced it.

### P(alive)
Output of the BG/NBD model: the model's estimated probability that a customer has not
yet permanently stopped purchasing, as of the snapshot date. Distinct from churn
probability (which is defined over a fixed forward window) — P(alive) has no fixed
horizon by construction.

---

## Concentration metrics

### Pareto share
"Top X% of customers generate Y% of revenue", computed over **net revenue per customer
within a stated window**. Customers are sorted by net revenue descending and the cumulative
revenue share is taken; X is the smallest customer count whose cumulative share reaches
the target (default 80%). Y is therefore computed, never assumed.

Customers with negative net revenue (returned more than they bought) sit at the bottom of
the ordering and are included in the total. Because the 80% crossing necessarily happens
inside the positive prefix, their presence changes the denominator but not which customers
are named as the top contributors. Their count and combined revenue are reported separately.

### Gini coefficient
Standard Gini of the customer-level **net revenue** distribution, over a stated window:

`G = Σᵢ Σⱼ |xᵢ − xⱼ| / (2 · n² · x̄)`

Gini requires a non-negative distribution. Net revenue is not guaranteed non-negative —
83 identified customers have negative net revenue — so those values are **clipped to zero**
for this metric only, and the count and combined revenue of the clipped customers are
reported alongside. Clipping is chosen over dropping them because dropping would silently
shrink the customer population and understate concentration.

`G = 0` means perfectly equal revenue per customer; `G → 1` means one customer holds all
of it. Reported with the window and the customer count, since Gini over 50 customers is
not comparable to Gini over 5,000.

### Product revenue concentration
Same Pareto procedure applied to `dim_product.net_revenue` restricted to `is_product =
TRUE`, and reported with the number of distinct product codes needed to reach the target.
Non-merchandise codes are excluded here because "which products drive revenue" is a
catalogue question.

---

## Seasonality metrics

All seasonality is computed on **net revenue per calendar day over a contiguous daily
index**, zero-filled from the first to the last day with any transaction. Zero-filling is
not cosmetic: the retailer is closed on Saturdays (104 of 105 Saturdays have no
transactions at all — average daily revenue £93 versus £35,134 on Tuesdays), so a
"revenue by weekday" table built only from days with sales would delete the single
largest weekly effect and report Saturday as if it did not exist.

- **Day-of-week effect** — mean net revenue per weekday across all calendar days of that
  weekday, and the ratio to the all-day mean. Zero days included.
- **Hour effect** — share of orders by invoice hour. Timestamps are local wall-clock with
  no timezone field; they are used as recorded. (Only 146 of 53,628 orders sit exactly at
  12:00:00, so the hour distribution is not materially inflated by default timestamps.)
- **Q4 lift** — for each calendar year with at least one non-Q4 day, `mean daily net
  revenue in Oct–Dec / mean daily net revenue outside Oct–Dec`. A year with no non-Q4
  days (2009, which begins in December) has no defined ratio and is reported as **not
  available**, never as 0 or ∞.
- **STL decomposition** — additive STL on the zero-filled daily series with a weekly
  period (7) and `robust=True`, since holidays and one-off wholesale orders are exactly
  the kind of outlier that would otherwise bend the trend line. Seasonal strength is
  reported as `max(0, 1 − Var(residual) / Var(residual + seasonal))`; 0 means the weekly
  pattern explains nothing, 1 means it explains everything.

---

## Market-basket metrics

Baskets are built from **ordinary product lines only** (`is_product`, neither `C` nor
`A`), at `invoice_no` grain, de-duplicated to one row per (invoice, stock code). An
"item" is a stock code, not a product name — descriptions are not 1:1 with codes.

For items A and B, over *N* baskets:

- **Support(A)** = baskets containing A / N.
- **Support(A, B)** = baskets containing both / N.
- **Confidence(A → B)** = Support(A, B) / Support(A) = P(B present | A present).
- **Lift(A → B)** = Confidence(A → B) / Support(B).

Lift > 1 means A and B co-occur more often than independence predicts; lift = 1 means no
association; lift < 1 means they substitute. **Both directions are reported** because
confidence is asymmetric and the pair is not.

Thresholds are explicit and reported with every result: minimum item support 0.1% of
baskets for an item to enter the candidate set, minimum pair support 0.2%, and minimum
confidence 0.05. Rules are ranked by lift, and the support and confidence that produced
each rule are always shown — a high-lift rule on 40 baskets is a curiosity, not a
merchandising plan.

---

## Model evaluation metrics

### Classification (churn)
- **PR-AUC**: area under the precision-recall curve — preferred over ROC-AUC here
  because churn is imbalanced.
- **Calibration / Brier score**: how well predicted probabilities match observed
  frequencies. A model isn't allowed to feed the decision layer (Phase 7) until it's
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
3) must be a dated entry in this section, not a silent edit — models trained under one
definition and evaluated under another is exactly the kind of bug this dictionary exists
to prevent.

- **2026-09-29** — initial version. Churn window left as placeholder (90 days), pending
  empirical purchase-interval analysis in Phase 3.
- **2026-11-05** — Phase 3 definitions added: revenue basis (gross vs net), purchase
  occasion, return rate (unit/value/line), cohort censoring, Pareto/Gini conventions,
  seasonality, and market-basket rules. **Behavioural metrics moved from invoice grain to
  purchase-occasion (customer-day) grain**: 3,008 customer-days contain multiple invoices
  (7.2% of orders), and counting each invoice separately produces zero-length intervals
  and inflates repeat rates. Revenue metrics are unchanged and still sum at line grain.
- **2026-11-05** — **churn window W finalized at 90 days** from 27,231 observed
  inter-purchase intervals, contributed by the 4,189 customers (of 5,881 with at least one
  purchase occasion) who have a second occasion to measure. The remaining 61 of the 5,942
  identified customers appear only on cancellation invoices. Interval distribution:
  median 31 days, p75 70, p90 147, p95 221; 90 days covers 80.7%. Churn-rate sensitivity
  across W ∈ {30, 60, 90, 120, 150, 180} is 61.5% → 36.7%, i.e. the target is stable.
  Active lookback L set equal to W (90 days). The snapshot table already implements both;
  no rebuild of the label horizon was required.
- **2026-11-05** — return rates use **signed** cancellation sums, negated. One `C` line
  (`C496350`, stock code `M`, manual charge) carries a positive £373.57, i.e. it reverses
  a charge rather than refunding a sale. Taking absolute values would overstate refunded
  units by 2 and refunded value by £747.14; negating the signed sum treats the reversal
  correctly.
- **2026-11-06** — **churn sensitivity figures corrected.** The Phase 3 probe that
  produced the original 50.7% (W = 60) → 40.2% (W = 180) range did not apply the
  right-censoring rule uniformly: it scored snapshots whose label window extended past
  the last observed transaction, which mechanically depresses the churn rate for the
  short windows. Re-running with censored snapshots excluded and `L = W` on both sides
  gives 61.5% (30), 48.6% (60), **42.9% (90)**, 39.7% (120), 37.9% (150), 36.7% (180).
  The SQL snapshot table and the pandas `churn_sensitivity` function now agree at 42.9%
  for W = 90, which is the regression check that caught the discrepancy.
- **2026-11-06** — `was_active_in_prior_window` now excludes cancellation and adjustment
  lines. Previously a customer whose only recent invoice was a refund was scored as
  active, biasing the churn rate upward at the eligibility stage; the label side already
  excluded reversals, so the two sides were inconsistent.
