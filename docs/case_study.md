# Case Study Brief

## Stakeholder

**Head of Growth / CRM Lead at a UK-based online gift & houseware retailer.**

This maps directly to the primary dataset: Online Retail II is a real, non-store online
retailer registered in the UK, selling mostly unique all-occasion gifts, with a customer
base skewed toward wholesalers but including a long tail of individual buyers, across
~1.07M transaction lines from Dec 2009 to Dec 2011.

Olist (Brazilian marketplace, ~100k orders) is used as a **secondary dataset** for
delivery and review-driven operations analysis only — see [Data Sources](#data-sources)
below for why it's not used for CLV/retention.

## The decision this stakeholder is making

Each quarter, the Head of Growth has a fixed retention/CRM budget and has to decide:

1. Which customers to spend it on (targeting).
2. How much to spend per customer (offer sizing, via expected value).
3. What revenue to expect next quarter regardless of that spend (baseline forecast),
   so the budget ask and the growth target are grounded in the same numbers.

Today this is done on gut feel and last-quarter's revenue trend. The project's job is to
replace that with a system that turns raw transaction history into a dollar-denominated
recommendation: *contact these customers, expect this much incremental profit, here's the
uncertainty.*

## The 3 business questions that drive everything

1. **Which customers are worth retaining, and how much should we spend on them?**
   (segmentation → CLV → churn risk → expected-value targeting)
2. **How much revenue will existing customers generate over the next 90 days?**
   (point-in-time CLV/churn models, leakage-free)
3. **What will total revenue look like next quarter, with uncertainty bands?**
   (backtested time-series forecasting)

Every phase of this project ladders up to one of these three questions. If a piece of
analysis doesn't answer one of them, it's EDA color, not a deliverable.

## Success criteria, in money and in numbers

A model or feature is "done" here only if it can be stated as a number the stakeholder
would act on:

- **Retention targeting:** the expected-profit optimizer must beat a naive "target
  everyone above a 0.5 churn-probability threshold" policy on expected profit, at the
  same budget. See Phase 7 (decision layer).
- **CLV / churn models:** must beat the BG/NBD probabilistic baseline and a naive
  recency rule on the agreed metrics (top-decile lift, calibration, MAE/RMSE) — or the
  write-up says plainly that they didn't, and why the simpler model is recommended
  instead.
- **Forecasting:** the selected model must beat seasonal-naive on MASE/sMAPE under
  rolling-origin backtesting, and its stated prediction interval must contain the truth
  at roughly its stated coverage (e.g. a 95% interval should contain actuals ~95% of the
  time across backtest folds).
- **Every metric used anywhere in the project is defined in
  [`metrics_dictionary.md`](metrics_dictionary.md) before it is computed.** No metric
  ships without a definition someone else could implement from scratch.

## Data sources

| Dataset | Role | Why |
|---|---|---|
| Online Retail II (UK gift retailer, ~1.07M rows, Dec 2009–Dec 2011) | Primary — all CLV, churn, segmentation, forecasting, decision-layer work | Real repeat-purchase behavior; a genuine non-contractual retention problem |
| Olist (Brazilian marketplace, ~100k orders, 9 tables) | Secondary — delivery time, review score, and operations analysis only | ~97% of Olist customers buy exactly once, so it cannot support CLV/retention/cohort work; using it there would produce a meaningless output (see the project roadmap's Reality Check section) |
| Synthetic marketing/traffic layer (optional, stretch) | If added: clearly labeled as synthetic, used only to demonstrate conversion-rate / funnel reasoning | Neither real dataset has sessions, traffic sources, or channels — conversion rate cannot be computed from them as-is, and it will never be presented as if it were real |

## Explicit non-goals

- **Not a live production system.** No real ad-platform or CRM integration; the FastAPI
  service and Streamlit dashboard are built to production standards but run against the
  batch datasets above.
- **Not a causal claim from observational data alone.** Churn-probability targeting
  identifies *who is likely to lapse*, not *who contacting would save* — that requires an
  uplift/experimental framing, which is treated explicitly as a separate, harder problem
  in the decision layer (Phase 7), not smuggled in as a side effect of the churn model.
- **Not claiming synthetic data is real.** Any simulated data (marketing layer, campaign
  response rates used in the profit optimizer) is labeled as an assumption or simulation
  everywhere it's used — in code, in docs, and in the dashboard.

## Open questions to revisit

- Exact churn window (30/60/90 days) — to be set from the empirical inter-purchase time
  distribution in Phase 3 (metrics/EDA), not assumed up front. Placeholder default: 90
  days, matching the horizon in the business questions above.
- Whether the synthetic marketing layer stretch goal is worth the time vs. spending that
  time deepening the core CLV/churn/forecast/decision chain. Default bias: skip it unless
  the core chain is done early.
