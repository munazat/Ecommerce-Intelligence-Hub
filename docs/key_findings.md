# Phase 3 — Key findings

**Scope:** 1,033,036 order lines, 53,628 invoices and 5,942 identified customers,
2009-12-01 → 2011-12-09 (Online Retail II). All figures are produced by
`uv run python -m hub.analytics.report`; every metric is defined in
[`metrics_dictionary.md`](metrics_dictionary.md) before it is computed.

| Headline | Value |
|---|---|
| Net revenue | £18,855,534 |
| Gross revenue / reversals | £20,465,198 / −£1,609,665 |
| Product revenue (ordinary orders) | £19,643,862 |
| AOV / median order value | £438.98 / £255.76 |
| Repeat purchase rate | 71.2% (4,189 of 5,881 buyers) |
| 90-day churn (pooled, 20 snapshots) | 42.9% |

## Six findings that change decisions

1. **Demand is a Q4 business, and the swing is large.** Q4 daily revenue is 1.62× the
   rest of 2010 and 1.92× in the partial 2011 Q4. The peak month (Nov 2011,
   £1,456,146) is 2.96× the trough (Apr 2011, £492,368). Purchasing, staffing and paid
   acquisition should be weighted to October–December; a flat monthly plan is wrong by
   roughly a factor of three at the extremes.

2. **Saturday is closed, and Thursday is the engine.** 104 of 105 Saturdays have zero
   revenue; the single trading Saturday makes the Saturday mean look non-zero (£93),
   which is why closure is detected by zero-day share. Thursday runs at 1.44× the
   average day. 94.9% of revenue is invoiced 09:00–17:59, peaking at noon. Service cover
   and campaign timing should follow that window.

3. **A very small customer base funds the business.** 59 customers (top 1%) generate
   31.3% of net revenue; 1,353 customers (22.8%) generate 80%. The customer Gini is
   0.740. Retention and account management should be tiered: losing a handful of
   wholesale accounts is a seven-figure event, while the long tail contributes
   single-digit percentages.

4. **The retention leak is in month 0, not month 12.** Cohort retention is 23.4% in
   month 1 and 22.7% in month 12 — the surviving base is stable, and almost all loss
   happens before the first repeat purchase. 28.8% of buyers never return. Onboarding
   and second-purchase conversion are the only high-leverage retention interventions.

5. **Churn is 42.9% per quarter and the definition is stable.** At W = 90 days, 42.9%
   of active customers have no purchase occasion in the next quarter. Sensitivity is
   61.5% at 30 days, 48.6% at 60, 39.7% at 120 and 36.7% at 180: the curve flattens
   after 90, so the decision does not hinge on the exact window. Snapshot-level churn
   ranges 31.3%–57.9%, so Phase 4 must compare like-for-like seasons, not pooled months.

6. **Assortment is concentrated and baskets are built on matching sets.** 1,048 of
   5,278 products (19.9%) generate 80.0% of product revenue (Gini 0.770). Overall value
   return rate is 7.14%, but two bulk-cancelled products account for 34.3% of cancelled
   value with 100% and 94.8% value return on 1-of-1 and 10-of-250 lines — a cash
   artefact, not a quality signal. The strongest basket pairs are herb markers
   (93.2% confidence, 165× lift) and Regency tea plates (91.1%); they appear in only
   ~0.5% of baskets, so they deepen existing baskets rather than create mass promotions.

## What to do next

- Build the churn model against the 90-day label, excluding right-censored snapshots,
  and compare same-month snapshots for seasonality.
- Target second-purchase conversion for the 1,692 one-time buyers before broad win-back.
- Segment retention economics by customer tier; the top 1% justify account-level effort.
- Treat Q4 capacity, Thursday/Sunday cover and 09:00–17:59 service hours as fixed inputs
  to planning, not seasonal surprises.
- Test bundles around matching sets; use line return rate for quality reviews and value
  return rate for cash-exposure reviews.

## Reproduce

```
uv run python -m hub.analytics.report
uv run pytest
```

Outputs: `reports/analysis.json`, `reports/tables/` (gitignored, regenerated),
`reports/figures/`, and the three notebooks in `notebooks/`.

## Honest limitations

- Churn is not observed; there is no cancellation event in this dataset. All churn
  numbers are conditional on the documented 90-day definition.
- The purchase-interval distribution is right-censored: 28.8% of buyers contribute no
  interval, so the true share of gaps beyond 90 days is understated.
- Association rules are co-occurrence, not causation; bundle lift must be tested.
- The Olist late-delivery vs review-score side-quest was **not run**: it requires the
  Kaggle dataset and no local copy exists. It is excluded from these findings rather
  than estimated from partial data.
