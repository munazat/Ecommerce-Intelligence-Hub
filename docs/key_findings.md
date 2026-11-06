# Key findings

Each phase has its own reproducible runner, and every number below comes from it. Metrics
are defined in [`metrics_dictionary.md`](metrics_dictionary.md) before they are computed;
method detail for segmentation is in [`segmentation.md`](segmentation.md).

| Phase | Runner | Scope |
|---|---|---|
| 3 — revenue, retention, assortment | `hub.analytics.report` | 1,033,036 lines, 53,628 invoices, 5,942 customers, 2009-12-01 → 2011-12-09 |
| 4 — customer segmentation | `hub.analytics.segment_report` | 5,251 buyers at the 2011-09-01 snapshot |

---

## Phase 3 — Revenue, retention and assortment

### Headline

| Headline | Value |
|---|---|
| Net revenue | £18,855,534 |
| Gross revenue / reversals | £20,465,198 / −£1,609,665 |
| Product revenue (ordinary orders) | £19,643,862 |
| AOV / median order value | £438.98 / £255.76 |
| Repeat purchase rate | 71.2% (4,189 of 5,881 buyers) |
| 90-day churn (pooled, 20 snapshots) | 42.9% |

### Six findings that change decisions

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
   ranges 31.3%–57.9%, so any churn model must compare like-for-like seasons rather
   than pooled months.

6. **Assortment is concentrated and baskets are built on matching sets.** 1,048 of
   5,278 products (19.9%) generate 80.0% of product revenue (Gini 0.770). Overall value
   return rate is 7.14%, but two bulk-cancelled products account for 34.3% of cancelled
   value with 100% and 94.8% value return on 1-of-1 and 10-of-250 lines — a cash
   artefact, not a quality signal. The strongest basket pairs are herb markers
   (93.2% confidence, 165× lift) and Regency tea plates (91.1%); they appear in only
   ~0.5% of baskets, so they deepen existing baskets rather than create mass promotions.

### What to do next from Phase 3

- Build the churn model against the 90-day label, excluding right-censored snapshots,
  and compare same-month snapshots for seasonality.
- Target second-purchase conversion for the 1,692 one-time buyers before broad win-back.
- Segment retention economics by customer tier; the top 1% justify account-level effort.
- Treat Q4 capacity, Thursday/Sunday cover and 09:00–17:59 service hours as fixed inputs
  to planning, not seasonal surprises.
- Test bundles around matching sets; use line return rate for quality reviews and value
  return rate for cash-exposure reviews.

---

## Phase 4 — Customer segmentation

### Headline

| Segment | Customers | Share | Lifetime gross | Next-90-day realised | Churn (active only) |
|---|---|---|---|---|---|
| Champions | 1,079 | 20.5% | 72.9% | 70.6% | 16.3% |
| Cooling mid value | 1,949 | 37.1% | 19.6% | 17.3% | 44.8% |
| Lapsed low value | 1,623 | 30.9% | 3.9% | 4.1% | 67.9% (53 active) |
| New / one-time | 474 | 9.0% | 1.9% | 7.4% | 45.6% |
| Returns-heavy accounts | 126 | 2.4% | 1.7% | 0.6% | 68.8% (16 active) |

Shipped model: K-Means, k = 5, silhouette 0.281, Davies-Bouldin 1.085. Population is every
customer with a purchase at the latest complete snapshot (5,251 of 5,314; 63 refund-only
accounts excluded), of whom 1,923 were active and 3,328 already lapsed.

### Five findings that change decisions

1. **Protect Champions with service, not discounts.** They are 20.5% of customers but
   72.9% of lifetime gross revenue and 70.6% of realised next-quarter revenue, at
   £1,731 per customer. Their churn is 16.3% — 26.6 points below the pooled 42.9% — so
   the marginal pound is better spent on retention operations than on price cuts that
   give away margin from customers who were going to buy anyway.

2. **The statistically optimal k is not the operationally useful one.** k = 4 wins
   silhouette (0.335 vs 0.281) and Davies-Bouldin (1.051 vs 1.085), but its 2,571-customer
   lapsed block holds 1,605 of the 1,623 lapsed one-time buyers (3.9% of revenue) together
   with 966 of the 1,949 cooling mid-value accounts (19.6% of revenue, 44.8% churn); the
   remaining 857 cooling accounts are folded into the high-value cluster. One group
   deserves a reactivation sequence, the other is a suppression list, and k = 4 makes them
   indistinguishable. k = 5 separates them and isolates a returns-heavy group for
   remediation, at a documented 0.054 silhouette cost. The full comparison curve and the
   k=4/k=5 cross-tab are written out so the override can be checked rather than trusted.

3. **The partition is stable, so it is not a random-start artefact.** 100 bootstrap
   refits give mean ARI 0.973, median 0.975, p10 0.956 and a worst case of 0.928; no
   refit fell below 0.5. The k = 5 solution is reproducible even though the silhouette
   is modest in absolute terms.

4. **Rule-based RFM is not a substitute.** The classic quintile RFM scheme agrees with
   the clustering at ARI 0.374. Four of its eight segments (2,495 customers, 47.5% of
   the base) contain zero active customers, so their churn rate cannot be measured at
   all; among the segments that can be scored, the rule-based churn spread is 33.4
   points versus 52.5 for the model. The rules also conflate size with value: RFM
   "Hibernating" holds 25.5% of customers but only 4.1% of revenue.

5. **The acquisition leak survives into the best model.** New / one-time customers are
   84.6% active yet still churn at 45.6%, and they contribute 7.4% of next-quarter
   revenue from 1.9% of lifetime revenue — the highest forward-to-lifetime ratio of any
   segment. A second purchase in the first quarter is still the single biggest lever,
   exactly as Phase 3's month-0 cohort finding predicted.

### Segment strategy

Illustrative net value is £184,273 over £2.64m of realised next-90-day revenue, and 87%
of it comes from Champions (£159,775). Cooling mid value adds £16,314, new customers
£8,349, and lapsed low value only £465 across 1,623 customers — a 0.5% reactivation
assumption on £67 of realised forward revenue per customer barely covers a £0.05
contact, so suppression is the correct action. Returns-heavy accounts are held at zero
assumed uplift (−£630 campaign cost) until the return pattern is explained; their 68.8%
churn may be a supply or fit problem that a discount would make worse.

### What to do next from Phase 4

- Run Champions on service and early-access, not discount; measure incremental spend
  against a holdout rather than against last quarter.
- Give the 1,949 cooling mid-value accounts a targeted reactivation sequence — they are
  the only large group with both meaningful revenue and a plausible win-back.
- Investigate the 126 returns-heavy accounts before marketing to them.
- Suppress lapsed low value; do not spend on the 30.9% of customers who contribute 3.9%
  of revenue.
- Evaluate Phase 5 churn and CLV models per segment, not only in aggregate: the same
  model error means different money in Champions and in lapsed low value.

---

## Reproduce

```
uv run python -m hub.analytics.report
uv run python -m hub.analytics.segment_report
uv run pytest
```

Outputs: `reports/analysis.json`, `reports/segments.json`, `reports/tables/` (gitignored,
regenerated), `reports/figures/`, and the four notebooks in `notebooks/`. The suite runs
90 tests.

## Honest limitations

- Churn is not observed; there is no cancellation event in this dataset. All churn
  numbers are conditional on the documented 90-day definition.
- The purchase-interval distribution is right-censored: 28.8% of buyers contribute no
  interval, so the true share of gaps beyond 90 days is understated.
- Association rules are co-occurrence, not causation; bundle lift must be tested.
- Churn labels exist only for customers active at the snapshot. In Phase 4 that means
  3,328 of 5,251 customers (63.4%) cannot be scored, and the rates for lapsed low value
  and returns-heavy rest on 53 and 16 active customers — they are directional, not
  precise.
- Segmentation stability is measured across bootstrap resamples of one snapshot
  (2011-09-01), not across seasons. Segment sizes may move with the Q4 peak.
- Strategy uplift figures are explicit assumptions, not causal estimates. There is no
  experiment in this dataset, so incremental revenue is a planning prior to be tested.
- Features are lifetime-to-date at the snapshot, so tenure, frequency and revenue are
  correlated; part of what the clusters call "value" is "length of relationship".
- The Olist late-delivery vs review-score side-quest was **not run**: it requires the
  Kaggle dataset and no local copy exists. It is excluded from these findings rather
  than estimated from partial data.
