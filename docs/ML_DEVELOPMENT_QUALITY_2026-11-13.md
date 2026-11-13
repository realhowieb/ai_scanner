# Development feature audit and prospective cohort

Production AI Confidence and PreBreakout are unchanged. The previous final test remains consumed. This work neither fits a model nor opens new test outcomes.

## Scope and findings

[Read-only audit](https://github.com/realhowieb/ai_scanner/actions/runs/43493917655) reproduced exactly the original 1,003 development/training signal-days, September 14–October 21, matching both the saved ID hash and full feature/label slice fingerprint. SQL stopped before October 22; the October 30 final test and future observations were not read by this audit.

| Production input | Missing overall | Missing early | Missing late | Descriptive PSI |
|---|---:|---:|---:|---:|
| 10-day trend | 74.28% | 95.59% | 63.35% | 0.8114 |
| 20-day trend | 74.28% | 95.59% | 63.35% | 0.8988 |
| Relative volume | 70.49% | 84.41% | 63.35% | 0.4371 |
| Dollar volume | 74.28% | 95.59% | 63.35% | 0.7771 |
| Scanner breakout score | 70.49% | 84.41% | 63.35% | 0.6881 |
| Scan gap | 70.49% | 84.41% | 63.35% | 0.5067 |

Early/late splits use the midpoint of unique development entry days, not an optimized date. Missing is a separate PSI bin; these are descriptive distribution differences, not significance tests or proof of a market-regime cause. The population, data capture and market state can all contribute. No numeric input is globally constant, but categorical coverage is sparse: EMA cross has only 33 observed values. Pattern-tag coverage and snapshot sizes also change over time.

Weekly development target prevalence varies: W38 29.4%, W39 9.1%, W40 26.6%, W41 21.3%, W42 31.0%, W43 34.7%. That is evidence of instability in this small sample, not a reason to tune a different target against the consumed final test.

For the 407 development rows carrying captured PreBreakout provenance, many history-derived fields are defaulted in every captured row: score/trend/volume deltas and slopes, compression and low/high structure features. This concerns the separate PreBreakout input contract; the ML v4 six-input recipe is not trained on that entire vector. Unrecorded provenance stays unknown. Zero-valued observed features are never assumed to be defaults.

The correct conclusion is that the evaluated recipe did not validate on this dataset. These results do not establish that the underlying inputs can never predict the target. Historical gaps must not be filled with present-day information.

## Prediction-time input cross-check

The [follow-up development-only audit](https://github.com/realhowieb/ai_scanner/actions/runs/43496346130) found, **for each of the six core inputs**, 169 rows where the canonical feature is missing but an observed, non-defaulted value exists in the captured PreBreakout input vector. Another 238 rows overlap. Five fields have two disagreements in that overlap; scan gap has none.

This identifies a reader/source-contract issue worth investigating before adding more estimators. It does not authorize blind filling: establish exact source, scan identity, timing and units, resolve the disagreements, then define a separate versioned development feature recipe. No values were filled and no model was refitted for this cross-check.

## New inputs

An [observation-only capture audit](https://github.com/realhowieb/ai_scanner/actions/runs/43494616299) found RSI and SPY/SMA20 regime capture beginning November 2, across 10 trading days as of November 13. All 2,800 CANDIDATE observations in that period contain both fields. Across CANDIDATE, CONTROL and NEAR_MISS, RSI exists on 6,998 of 7,000 observations and regime on all 7,000.

These fields live in immutable scheduled research metadata. They are **not** currently in the ML v4 feature schema/Encoder. A versioned reader/feature-set change and development-only ablation are needed before a challenger can use them. Their existence or 20 days of coverage is not evidence of predictive value. The stale schema/audit descriptions have been corrected. The feature schema itself remains v1 and no new training input is silently added.

## Agreed prospective protocol

The initial November 16 test proposal was amended before registration and before any collection, following the capture audit and the owner's new-input-first decision. No prospective outcome was inspected to make this change.

- New development collection: November 16–December 14 (20 fresh trading sessions). Existing scheduled scans already save these observations. This period is not the test.
- Last five-day development windows mature by December 21; December 22 provides an embargo/buffer. Lock one primary target, exact feature schema, model, parameters and evaluation settings before test start.
- Untouched test: December 23–January 22 (20 trading sessions, excluding exchange holidays).
- Earliest complete review: February 2, 2027, allowing the final five-day outcome window and a reporting buffer. At least 300 matured signal-days is the minimum planned sample, not a guarantee of statistical power or an automatic pass.
- Fixed end. Never shorten or extend based on observed results. An insufficient sample is inconclusive.

The complete protocol is `ml_forward_cohort_2026-12-23.json`.

## Collection and access controls

Registration is insert-once and must precede the window. The collector stores all public Scanner identities plus whitelisted prediction-time fields; it never selects returns, extrema, labels or outcome timestamps. Snapshots are append-only, and repeated collection is idempotent. First-per-ticker/UTC-day counts are reported without looking at outcomes. Original source scan records remain immutable and can later provide versioned RSI/regime features via their saved scan identity.

The entire reserved time interval is quarantined from development before outcome reads, even if the collector has not run yet. The ML v4 loader reads only frozen membership instead of the entire database window; new development freezes exclude reserved IDs using identities and timestamps only. No automatic release or evaluation path is implemented. A future explicit release must verify a pre-test candidate/target lock, adequate maturity and the durable final-test ledger before outcome access. Do not use unrelated admin/outcome tools to peek at this cohort for model selection.

`ML Forward Cohort Collection` runs daily at 23:50 UTC on weekdays once deployed to main; fixed membership boundaries are enforced in the database code. Registration is a separate explicit manual action. It does not change scores, alerts, plans or production models.

## Prioritized development work

1. Reconcile canonical scan joins with genuinely captured model inputs, including the two overlap disagreements. Keep unknown/defaulted values distinct and never substitute present-day data.
2. Add RSI and versioned SPY/SMA20 context through an explicit research schema/feature set, with coverage and as-of tests. Do not change production scoring.
3. Compare predeclared feature groups on the fresh development period, including within-day ranking and a time/base-rate baseline. Changing capture completeness can otherwise masquerade as predictive signal in pooled metrics.
4. Lock one target and recipe before December 23. Only after the untouched period matures and access is explicitly reviewed should the final-test ledger be used again, on the new cohort only.
