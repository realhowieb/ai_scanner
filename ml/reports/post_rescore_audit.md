# HSF ML v3 Post-Rescore Integrity and Baseline Evaluation

## Decision

**DATA INTEGRITY: PASS. ML READINESS: NOT_READY for production-model validation or tuning.**
The production repair persisted successfully. It did not repair the missing
prediction provenance and target information required for an honest ML baseline.
No models were fitted, predictions regenerated, scores changed, or database writes
performed by this audit. Main remains unchanged.

## Evidence and scope

- Main inspected: `4866e642ff4f06fba731ff5ce71c3652cc1c54b1`.
- Repair apply: Actions run `37887421482`.
- Subsequent dry run: `37889726520`, successful, zero scorable rows and zero writes.
- Read-only production audit: `37890129203`, successful, code `a47c19d`.
- Full machine-readable results: `post_rescore_audit.json` alongside this report.
- Query: opportunity observations since September 1, before the repair start.
  Actual population: 1,041 rows. Read-only repeatable-read transaction, bounded
  query and timeout; provider reads happen after releasing the transaction.
- Recovered rows identified by outcome timestamps in the isolated repair interval
  October 9 05:12-05:16 UTC. Original empty-label timestamps were overwritten and
  cannot be reconstructed. Before coverage masks these rows' repaired fields;
  the repair UPDATE required all three original returns to be NULL.
- This is retrospective analysis of frozen values, not retrained walk-forward CV.
  No claim of independent held-out production-model performance is made.

## Integrity

| Check | Result |
|---|---:|
| Expected / found repaired observations | 613 / 613 |
| Complete repaired labels / benchmarks | 613 / 613 |
| Duplicate labels / benchmarks | 0 / 0 |
| Orphan labels / benchmarks | 0 / 0 |
| Snapshot epoch / observation timestamp mismatches | 0 |
| Remaining premature observations | 0 |
| Remaining unscorable | 16 |
| Awaiting outcome window | 75 |
| Canonical training-eligible outcome rows | 950 |

Labels and benchmark returns are columns on the same `signal_outcomes` row,
not child records requiring separate foreign keys. Logical uniqueness uses
source, source event, ticker and signal type. `source_event_id` is the snapshot
epoch timestamp, not an observation-table foreign key. The JSON records the
actual database constraints. Integrity PASS is structural, not certification of
price accuracy or model-target equivalence.

Idempotency: the subsequent canonical dry run selected only the 16 EA rows,
reported zero newly scorable rows, and wrote nothing. None of the repaired 613
remain premature. Recovery dates: September 14 = 608; September 28 = 5.

### EA failure

The provider returned 64 completed daily bars, May 4 through **August 4**.
The 16 observations require September 14 (5), 15 (5), 16 (5), and 17 (1).
No returned bar reaches any entry date. `_entry_position` therefore cannot find
an entry bar, and `score_signal` returns None, before it could evaluate the five
future sessions. See `analytics/signal_outcomes.py:16` and `:78`.
This proves the immediate data failure, not why the provider's coverage stops in
August. Provider listing/corporate-action coverage remains unverified; do not
infer a delisting or manufacture outcomes.

## Before and after

Both populations contain 1,041 observations. Labeled rows rise from 337 to 950:
**+613, +181.90%**. Label coverage rises from 32.37% to 91.26%.

| Horizon | Before + / - | After + / - | Before positive rate | After | Change (pp) |
|---|---:|---:|---:|---:|---:|
| 1 day | 200 / 137 | 422 / 528 | 59.35% | 44.42% | -14.93 |
| 3 days | 180 / 157 | 406 / 544 | 53.41% | 42.74% | -10.68 |
| 5 days | 190 / 147 | 473 / 477 | 56.38% | 49.79% | -6.59 |

Here positive means existing `return_h > 0`, not FutureQualitySetupHit.
Direction and recommendation were not recorded, so directional counts are
unavailable, not zero. Source is opportunity throughout. Date, horizon, tier,
setup, source and unavailable-direction/recommendation breakdowns are in JSON.

## Model performance and calibration

Active registry: PreBreakout id 13, `prebreakout-xgb-v9`, trained September 10
16:46 UTC, 75 features, target `FutureQualitySetupHit`. AI confidence id 3,
`ai-confidence-xgb-v1`, trained September 9, target `ForwardReturnHit`.

**Current production-model ROC AUC, PR AUC, accuracy, balanced accuracy,
precision, recall, F1, log loss and Brier: unavailable.**

Reasons:
- Neither model's target is simply positive close-to-close return.
- Zero observations freeze served model version; zero store AI confidence.
- After selecting the first observation per ticker/entry-day before examining
  labels, and excluding observations through September 22 (conservative eight
  trading-session buffer after training), there are 104 signal-days but **zero
  eligible labeled rows with a stored PreBreakout probability**.
- Missing values are not replaced, and present-day inference is not substituted
  for historical predictions.

Previous comparable production-model AUC: unavailable. Current: unavailable.
Delta: unavailable. The historical training AUC and prior tiny walk-forward
sample are not comparable. Do not label this model overconfident or
underconfident from these data. **CALIBRATION: UNVERIFIABLE_FOR_MODEL_TARGET**;
probability buckets have no eligible observations. HSF Score is a heuristic,
not a probability, so Brier/calibration are not manufactured from HSF/100.

### Descriptive HSF ranking, not ML validation

Same selection policy before/after: first ticker/entry-day observation, chosen
before label filtering. Labeled signal-days increase 130 to 319, across 13 entry
days. The previously-valid-only cohort has 134 signal-days, but four are later
observations excluded by the global first-observation policy.

| Horizon | Before HSF AUC | After | Delta | After 95% day-bootstrap interval |
|---|---:|---:|---:|---|
| 1 day | .4457 | .6160 | +.1703 | .3817-.6474 |
| 3 days | .4748 | .6237 | +.1489 | .3989-.6498 |
| 5 days | .4403 | .6097 | +.1694 | .3702-.6694 |

These are descriptive ranking associations, not a model improvement. All
intervals include .5. Three hundred entry-day bootstrap resamples account for
within-day dependence but not all cross-day dependence from overlapping
horizons; intervals are approximate, not a significance claim.

## Segments and recovered cohort

Best supported by sample count: CAUTION (237 labeled signal-days, 13 days).
Its five-day HSF AUC is .6577, with a very wide .2664-.6972 interval.
WATCH has 56 signal-days across 13 days and one-day AUC .3730; this is a weak
descriptive segment, not proof of an inverse trading strategy. STRONG has only
26 across nine days and is under-sampled. Momentum (32) and generic Signal
(113) each occur on one entry day; their apparent performance is not robust.
Entry-day segments cannot establish temporal generalization. No best/worst
production-model segment can be identified without usable predictions/targets.

| Cohort | Rows | Signal-days | STRONG / WATCH / CAUTION | Entry days |
|---|---:|---:|---|---:|
| Recovered | 613 | 189 | 13 / 32 / 568 | 2 |
| Previously valid | 337 | 134 | 86 / 103 / 148 | 13 |

| Horizon | Recovered positive rate | Previously valid | Recovered benchmark beat | Previous benchmark beat |
|---|---:|---:|---:|---:|
| 1 day | 36.22% | 59.35% | 51.55% | 56.88% |
| 3 days | 36.87% | 53.41% | 35.07% | 51.67% |
| 5 days | 46.17% | 56.38% | 33.28% | 52.04% |

Benchmark support: 613 repaired vs 269 previously-valid rows. Recovered median
excess returns: +0.135%, -0.915%, -1.937% for 1/3/5 days. This is not a matched
date/tier experiment. Tier/date composition differs substantially.
Stored PreBreakout output exists for 586 recovered rows (mean 13.04 on the
stored percent scale, quartiles all 13.1), versus only five previously-valid
rows (mean 20.08). This is a coverage/concentration warning, not evidence of
calibration. Cohort-level model performance by tier/recommendation is blocked
by the same provenance/target limitations. Recommendations are all unrecorded.

**RECOVERED 613 IMPACT: MIXED.** Positive: coverage and missing-outcome bias
improve. Negative: observed success rates fall. Neutral for production model
quality: no comparable model-target performance is established. Concentration
and tier mix explain why global descriptive statistics move; 608 observations
from one date must not be treated as 608 independent market periods.

## AUC diagnosis and next move

- **P0:** Missing served-model/target provenance prevents a trustworthy ML AUC.
  This is an evaluation blocker, not proof of poor model discrimination.
- **P1:** Cohort selection and temporal concentration materially affect the
  descriptive AUC; only 13 labeled dates and broad intervals. Probability
  missingness and near-constant recovered outputs further limit diagnosis.
- **P2:** Class balance changes and tier mixing matter; the five-day population
  is now nearly balanced. Feature drift, redundancy, label noise and leakage
  cannot be ruled in/out without the original inputs and target path. They are
  untested hypotheses, not established explanations. Horizons are kept separate.

Exactly one next run: **HSF ML v3 - Frozen Prediction and Target Provenance Audit**.
Verify which predictions, artifact versions, point-in-time inputs and original
target paths can be joined reliably before selecting an evaluation cohort.
This has higher value than tuning or calibration repair because neither can
be judged against the current incomplete evidence. Do not implement it here.

## Verification and preservation

New files: `analytics/post_rescore_audit.py`, `scripts/post_rescore_audit.py`,
`tests/test_post_rescore_audit.py`, and this Markdown/JSON report pair.
Relevant regression suite: **101 passed, 0 failed, 10 subtests passed**, nine
existing deprecation warnings (Starlette/httpx and NumPy timedelta).
Ruff: all three added Python files pass. Production audit and canonical dry-run
workflows passed. Tests cover cohort boundaries, no input mutation, label-safe
deduplication, maturity/finite checks, absent probabilities, and one-class AUC.
No scoring, model weights, training, thresholds, recommendation logic, scanner,
research history, or existing report files were modified. No main promotion.
