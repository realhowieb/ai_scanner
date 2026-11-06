# ML preparation and fixed experiment plan — November 6, 2026

This preparation does not authorize a production switch. No production dataset
was finalized, no production-data model was fitted, and the real final test was
not scored. See the [earlier readiness audit](ML_TRAINING_READINESS_AUDIT_2026-11-06.md)
for the conditional November 12–13 capture forecast.

## Dataset verification diagnosis

The read-only [diagnostic run](https://github.com/realhowieb/ai_scanner/actions/runs/42285627169)
succeeded at 20:08 UTC. Both rebuilds retained all 1,041 original members, with
zero missing or unexpected IDs. The stored fingerprint remains unchanged.

| Evidence | Frozen metadata | Current rebuild |
|---|---:|---:|
| Matured five-day observations | 337 | 1,025 |
| Pending | 75 | 0 |
| Unavailable | 629 | 16 |
| Benchmark gaps among matured observations | 128 | 0 |
| Matched scans | 233 | 233 |
| No scan record | 95 | 59 |
| Only later scans | 703 | 739 |

The current rebuild fingerprint is
`sha256:6e9ef91681428665f1e2e50eb195a962d4f2a7ef11d27981cc3701a4c3d80a13`.
Masking outcomes to the original freeze instant still fails verification
(`sha256:6b02618c95662b471b4a09fd2f431cf08b4fa8e34e02a7e65470d0f2bac22245`).
Thus a simple current-maturity rebuild is insufficient to reconstruct the old
version. Its original code revision was
`2990c99ef1a72a1c5ce8db53a96f0e9dfd0b3cc1`; original feature/label matrices and
component hashes were not retained. Aggregate changes are proven, but they do
not uniquely attribute every changed cell between code, historical data and
availability. Do not relabel the rebuilt data as the old version or replace its
stored hash. New dataset builds now emit component hashes, preserving the
existing whole-dataset hash contract.

The diagnostic enforces read-only transactions. That guard exposed an existing
research reader's schema initializer; scan reads now issue no schema writes.
The successful run retained the guard.

## Fresh, unfinalized preview

Cohort: September 14 through November 6, excluding the bootstrap burst.
These are counts at 20:08 UTC, not a frozen dataset or performance results.

- 1,834 observations; 798 eligible certified, matured signal-days.
- Actual primary target: 218 positive / 580 negative (27.32% minority share).
- Training split: 287 rows, including 70 positives; four usable pre-holdout folds.
- Proposed reserved block: 214 rows starting October 23; split marked valid.
  No final predictions or scores were produced. The gap between eligible, training
  and reserved counts reflects the split's purging and eligibility rules.
- Remaining live gate: `ai_confidence_frozen` 80%, requires at least 95%.

Rerun these checks on the actual freeze day. The earlier audit's statement that
actual-target counts were unverified is superseded by this preview; finalization
and reproducibility verification are still outstanding.

## Pipeline rehearsal and production capture

The offline synthetic rehearsal fitted XGBoost, LightGBM and CatBoost, verified
saved artifacts, exercised development evaluation and synthetic final-report
reuse, and generated 30 shadow predictions. Production writes: zero. Its SQLite
final-test ledger was separate from production. Synthetic performance is not
evidence of investment value.

A bounded read-only production aggregate at 20:00 UTC found 350/350 recent
predictions for each model. PreBreakout had 350 complete inference identities;
AI Confidence had 50, leaving 300 predictions without complete identity evidence.
Those 300 are not missing predictions. The new admin capture card distinguishes
prediction gaps, identity gaps and lag behind the latest saved opportunity.
It does not claim model quality or detect a stopped scanner when no observations
arrive; scanner cadence has its own health checks. Weekends do not create a
wall-clock stale warning.

## Fixed comparison and decision rules

Before any real training, require all live readiness and actual-target checks.
Build the new cohort twice with unchanged inputs/code; require equal membership
and fingerprints. Review and finalize a new version, then pin its version, hash,
freeze instant and code revision. Preserve the old failed version.

The six recipes are the cross-product of XGBoost, LightGBM and CatBoost with
`production` and `leakage_safe` feature sets, using the fixed provider defaults
recorded in `docs/ml_experiment_plan_2026-11-06.json`. No hyperparameter search or
additional ensemble recipes in this comparison.

Primary target: `F_clean_path_5d`, MFE >= +4% and MAE > -2% over five trading days.
This is an order-free clean-path target, not the sequence of first threshold hits.
Use certified matured signal-days, identical folds, at least 60 training rows,
30 validation rows and five validation examples per class, purging and a one-day
embargo. Require minority share >=20%, both training classes, at least three
pre-holdout folds and a reserved block >=100 rows across >=5 entry days.

First run development evaluation only (`mode=evaluate`, `register=false`,
`final_holdout=false`). Report raw/calibrated ROC-AUC, PR-AUC, Brier, calibration,
top-ranked precision/lift, fold stability and returns after fixed 20 bps
round-trip costs. Compare majority-class and AI Confidence-as-served baselines;
served comparisons must use the same rows and never backfill with today's model.

Selection rule: rank the six recipes by pooled out-of-fold **raw ROC-AUC**,
descending. Break exact reported ties by lower raw Brier, then provider order
XGBoost, LightGBM, CatBoost, then `leakage_safe` before `production`. Reject a
recipe with missing/invalid fold results. This ranks development evidence; it
does not automatically qualify a winner. Review uncertainty, fold stability,
calibration, baseline comparisons and costs. If none warrants a final test,
stop and retain production. Do not change the selection rule after seeing scores.

Lock the selected recipe before a separately approved real final evaluation.
Use the durable claim/report-reuse ledger. Do not use final scores to select a
different recipe or repeatedly probe the block. Candidate registration, forward
shadow evidence and a production replacement remain separate reviewed stages.
Passing readiness does not train or replace AI Confidence or PreBreakout.
