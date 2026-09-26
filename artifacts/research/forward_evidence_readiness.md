# Forward Evidence Readiness

Generated 2026-09-26T07:43:25.036718+00:00 · schema `hsf-forward-readiness-1.0` · READ-ONLY · no effectiveness statistics

## State: **COLLECTING**

- Reason: no regular-session forward observations yet
- Limiting factor: **NO_FORWARD_DATA**
- LONG readiness: **COLLECTING** · SHORT readiness: **INSUFFICIENT**
- Estimated trading days until ready: **UNKNOWN** (non-sample gates failing (D_horizon_maturation, E_maturation_parity, F_directional_integrity); their rate of improvement is not a collection rate. Sample-size gates alone: UNKNOWN (only 0 completed forward trading day(s); need >= 2 to measure collection rates))
- RUN55_RERUN_RECOMMENDED = **false**

## Forward epoch

- Start: `2026-09-26T07:23:11+00:00` · first forward scan run: `None`
- Run 55 evaluation commit `284e2ac8ec` · criteria commit `c5d34a751d` · workflow run 36226568240
- Scanner scoring version: None
- Schemas: observation `hsf-obs-1.0`, outcome `hsf-outcome-1.0`

## Gates (pre-registered)

| Gate | Status | Detail |
|---|---|---|
| A_trading_days | ❌ FAIL | 0 completed forward trading days (min 10, preferred 20) |
| B_scan_runs | ❌ FAIL | 0 regular-session forward scan runs (min 50, preferred 100) |
| C_cohort_clusters | ❌ FAIL | min matured scan-run clusters = 0 (CANDIDATE at +15m); need >= 30 for every cohort at every horizon |
| D_horizon_maturation | ❌ FAIL | +5m: no eligible observations yet (need 80%) → FAIL; +15m: no eligible observations yet (need 80%) → FAIL; +30m: no eligible observations yet (need 80%) → FAIL; +60m: no eligible observations yet (need 70%) → FAIL |
| E_maturation_parity | ❌ FAIL | +5m: not measurable yet (a cohort has no eligible observations) → FAIL; +15m: not measurable yet (a cohort has no eligible observations) → FAIL; +30m: not measurable yet (a cohort has no eligible observations) → FAIL; +60m: not measurable yet (a cohort has no eligible observations) → FAIL |
| F_directional_integrity | ❌ FAIL | 0 matured forward outcomes (need >= 100 to judge); coverage {'directional_return': None, 'mfe': None, 'mae': None} |
| G_effective_clusters | ❌ FAIL | min paired scan-run clusters = 0 (min 20 = Run 55 STRONG, preferred 30); observations within one run are not independent |
| H_research_integrity | ✅ PASS | no integrity issues found |

## Run 55 baseline vs forward epoch

| Metric | Run 55 | Forward epoch | Gate |
|---|---|---|---|
| Trading days | 3 | 0 | >= 10 (pref 20) |
| Scan runs (regular session) | 13 | 0 | >= 50 (pref 100) |
| +60m CANDIDATE maturation % | 47.6 | — | >= 70% |
| +60m CONTROL maturation % | 8.6 | — | >= 70% |
| +60m parity gap (pp) | 39.0 | — | <= 10 (pref 5) |
| +60m CANDIDATE∩CONTROL clusters | 9 | 0 | >= 20 (pref 30) |
| LONG matured clusters (+60m) | — | 0 | >= 30 |
| SHORT matured clusters (+60m) | 0 | 0 | separate |
| Directional/MFE/MAE coverage % | 54.7 | — | >= 90% |

## Coverage

- Trading days: 0 (0 completed) · calendar days 0 · first — · latest —
- Scan runs: 0 total, 0 regular-session, 0 successful (regular-session run containing all three cohorts) · unique symbols 0

| Cohort | Observations | Symbols | Scan runs | Matured | Unmatured |
|---|---|---|---|---|---|
| CANDIDATE | 0 | 0 | 0 | 0 | 0 |
| NEAR_MISS | 0 | 0 | 0 | 0 | 0 |
| CONTROL | 0 | 0 | 0 | 0 | 0 |

## Horizon maturation (regular session, time-eligible observations)

| Horizon | Cohort | Eligible | Matured | Maturation % | Directional % | MFE % | MAE % | Clusters | NOT_YET_ELIGIBLE | PRICE_DATA_UNAVAILABLE | INSUFFICIENT_FUTURE_BARS | RATE_LIMITED | RETIRED | FILTERED_BY_POLICY | OTHER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| +5m | CANDIDATE | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| +5m | NEAR_MISS | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| +5m | CONTROL | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| +15m | CANDIDATE | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| +15m | NEAR_MISS | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| +15m | CONTROL | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| +30m | CANDIDATE | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| +30m | NEAR_MISS | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| +30m | CONTROL | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| +60m | CANDIDATE | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| +60m | NEAR_MISS | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| +60m | CONTROL | 0 | 0 | — | — | — | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

_per-observation failure reasons are not persisted; PRICE_DATA_UNAVAILABLE and INSUFFICIENT_FUTURE_BARS are inferred, RATE_LIMITED is only visible run-level in maturation_run_report._

## Maturation parity

| Horizon | CANDIDATE % | NEAR_MISS % | CONTROL % | Parity gap (pp) | Measurable |
|---|---|---|---|---|---|
| +5m | — | — | — | — | no |
| +15m | — | — | — | — | no |
| +30m | — | — | — | — | no |
| +60m | — | — | — | — | no |

## Directions

| Direction | Observations | Matured (+60m) | Scan runs | Matured clusters (+60m) |
|---|---|---|---|---|
| LONG | 0 | 0 | 0 | 0 |
| SHORT | 0 | 0 | 0 | 0 |
| CONTROL_UNDIRECTED | 0 | 0 | 0 | 0 |

## Data quality

- pre_epoch_observations_excluded: 3713
- legacy_untagged_excluded: 520
- duplicate_observation_ids: 0
- conflicting_observation_ids: 0
- duplicate_outcomes: 0
- conflicting_outcomes: 0
- orphan_forward_outcomes: 0
- point_in_time_violations: 0
- direction_transform_mismatches: 0
- invalid_market_values: 0
- cohort_overlap: {}
- scoring_version_drift: False
- latest maturation run: {'generated_at': '2026-09-25T23:42:47.808993+00:00', 'schema': 'hsf-maturation-1.1', 'symbols_deferred': 1213, 'dry_run': False, 'retired': None, 'failures': {'INSUFFICIENT_FUTURE_BARS': 657, 'PRICE_DATA_UNAVAILABLE': 911}}

## Progress

- trading_days_progress: 0.0
- scan_runs_progress: 0.0
- candidate_cluster_progress: 0.0
- near_miss_cluster_progress: 0.0
- control_cluster_progress: 0.0
- horizon_coverage_progress: 0.0
- maturation_parity_progress: 0.0
- bottleneck: trading_days_progress
- sample_size_progress: 0.0

_Evidence quantity/quality only. No win rate, return, cohort difference, correlation, bucket or threshold statistics are computed or reported._
