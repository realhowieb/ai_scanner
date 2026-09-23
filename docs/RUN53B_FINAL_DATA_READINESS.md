# Run 53B - Final Run-54 Data Readiness

## 1. Executive Summary

**Final verdict: DATA NOT READY**

**RUN54_READY = NO**

The production cohort structure is trustworthy: 2,233 observations, no duplicate
IDs, no conflicting duplicates, no Candidate/Control overlap, no invalid values,
and no point-in-time violations. Modern cohort tagging is present and the 520
legacy Candidate rows are deterministically separable.

The headline `missing_fields = 100%` was mostly an auditor/schema-expectation
problem. `build_observation()` records every absent field in a broad 19-field
canonical envelope, while the scheduled scanner intentionally captures only the
fields it emits. The old auditor reduced that list to a boolean and therefore
treated one absent optional indicator the same as an absent required identity or
outcome field.

The field-level audit nevertheless found two real readiness blockers:

1. All 800 modern Control observations lack a direction.
2. All 2,521 modern matured outcome records lack `directional_return`, `mfe`, and
   `mae`. They were created before the Run-53A normalization fix and cannot satisfy
   the Run-54 effectiveness contract.

There is consequently no complete modern Candidate-vs-Control sample at any
horizon. Run 52 is deployed, but no production maturation artifact from a commit
containing Run 52 exists yet, so backlog drain is also unproven.

## 2. Production Dataset Snapshot

Source: read-only Neon audit workflow
[35926807979](https://github.com/realhowieb/ai_scanner/actions/runs/35926807979),
commit `b4faff8`, generated 2026-09-23 UTC.

| Cohort | All observations | Modern | Legacy | Modern matured | Modern unmatured | Symbols | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Candidate | 1,120 | 600 | 520 | 449 | 151 | 199 | 13 |
| Near Miss | 313 | 313 | 0 | 174 | 139 | 177 | 7 |
| Control | 800 | 800 | 0 | 112 | 688 | 773 | 8 |
| **Total** | **2,233** | **1,713** | **520** | **735** | **978** | - | - |

Integrity: duplicate IDs `0`; conflicting duplicates `0`; Candidate/Control
within-run overlaps `0`; PIT violations `0`; invalid observations `0`.

Three Near-Miss/Control overlaps exist within a run. They do not contaminate the
primary Candidate-vs-Control comparison, but Run 54 should avoid treating those
two cohorts as mutually exclusive without deduplication.

## 3. Missing-Field Root Cause

The old audit incremented `missing_field_observations` whenever
`data_quality.missing_fields` was non-empty. That list is generated against all
canonical `MARKET_FIELDS + INDICATOR_FIELDS`, not against an analysis contract.

Scheduled scanner capture currently maps:

- Market: `price`, `volume`
- Indicators: `gap_pct`, `chg_pct`, `rvol`, `atr_pct`

It does not promise OHLC, EMA, RSI, ADX, VWAP, SuperTrend, EWO, model probability,
or rank for every cohort. Therefore every row can legitimately have a non-empty
canonical missing list. The auditor now preserves the compatibility count but
also reports each field and classifies it as `OPTIONAL`, `LEGACY_SCHEMA`,
`EXPECTED_NULL`, or `ACTUAL_DATA_DEFECT`.

## 4. Field-Level Missingness

### Point-in-time observation fields

| Field | Candidate | Near Miss | Control | Classification | Run-54 impact |
|---|---:|---:|---:|---|---|
| observation_id | 0 | 0 | 0 | Required | None |
| symbol | 0 | 0 | 0 | Required | None |
| timestamp | 0 | 0 | 0 | Required | None |
| market_context.scan_id | 0 | 0 | 0 | Required | None |
| research_cohort | 520 | 0 | 0 | LEGACY_SCHEMA | Exclude legacy rows |
| direction | 0 | 0 | 800 | ACTUAL_DATA_DEFECT | Blocks directional Control comparison |
| market.price | 0 | 0 | 0 | Required | None |
| scanner_score | 0 | 0 | 800 | OPTIONAL | Exclude score-stratified Control analysis |
| rank | 1,120 | 313 | 800 | OPTIONAL | Exclude rank analysis |
| PreBreakout probability | 1,120 | 313 | 800 | OPTIONAL | Exclude PreBreakout analysis |
| AI Confidence | 1,120 | 313 | 800 | OPTIONAL | Exclude AI-confidence analysis |

### Canonical optional fields

| Field family | Candidate missing | Near Miss missing | Control missing | Classification |
|---|---:|---:|---:|---|
| open/high/low/previous_close | 1,120 each | 313 each | 800 each | OPTIONAL |
| ema9/ema21/rsi/adx/vwap/vs_vwap_pct | 1,120 each | 313 each | 800 each | OPTIONAL |
| supertrend_direction/ewo | 1,120 each | 313 each | 800 each | OPTIONAL |
| gap_pct/chg_pct | 0 | 0 | 800 each | OPTIONAL for cohort effectiveness |
| rvol/atr_pct | 0 | 1 each | 800 each | OPTIONAL for cohort effectiveness |

### Modern matured outcome fields

| Field | Candidate missing / 1,581 | Near Miss missing / 611 | Control missing / 329 | Classification |
|---|---:|---:|---:|---|
| horizon | 0 | 0 | 0 | Required after maturity |
| evaluation_time | 0 | 0 | 0 | Required after maturity |
| raw_return | 0 | 0 | 0 | Required after maturity |
| directional_return | 1,581 (100%) | 611 (100%) | 329 (100%) | ACTUAL_DATA_DEFECT for Run 54 |
| mfe | 1,581 (100%) | 611 (100%) | 329 (100%) | ACTUAL_DATA_DEFECT for Run 54 |
| mae | 1,581 (100%) | 611 (100%) | 329 (100%) | ACTUAL_DATA_DEFECT for Run 54 |

Unmatured outcome fields are `EXPECTED_NULL`, not defects. Optional point-in-time
features are never zero-filled; analyses that require them must exclude the row.

## 5. Run-54 Required Field Contract

### Identity

`observation_id`, `symbol`, `timestamp`, `market_context.scan_id`, and an explicit
`research_cohort` tag.

### Point-in-time

`direction` and `market.price` are required for directional effectiveness.
Scanner score, rank, signal fields, ML probability, and PreBreakout probability
are required only for analyses that explicitly use them.

### Outcome, per matured horizon

`horizon`, `evaluation_time`, `raw_return`, `directional_return`, `mfe`, and `mae`.
Entry price is the point-in-time `market.price`. Future price is deterministically
derived as `entry_price * (1 + raw_return)` because it is not persisted directly.

An analysis must drop rows missing one of its required fields. It must never
coerce absent optional fields or outcomes to zero. No outcome is required before
its horizon matures.

## 6. Legacy vs Modern Dataset

| Population | N |
|---|---:|
| ALL_CANDIDATE | 1,120 |
| LEGACY_CANDIDATE | 520 |
| MODERN_EXPLICIT_CANDIDATE | 600 |
| MODERN_NEAR_MISS | 313 |
| MODERN_CONTROL | 800 |

Run 54 can isolate modern data using the explicit top-level or market-context
`research_cohort` tag. It must not rely on `cohort_of()` alone because that helper
intentionally infers untagged legacy rows as Candidate.

## 7. Post-Run-52 Maturation Evidence

No available production maturation artifact was generated from a commit that
contains `3e5320e`.

| Run | Commit | Schema | Dry run | Ready symbols | Processed | Deferred | New outcomes | Already reported | Post-fix? |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| 35911847604 | `b31c553` | 1.0 | Yes | 738 | 400 cap | 338 | 3,981 would-write | 0 | No |
| 35918775967 | `a357829` | 1.0 | No | 1,042 | 400 cap | 642 | 2,336 | 2,196 | No |

Schema 1.0 lacks `ready_observations`, `failed_symbols`,
`oldest_pending_age_min`, and `estimated_clearance_runs`. Neither run is a
descendant of the Run-52 commit. The second run demonstrates the old redundant
work problem, not the fix.

## 8. Backlog Trend

**Classification: INSUFFICIENT_POST_FIX_EVIDENCE**

The pre-fix snapshots show ready symbols rising from 738 to 1,042 and deferred
symbols rising from 338 to 642. They cannot answer whether the outcome-aware
loader subsequently collapsed redundant fetching. There is no evidence yet that
oldest pending age is falling or that maturation throughput exceeds new capture.

One successful non-dry maturation run from a commit containing Run 52 is the
minimum next check. Two or more comparable runs are needed to classify the trend
as draining, stable, or growing.

## 9. Control Cohort Maturity

These counts are modern observations with a raw matured outcome. `Run-54 eligible`
is zero because direction, directional return, MFE, and MAE are incomplete.

| Horizon | Candidate LONG raw N | Control UNKNOWN raw N | Raw sample evidence | Run-54 eligible N |
|---|---:|---:|---|---:|
| +5m | 449 | 112 | MODERATE | 0 |
| +15m | 440 | 90 | PRELIMINARY (Control) | 0 |
| +30m | 361 | 72 | PRELIMINARY (Control) | 0 |
| +60m | 331 | 55 | PRELIMINARY (Control) | 0 |

The raw-return counts are promising, but they do not satisfy the explicit
directional effectiveness contract and must not be presented as Run-54 results.

## 10. LONG/SHORT Live Verification

`LONG matured N` with complete directional fields: **0**.

`SHORT matured N` with complete directional fields: **0**.

Representative pre-fix raw outcomes:

| Direction | Symbol | Horizon | Entry | Derived future | Raw return | Directional | MFE | MAE |
|---|---|---|---:|---:|---:|---|---|---|
| LONG | SEI | +5m | 67.4250 | 67.3800 | -0.000667 | null | null | null |
| LONG | OLLI | +5m | 82.1050 | 82.1550 | +0.000609 | null | null | null |
| UNKNOWN Control | VWO | +5m | 60.1250 | 60.1100 | -0.000249 | null | null | null |

Deterministic LONG/SHORT formula tests pass after Run 53A, but these live rows
predate that fix and cannot verify the production directional fields.

`SHORT_LIVE_EVIDENCE = INSUFFICIENT`. SHORT conclusions must be excluded. LONG
analysis is also blocked until fresh complete outcomes exist.

## 11. Remaining Risks

1. Control direction has no production capture contract.
2. Existing outcomes are immutable first-write-wins, so pre-fix null directional
   metrics will not be repaired by simply rerunning maturation.
3. No post-Run-52 backlog artifact exists.
4. Near-Miss/Control overlap exists for three symbol/run pairs.
5. Rank and model probabilities are unavailable for the current research rows;
   analyses requiring them must be explicitly excluded.

## 12. Ten-Gate Run-54 Readiness Assessment

| Gate | Status | Evidence |
|---|---|---|
| 1. Modern cohort tagging trustworthy | PASS | 1,713 explicit modern rows |
| 2. Cohort separation valid | PASS | Candidate/Control overlap = 0 |
| 3. PIT integrity clean | PASS | PIT violations = 0 |
| 4. Required field contract satisfied | FAIL | Control direction missing on 800/800 |
| 5. Legacy deterministically excluded | PASS | 520 legacy Candidate rows identified |
| 6. Outcome calculations valid | PASS | Deterministic LONG/SHORT tests pass |
| 7. LONG live direction verified | CONDITIONAL | Formula verified; complete live examples = 0 |
| 8. SHORT live direction verified/restricted | CONDITIONAL | No live SHORT evidence; conclusions restricted |
| 9. Maturation produces usable modern outcomes | FAIL | Complete modern horizon rows = 0; no post-fix backlog run |
| 10. Adequate scoped effectiveness sample | FAIL | No complete Candidate/Control horizon-direction pair |

## 13. Final Recommendation

Do not begin Run 54 effectiveness analysis yet.

The next evidence-producing steps are narrowly scoped:

1. Define a point-in-time direction for Control capture without changing scoring,
   sampling, or ranking.
2. Let new observations mature under the Run-53A outcome implementation. Do not
   overwrite immutable pre-fix outcomes.
3. Run the existing maturation workflow from a commit containing Run 52 and
   collect schema-1.1 backlog telemetry for at least two runs.
4. Re-run the cohort audit. Run 54 can begin as soon as one modern
   Candidate-vs-Control horizon/direction pair has at least 30 complete rows in
   each cohort, with conclusions scoped to that evidence.

**DATA NOT READY**

**RUN54_READY = NO**
