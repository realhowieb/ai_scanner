# Run 58 — Cohort maturation parity & missingness audit

**Root cause: MARKET_DATA_AVAILABILITY_EFFECT (confidence HIGH), driven by
cohort composition.** Historical scheduler starvation was real, but it affected
every cohort and *narrowed* the CANDIDATE/CONTROL gap rather than causing it. No
pipeline bug was found, and **no code fix was applied (AUDIT_ONLY)**. The
current 2,000-symbol scheduler reaches every cohort equally.

This is research-integrity work only. Scoring, ranking, cohort selection,
universe, outcome formulas, maturation behavior and the Run 56 epoch are
unchanged. No outcome values appear anywhere; the audit passes a token-level
anti-peeking guard, and its output is invariant to return values (tested).

## Method

- **Run 55 snapshot:** `artifacts/research/maturation_parity_run55_snapshot.json`.
  It comes from the exact Run 55 input (workflow 36226568240), evaluated as of
  2026-09-26 07:23 UTC.
- **Current state plus scheduler trace:** read-only workflow
  `maturation-parity-audit.yml`,
  [run 36229121325](https://github.com/realhowieb/ai_scanner/actions/runs/36229121325).
  It runs a **production-faithful DRY RUN** of the maturation worker: same loader
  (limit 5,000, outcomes attached), cap 2,000, batched Alpaca with retries, and
  retirement. A new `trace` hook records a status for every (observation,
  horizon). Nothing is written.
- **Taxonomy:** every eligible-but-unmatured horizon gets one reason. With the
  trace these are **PENDING_BACKLOG** (bars exist; a real run writes it now),
  SCHEDULER_DEFERRED, RATE_LIMITED, PRICE_DATA_UNAVAILABLE (provider returned no
  bars), INSUFFICIENT_FUTURE_BARS, RETIRED_WINDOW_CLOSED, INELIGIBLE_SYMBOL,
  INVALID_ANCHOR, PIPELINE_ERROR, or UNKNOWN. Without a trace, only structural
  reasons are assigned and everything else stays UNKNOWN.
- **Historical replay:** the worker's own ordering is replayed under a 400-symbol
  cap at the **10 real scheduled maturation run times** (2026-09-22 19:47 →
  09-25 23:41 UTC; GitHub throttled the `*/30` schedule).

## 1. Run 55 coverage is real and reproduced exactly

Run 55 primary population (regular-session anchors), identical to the Run 55
report:

| Horizon | CANDIDATE | NEAR_MISS | CONTROL | Gap | Class |
|---|---|---|---|---|---|
| +5m | 54.75% | 45.68% | 14.62% | 40.13 pp | CRITICAL |
| +15m | 54.08% | 44.21% | 11.85% | 42.23 pp | CRITICAL |
| +30m | 49.83% | 40.78% | 9.92% | 39.91 pp | CRITICAL |
| +60m | **47.58%** (571/1,200) | 37.85% (232/613) | **8.62%** (112/1,300) | 38.96 pp | CRITICAL |

**Current historical state (all sessions).** No real maturation has run since
the Run 55 snapshot, so this equals the snapshot:

| Horizon | CANDIDATE | NEAR_MISS | CONTROL | Gap | Projected after backlog drains | Projected gap |
|---|---|---|---|---|---|---|
| +5m | 61.21% | 51.19% | 18.81% | 42.4 pp | 99.9% / 100.0% / 50.9% | 49.1 pp CRITICAL |
| +15m | 60.64% | 49.93% | 15.12% | 45.5 pp | 98.8% / 98.3% / 41.3% | 57.5 pp CRITICAL |
| +30m | 57.00% | 46.98% | 12.81% | 44.2 pp | 91.7% / 91.0% / 34.0% | 57.7 pp CRITICAL |
| +60m | 54.86% | 44.18% | 11.12% | 43.7 pp | 87.6% / 86.4% / 28.1% | 59.5 pp CRITICAL |

**Forward epoch:** `FORWARD_PARITY_STATUS = NO_FORWARD_DATA`. No scheduled scan
has run since the epoch began at 2026-09-26 07:23 UTC.

## 2. Missingness by cohort (+60m, traced)

| Reason | CANDIDATE (1,400) | NEAR_MISS (713) | CONTROL (1,600) |
|---|---|---|---|
| PENDING_BACKLOG | 458 | 301 | 272 |
| INSUFFICIENT_FUTURE_BARS | 174 | 97 | **730** |
| PRICE_DATA_UNAVAILABLE | 0 | 0 | **403** |
| INELIGIBLE_SYMBOL (preferred share) | 0 | 0 | 17 |
| SCHEDULER_DEFERRED / RATE_LIMITED / RETIRED / PIPELINE_ERROR / UNKNOWN | 0 | 0 | 0 |

Candidate and near-miss missingness is almost entirely **backlog**, meaning the
data exists. Control missingness is mostly **no data at all** or **too few
bars**.

## 3. Timing does not explain it

All cohorts are captured in the same scan runs at the same timestamps: P25 09:36,
median 14:34, P75 15:36 ET for every cohort.

The +60m horizon crosses the 16:00 close for 21.4% of candidates, 22.9% of
near-misses and 25.0% of controls, a 3.6 pp spread. There are no 10:30–12:00
scans, because the slots are 09:35 / 12:35 / 15:35. The 15:36 slot has a
structurally close-crossing +60m for **every** cohort; there, controls matured
12/300 against candidates' 100/300. Timing hits all cohorts alike, and controls
fare worse within every slot.

## 4. Symbol composition explains it

| | CANDIDATE | NEAR_MISS | CONTROL |
|---|---|---|---|
| Unique symbols | 261 | 277 | 1,496 |
| Observations per symbol | 5.36 | 2.57 | 1.07 |
| Capture $-volume p10 / p50 / p90 ($M) | 0.51 / **12.78** / 68.5 | 0.40 / 11.31 / 58.8 | 0.00 / **0.014** / 2.58 |
| Median capture price | $103 | $99 | $25 |
| Symbols never matured | 28.0% | 38.3% | **81.1%** |
| Preferred / malformed | 0 | 0 | 17 preferred (pre-exclusion captures) |

Controls are a deterministic sample of the *whole evaluated universe*.
Candidates pass the scan's $5M 20-day liquidity floor. At capture, control
median dollar volume is **0.11%** of candidates'. On the IEX minute-bar feed,
such names often print no bars, or fewer than the 60 bars that "+60m" requires
within the forward window.

- **Exchange / ETF status:** not persisted on observations.
- **Duplicate symbols within a scan:** 0.
- **Cross-cohort symbol overlap:** 24 candidate∩control, 35 near-miss∩control,
  and 170 candidate∩near-miss (across different runs).

## 5. Scheduler fairness

**Static audit** (`scripts/mature_observations.py` and its loader):
- **Grouping and batching:** grouping is by symbol and batching is by oldest
  pending anchor. Neither looks at cohort.
- **Retirement:** time-based.
- **Bars:** one bar series per symbol serves every cohort.
- **When the cap binds:** the ordering is cohort-neutral only while the cap does
  not bind. Once it binds, persistently failing old symbols hold the head of the
  queue, and a recurring symbol's newer observations ride along with its oldest
  pending one. A test demonstrates this.

**Historical 400-cap starvation: YES, it occurred.** Replaying the 10 real runs,
the share of still-unmatured +60m work that was **never attempted** was:
- CANDIDATE: 99.4%
- NEAR_MISS: 97.7%
- CONTROL: 68.9%

At the snapshot state, a 400 cap would have reached 28.8% of control work and
0.04% of candidate work. Old control symbols sit at the head of the queue.
Starvation therefore **delayed candidates more than controls**. Its contribution
to the CONTROL−CANDIDATE gap is **−15.7 pp**: it narrowed the gap.

**Current 2,000 cap: no starvation.** The dry run had 1,593 ready symbols and 0
deferred. Every cohort's ready work was reached (reach gap 0 pp), with 27 Alpaca
requests and 0 × 429.

## 6. Symbol-level sharing

Across 4,478 (symbol, horizon) cases where one cohort matured and another did
not, **all are DIFFERENT_SCAN_RUN**: different anchors, and so different future
bars. There are **0 same-anchor mismatches**, so bars fetched for one cohort
never failed to mature another cohort's identical observation. A test also shows
one bar series maturing both cohorts at the same anchor.

## 7. Retirement

Retirement is not yet active: the oldest observations are under 6 days old, so 0
are retired in every cohort. The code path is time-based and cohort-agnostic, and
a test shows equal ages retire at equal rates. It will inherit the availability
gap, because no-data controls stay unmatured until they retire, but it does not
create one.

## 8. Root-cause verdict

**Pre-declared rule:** each mechanism's contribution is the control missing rate
minus the candidate missing rate for its reasons, and ≥ 5 pp is material.

| Mechanism | Contribution to the +60m gap |
|---|---|
| MARKET_DATA_AVAILABILITY_EFFECT | **+58.4 pp** |
| COHORT_COMPOSITION (ineligible symbols) | +1.1 pp |
| EXPECTED_TEMPORAL (close-crossing spread) | 3.6 pp |
| HISTORICAL_SCHEDULER_STARVATION | **−15.7 pp** |
| RETIREMENT / PIPELINE_BUG / UNATTRIBUTED | 0 |

**Classification: MARKET_DATA_AVAILABILITY_EFFECT. Confidence: HIGH.** The basis
is a real-data trace, with ≥ 300 eligible observations per cohort and 0 pp
unattributed. The availability gap is **driven by composition**: control
liquidity is 0.11% of candidate liquidity.

**Does this bias comparisons?** Yes, if it is ignored. Matured controls are the
liquid minority of the control sample, not a random draw from the universe. A
naive CANDIDATE-vs-CONTROL comparison would compare liquid breakout names with
a *liquidity-selected* subset of controls.

## 9. Fix policy: AUDIT ONLY

The Part 10 conditions are not met:
- There is no reproducible implementation defect in the current pipeline: the
  2,000 cap does not bind, and there are no same-anchor mismatches.
- The remaining gap comes from how controls are selected and what the IEX feed
  covers. It is **not** an implementation error, and "correct" behavior is a
  research-design choice, not an unambiguous fix.

Changing control selection, the price feed, or the +60m bar-count definition is
out of scope for Run 58.

**Instrumentation added** (no behavior change):
- `scripts/mature_observations.py` has an optional status-only `trace` hook,
  default off. The report, writes and outcome values are identical with and
  without it (tested).
- Run 56 parity entries now carry HEALTHY / ACCEPTABLE / WARNING / CRITICAL
  classes.
- Run 56 surfaces `maturation_capacity_binding` from the latest scheduled
  maturation report, so a future return of cap-driven starvation is visible.
- Run 56 gates and the epoch are unchanged.

## Implications for Run 56

With the current design, forward parity will almost certainly measure about
60 pp at +60m and trip Gate E (≤ 10 pp) once parity is measurable. That makes
Run 56 **DATA_QUALITY_BLOCKED / DATA_PIPELINE_BIAS**. This is the gate working
as intended: more time will not fix a composition-driven gap.

## Recommended Run 59 scope (research design, pre-registered, forward-only)

Decide **before** further forward data is analyzed, and do not use effectiveness
results to decide:

1. **Liquidity-comparable controls.** Draw controls from symbols that pass the
   same liquidity floor as candidates (or stratify by capture dollar volume).
   This changes control selection, so it needs a new pre-registered forward epoch.
2. **Minute-bar feed for maturation.** SIP instead of IEX would give sparse names
   bars. It needs a cost/entitlement check and must apply to every cohort.
3. **Horizon definition.** "+60m" counts bars, not minutes, so sparse names need
   more wall-clock time. A wall-clock horizon is an outcome-formula change and
   needs its own run.
4. **Alternatively, keep the design** and pre-register a liquidity-matched
   CANDIDATE-vs-CONTROL comparison in the Run 55 rerun. Then Gate E has to be
   read with that matching.

**Operational note for Monday's first real maturation:** about 4,900 outcomes
will be saved, each over its own DB connection. Watch the 20-minute workflow
limit. Saves are idempotent, so a timeout only defers the remainder to the next
run.
