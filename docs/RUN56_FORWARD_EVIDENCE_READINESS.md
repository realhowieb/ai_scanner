# Run 56 — Forward evidence readiness monitor

Run 55 returned **INSUFFICIENT_EVIDENCE**. The cause was too little history
(3 trading days) plus uneven outcome coverage, not proof of no edge. Run 56 adds
a read-only monitor that answers one question:

> **When do we have enough trustworthy forward production evidence to rerun Run 55?**

The scanner is **frozen** while forward evidence accumulates. Run 56 changes no
scoring, weights, ranking, tiers, conflicts, cohort selection, universe,
PreBreakout, outcome or maturation formulas, scheduling of the scanner, or any
stored observation or outcome.

## Components

| File | Role |
|---|---|
| `analytics/forward_readiness.py` | Pure monitor: epoch, metrics, gates, state, progress, estimate, anti-peeking guard |
| `scripts/forward_evidence_readiness.py` | Loads the store read-only (or `--input` snapshot); writes `artifacts/research/forward_evidence_readiness.{json,md}`; prints and exports `RUN55_RERUN_RECOMMENDED` |
| `.github/workflows/forward-evidence-readiness.yml` | Weekdays 23:15 UTC (after the last maturation cycle at 22:30) plus manual dispatch. Read-only. Publishes the JSON and Markdown as an artifact and step summary. Never triggers Run 55 |
| `tests/test_forward_readiness.py` | 26 deterministic tests |

## Forward epoch (Part 1)

The epoch starts at **`2026-09-26T07:23:11Z`**, when the Run 55 production
analysis ran (workflow 36226568240). Commits: evaluation `284e2ac`, criteria
`c5d34a7`. Schemas: `hsf-obs-1.0` and `hsf-outcome-1.0`.

- Only explicitly tagged observations **anchored at or after** the epoch count.
- Everything Run 55 inspected is excluded and counted
  (`pre_epoch_observations_excluded`), but never deleted.
- `forward_epoch_start_scan_run` is the first forward regular-session scan run.
- The scanner scoring version comes from the scored rows. Drift during the epoch
  raises a WARN on Gate H.

## Measurement population

Every metric uses the Run 55 primary population:
- regular-session anchors (09:30–16:00 ET);
- explicit cohorts;
- duplicate observation IDs collapsed.

A horizon is **eligible** (settled) once `anchor + horizon + 15 min slack + 45
min grace` has passed, meaning at least one maturation cycle had the chance to run.
Maturation % = matured ÷ eligible.

Unmatured reasons (Part 3):
- **NOT_YET_ELIGIBLE:** the horizon has not settled yet.
- **FILTERED_BY_POLICY:** the symbol is excluded by US_MARKET symbol rules.
- **RETIRED:** the anchor is at least 6 days old.
- **INSUFFICIENT_FUTURE_BARS** (inferred): another horizon of the same
  observation matured.
- **PRICE_DATA_UNAVAILABLE** (inferred): nothing matured after at least 1 day.
- **OTHER:** eligible but less than a day old, i.e. awaiting a cycle.
- **RATE_LIMITED:** per-observation reasons are not persisted, so this is only
  visible at run level, from the latest `maturation_report.json` the workflow
  downloads.

## Pre-registered gates (Part 4)

These were fixed before any forward data existed. Each gate reports
PASS / WARN / FAIL / NOT_APPLICABLE with its reason.

| Gate | Rule |
|---|---|
| A Trading history | Completed forward trading days: FAIL < 10, WARN 10–19, PASS ≥ 20. A day is complete at 17:15 ET, once the 15:35 slot's +60m has matured |
| B Scan-run clusters | Regular-session forward runs: FAIL < 50, WARN 50–99, PASS ≥ 100 |
| C Cohort representation | Every cohort has ≥ 30 matured scan-run clusters at **every** horizon |
| D Horizon maturation | Minimum cohort maturation ≥ 80% at +5m/+15m/+30m and ≥ 70% at +60m. At +60m only, a shortfall is WARN if the parity gap is ≤ 5 pp (cohort-neutral) |
| E Maturation parity | Per horizon, gap = max − min cohort maturation %: PASS ≤ 5 pp, WARN ≤ 10 pp, FAIL > 10 pp |
| F Directional integrity | Stored directional_return / MFE / MAE coverage ≥ 90% of matured forward outcomes. It is "judged" once there are ≥ 100 outcomes |
| G Effective sample | Scan runs in which **both** arms matured (CANDIDATE∩CONTROL, CANDIDATE∩NEAR_MISS), minimum over horizons: FAIL < 20 (Run 55 STRONG), WARN 20–29, PASS ≥ 30. Rows within one run are not independent |
| H Research integrity | FAIL on any PIT violation, conflicting duplicate, conflicting observation ID, CANDIDATE∩CONTROL overlap, direction-transform mismatch, invalid market value, or near-miss overlap > 1%. WARN on scoring-version drift, minor overlaps, or identical duplicates |

## States (Part 6)

- **DATA_QUALITY_BLOCKED:** any of the following:
  - H fails.
  - F fails once judged.
  - D or E fail on a horizon whose parity is **measurable**, meaning every
    cohort has ≥ 100 settled observations from ≥ 5 runs. More time will not fix
    this, so `limiting_factor` = `DATA_PIPELINE_BIAS` (or `DATA_INTEGRITY` for H/F).
- **READY_FOR_RUN55_RERUN:** no gate fails (WARNs allowed).
  `RUN55_RERUN_RECOMMENDED=true`.
- **APPROACHING_READY:** only time-related gates fail (A/B/C/G, or D/E/F not yet
  measurable or judged), and the bottleneck sample-size progress is ≥ 75%.
- **COLLECTING:** healthy but insufficient. `limiting_factor` =
  `NOT_ENOUGH_TIME`, or `NO_FORWARD_DATA` when nothing has been collected yet.

Direction readiness (Part 5):
- LONG is READY only when the overall state is READY, and INSUFFICIENT when blocked.
- SHORT is INSUFFICIENT when there are zero SHORT observations, and READY only
  with ≥ 30 matured +60m SHORT clusters and ≥ 10 trading days.
- SHORT = 0 never blocks LONG, and no SHORT rows are fabricated.

## Progress and estimate (Part 7)

Progress for each gate is its value ÷ minimum, capped at 1:
- trading days;
- scan runs;
- per-cohort clusters;
- horizon coverage;
- parity.

`bottleneck` names the slowest time-related gate. It feeds the state rule.

`estimated_trading_days_until_ready` is a linear extrapolation of observed runs
and matured clusters per completed trading day, taking the worst of A/B/C/G:
- It is **UNKNOWN** with fewer than 2 completed days, when a count is not
  growing, when blocked, or when a non-time gate fails.
- In that last case `estimated_trading_days_until_sample_gates` still reports
  the time-only figure.

## Anti-peeking (Part 9)

The monitor answers "do we have enough evidence?", never "what does it say?":
- **What it reads.** It opens outcome records only to test field *presence* and
  to run equality-based integrity checks: duplicate conflicts, and whether
  directional_return equals ±raw_return.
- **What it never computes.** Win rates, mean or median returns, cohort return
  differences, correlations, buckets, thresholds, or feature and conflict
  effectiveness.
- **Enforcement.** `assert_no_effectiveness_metrics` runs on every report and
  rejects any key that looks like an effectiveness statistic. A test also proves
  the whole report is **byte-identical** when every return is replaced
  (`test_output_is_invariant_to_returns`).

## Validation on Run 55 data

This was a replay of the Run 55 input snapshot with the epoch temporarily moved
back to include it; it was not committed. The monitor reproduced the known
problem from the raw data:
- **+60m maturation:** CANDIDATE 47.6%, NEAR_MISS 37.9%, CONTROL 8.6%, a gap
  of 39.0 pp.
- **+5m maturation:** CONTROL 14.6%, a gap of 40.1 pp.
- **Parity was measurable** at every horizon, so the state was
  **DATA_QUALITY_BLOCKED / DATA_PIPELINE_BIAS**, not "wait longer".

At +60m, 702 of 1,300 settled controls were inferred PRICE_DATA_UNAVAILABLE.
Much of that dataset predated the Run 54 maturation fixes (400-symbol cap,
unretried 429s), so the forward epoch is the first clean test of parity.

**If control parity stays blocked after the backlog drains, that is a design
question, not a bug to patch here.** The control sample is drawn from the broad
universe, many names have no IEX minute bars, and CANDIDATEs are liquid by
construction. The fix, for example liquidity-stratified controls or the SIP feed
for maturation, belongs in a separate run and must not be adjusted inside the
frozen experiment.

No correctness bug was found in maturation during Run 56.
