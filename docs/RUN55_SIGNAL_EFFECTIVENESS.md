# Run 55 — Research evidence & signal effectiveness

**Verdict: INSUFFICIENT_EVIDENCE** · evidence quality **INSUFFICIENT** ·
Run 56: **F. COLLECT_MORE_DATA**

Run 55 is an evaluation run. It changed no scoring, ranking, tier, cohort,
universe, scheduling, PreBreakout, or outcome logic, and wrote no observation or
outcome. The analysis reads the production store and reports what it finds.

## How the verdict was produced

- **Pre-registration.** All decision rules (`CRITERIA`, `readiness`,
  `primary_verdict`, `run56_recommendation` in `analytics/signal_evidence.py`)
  were committed in `c5d34a7` before the analysis ran on production data.
  Nothing was tuned afterwards.
- **Run.** Read-only workflow `signal-effectiveness.yml`,
  [run 36226568240](https://github.com/realhowieb/ai_scanner/actions/runs/36226568240)
  (2026-09-26 07:23 UTC, Neon).
- **Outputs.** `artifacts/research/run55_signal_effectiveness.json` and `.md`
  hold the full tables for all 14 sections. The exact input snapshot
  (`run55_input_snapshot.json`, 10 MB) is attached to the workflow run. To replay:
  `python -m scripts.analyze_signal_effectiveness --input run55_input_snapshot.json`.
- **Primary population.** Explicitly tagged (Run 47+) observations, anchored in the
  regular session (09:30–16:00 ET), with MATURED outcomes, deduplicated across
  cohorts. The 520 legacy untagged observations are excluded from every analysis.
- **Statistics.** Directional returns are winsorized per horizon at the pooled
  0.5/99.5 percentiles. Differences and correlations get 95% CIs from a cluster
  bootstrap over scan runs. The pre-declared meaningful effect is 25 bp.
  EDGE_DETECTED requires CANDIDATE > CONTROL with MODERATE+ evidence at ≥ 3 of 4
  horizons. NO_EDGE_DETECTED requires the null to be powered at ≥ 3 horizons.
  Anything else is INSUFFICIENT_EVIDENCE.

## Answers

| # | Item | Result |
|---|---|---|
| 3 | Observations analyzed | 4,233 total: 3,713 explicit, 520 legacy (excluded) |
| 4 | Matured observations analyzed | 1,523 explicit with ≥ 1 matured horizon (5,569 outcome records). The primary population at +60m is 913 |
| 5 | Coverage by horizon (explicit, CANDIDATE / NEAR_MISS / CONTROL) | +5m 857/365/301 · +15m 849/356/242 · +30m 798/335/205 · +60m 768/315/178 |
| 6 | Cohort counts (explicit) | CANDIDATE 1,400 · NEAR_MISS 713 · CONTROL 1,600 |
| 7 | Candidate − control (Δ mean, 95% cluster CI) | +5m **+0.26%** [+0.10, +0.38] MODERATE · +15m +0.09% [−0.19, +0.25] · +30m +0.12% [−0.66, +0.76] · +60m +0.48% [−0.13, +0.94] INSUFFICIENT (9 runs) |
| 8 | Candidate − near-miss | +5m +0.06% [+0.01, +0.15] · +15m +0.09% [−0.01, +0.19] · +30m +0.21% [+0.01, +0.42] (9 runs, INSUFFICIENT) · +60m +0.20% [−0.08, +0.55] |
| 9 | Score monotonicity | **INCONCLUSIVE**. Spearman CI > 0 at 4/4 horizons, but 2 adjacent-bucket violations at +60m (the pre-declared rule allows at most 1) |
| 10 | Tier separation | **INCONCLUSIVE**. Production tiers are not persisted on observations (0 of 913 carry one) |
| 11 | Strongest positive signal | BreakoutScore rank vs directional return: Spearman +0.05 to +0.12, CI > 0 at every horizon. At +5m the win rate rises from 35.3% in the lowest score quintile to 56.6% in the highest |
| 12 | Strongest negative / failed assumption | A selected CANDIDATE is not, on its own, a winning long. Its +60m win rate is 44.0% [39.9, 48.1] (below a coin flip) and its mean is −0.09%. At +60m every cohort lost on average. Any edge is relative: CONTROL lost more |
| 13 | Overall verdict | **INSUFFICIENT_EVIDENCE** |
| 14 | Evidence quality | **INSUFFICIENT** |
| 15 | Run 56 action | **F. COLLECT_MORE_DATA**, with the specifics below |

## Readiness: CONDITIONALLY_READY

- **Integrity is clean.** There are 0 point-in-time violations, 0 duplicate or
  conflicting outcomes, 0 orphans and 0 invalid values. 5 cross-cohort overlaps
  were deduplicated. 8 returns exceed ±50% (kept, then winsorized).
- **Failed conditional gates:**
  - Only 16 scan runs (fewer than 20).
  - Stored `directional_return` and MFE/MAE exist on only 54.7% of outcomes. Rows
    matured before Run 53A are derived from `raw_return`; their MFE/MAE cannot be
    recovered.
  - SHORT n = 0.
- **Transform check.** Of 2,451 stored `directional_return` values, 0 disagree
  with the direction transform.

## Why the evidence is insufficient

1. **Three trading days.** Every explicit observation comes from 2026-09-23 to
   09-25, which is 13 regular-session runs. The cluster bootstrap has 8–11
   clusters per comparison, so no horizon beyond +5m reaches the pre-declared
   power, and intervals with this few clusters tend to be too narrow.
2. **Maturation survivorship.** At +60m, 571 of 1,200 regular-session
   CANDIDATEs have an outcome (48%) but only 112 of 1,300 CONTROLs do (8.6%).
   Controls are a broad-universe sample and often have no IEX bars, and the ~4,900
   outcomes found by the 2026-09-26 dry run are not written yet (the next
   scheduled maturation is Monday). The matured CONTROL subset is therefore not a
   random sample of the universe.
3. **The one positive horizon is fragile.** This is a post-hoc sensitivity check,
   not used for the verdict. Restricted to runs that contain both cohorts, the
   +5m difference *averaged per run* is **−0.05%**, positive in only 6 of 10 runs.
   The pooled +0.26% comes from how runs are weighted. At +60m the run-matched
   difference is +0.61% (5 of 6 runs positive), with only 6 runs.

## Other sections (full tables in the artifact)

- **LONG vs SHORT: INSUFFICIENT.** Every CANDIDATE and NEAR_MISS row's first
  scanner is `breakout` (long). CONTROL has no direction and is measured long.
- **Time of day: reported as NOT_MATERIAL, but read it as insufficient.** The
  pre-declared rule compares candidate-bucket CIs, which are i.i.d. intervals,
  while each bucket holds only 2–3 scan runs. 10:30–12:00 has no scans because
  the scheduled slots are 09:35 / 12:35 / 15:35 ET. Post-market candidates
  (+0.74% at +60m) are measured on sparse extended-hours bars.
- **Regime: REGIME ANALYSIS UNAVAILABLE.** `market_regime` is never populated at
  capture. It was not reconstructed, to avoid leakage.
- **Features (CANDIDATE + NEAR_MISS, 14 features × 4 horizons, so expect ~3
  spurious hits):**
  - USEFUL: `score`, and `price` (CI > 0 only at +60m).
  - NEUTRAL: rvol, gap, % change, ATR%, volume, dollar volume, trigger count,
    gap_up, gap_down.
  - INSUFFICIENT: `is_breakout` (−0.42% at +60m, n = 56), `unusual_vol`
    (+0.15 to +0.45%, n = 88) and `momentum` (only ~100 "without" rows).
  - Not persisted, so not measurable: conflicts, confirmation/agreement,
    EMA 9/21, RSI, breakout distance, PreBreakout, VWAP/ADX/SuperTrend/EWO.
- **MFE/MAE (+60m only):**
  - Excursions widen with score. Mean MFE goes from +0.59% in Q1 to +1.93% in
    Q5, and the MFE/|MAE| ratio from 0.63 to 1.12.
  - Q4–Q5 reverse after a ≥ 1% favourable move in 6.8–8.8% of cases, against
    1.2% in Q1.
  - Early-adverse-then-winner rates are flat (~17%) across quintiles.
  - Higher scores therefore pick bigger movers in both directions. That is
    consistent with a volatility effect rather than a clean directional edge.

## Run 56 recommendation: F. COLLECT_MORE_DATA

The Run 56 code is not implemented here. Concretely:

1. **Let maturation catch up.** Monday's scheduled maturation should write about
   4,900 outcomes (the 2000 cap now covers the whole backlog). Re-run
   `signal-effectiveness.yml` unchanged afterwards. The criteria stay frozen.
2. **Accumulate runs.** The pre-declared STRONG level needs ≥ 20 scan runs per
   comparison. At the current +60m MDE, CANDIDATE − CONTROL needs about 910 more
   matured CONTROL rows. At the observed 8.6 regular-session controls per run that
   is about 100 runs (~7 weeks). If the maturation fixes lift the control rate
   toward the candidate rate, it falls to about 20–25 runs (~2 weeks).
3. **Watch control maturation.** Before trusting CANDIDATE − CONTROL, check the
   matured share of controls. If it stays far below candidates after the backlog
   drains, the comparison needs a liquidity-matched control. That is a data-quality
   question (G), to be decided then.
4. **Instrumentation, not scoring.** Parts 4, 8 and 9 cannot be answered from
   current records. Persisting point-in-time DT tier, conflicts, EMA 9/21,
   breakout distance, PreBreakout and market regime at capture is the only way to
   evaluate them later.
