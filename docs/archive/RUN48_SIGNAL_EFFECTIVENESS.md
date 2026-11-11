# Run 48 — signal effectiveness & selection-value analysis

**Measurement only. No production scoring/scanner/ML/ranking changed.** This run
delivers a reproducible, point-in-time-safe, cohort-/sample-/uncertainty-aware
framework to answer: *does HSF select better stocks than it rejects, and which
signals carry evidence?* — and runs it against the available Run 47 research
dataset.

> **EFFECTIVENESS VERDICT: INSUFFICIENT LIVE DATA.**
> Run 47 cohort capture began **2026-09-23** (this session), and the outcome
> maturation worker has not accumulated matured, paired candidate/control
> outcomes at the required scale. The framework is complete and validated on
> fixtures; production findings must wait for data. This is a **FRAMEWORK PASS**,
> not a failure.

## Why insufficient (the honest state)

Effectiveness requires **matured, direction-adjusted outcomes** for both the
CANDIDATE and CONTROL cohorts, at N ≥ 30 paired. As of this run:
- Cohorts have existed for < 1 day (first live capture 2026-09-23).
- Only a handful of forced scans have written ~250 research rows each.
- Intraday outcomes (+5m…+60m) mature only after the maturation worker runs over
  aged observations; EOD needs the session to close.
- `usable_paired_n` is effectively 0 → `evidence_level = INSUFFICIENT`.

The runner reports this honestly (`EFFECTIVENESS VERDICT: INSUFFICIENT LIVE DATA`)
rather than manufacturing conclusions.

## Evidence-level rule

By usable paired sample size: **INSUFFICIENT** (< 30) · **PRELIMINARY** (< 100) ·
**MODERATE** (< 500) · **STRONG** (≥ 500). Every breakdown reports N and its
level; conclusions below MIN_SAMPLE (30) are suppressed/labeled.

## What the framework computes (all live once data matures)

- **Cohort performance** (`cohort_performance`): per cohort × horizon — N, mean &
  **median** directional return, win rate (+ Wilson CI), median MFE/MAE,
  positive-% and meaningful-move-% (robust, not mean-only).
- **Selection lift** (`selection_lift`): CANDIDATE vs CONTROL and vs NEAR_MISS —
  median-difference estimate with a **deterministic seeded bootstrap CI of the
  difference** (Task 4/5).
- **Ranking** (`rank_bucket_analysis`): do higher-ranked candidates outperform
  lower ones (Task 18)?
- **Score bands** (`signal_effectiveness` on prebreakout / ai_confidence):
  quantile bands, monotonicity flag, dead-range/pile-up detection (Task 6/7/8).
- **Individual signals** (rvol/adx/vwap/ema/rsi/supertrend/ewo/gap/chg): quantile
  bins → outcome relationship, non-linear-friendly (Task 9).
- **Grouping** (`group_analysis`): direction (LONG/SHORT), regime, session/
  time-of-day (Tasks 12–14).
- **Scorecard** (`signal_scorecard`): STRONG POSITIVE / POSITIVE / MIXED / NEUTRAL
  / NEGATIVE / INSUFFICIENT DATA per signal (Task 24).
- **Recommendations** (`build_recommendations`): KEEP / INVESTIGATE / SIMPLIFY /
  RECALIBRATION / POTENTIAL REMOVAL / INSUFFICIENT — written to
  `artifacts/automation/run48_recommendations.json` for Run 49 (Task 25/26).

## Multiple-testing discipline

`ANALYSIS_MANIFEST` separates **PRE-SPECIFIED** (cohort performance, selection
lift, rank buckets, PreBreakout bands, scorecard) from **EXPLORATORY** (per-signal
bins, regime/session/direction splits, combinations). Exploratory results are
uncorrected and must not be presented as confirmed; effect size + CI + N accompany
every finding. Combination analysis is limited to existing product hypotheses (no
brute-force).

## Fixture validation (framework works when data exists)

On a synthetic dataset (200 candidates ~+0.4%, 200 controls ~0%, 100 near-misses
~+0.2%, matured +60m), the framework correctly reported: `evidence=MODERATE`,
**candidate-vs-control median lift +0.54% (95% bootstrap CI [+0.25%, +0.76%])**,
cohort medians ordered CANDIDATE > NEAR_MISS > CONTROL. This demonstrates the
framework detects selection value when the sample supports it — it is not biased
toward null.

## Report answers (pending live data)

A–I (does HSF add selection value; strongest/weakest signals; do higher scores
mean better outcomes; candidates vs near-misses/controls; favorable/hurtful
conditions; Run 49 targets) are all computed by the framework but currently return
**INSUFFICIENT** — no trustworthy production claim can be made yet.

## Recommended accumulation before rerun

- **Schedule the maturation worker** to run against the accumulating cohorts (it
  exists: `.github/workflows/mature-observations.yml`).
- Accumulate **≥ ~2–4 weeks** of HEALTHY scheduled scans so paired
  candidate/control matured outcomes reach **MODERATE** (≥ 500 paired) at the
  key horizons; PRELIMINARY (≥ 100) is the earliest point for tentative reads.
- Then rerun `python -m scripts.run48_effectiveness` (or via a workflow with the
  Neon secret) — the verdict flips to SEE_REPORT with real statistics.

## What Run 49 should do

Nothing yet. Run 49 acts only on evidence: it should consume
`run48_recommendations.json` **after** the dataset reaches at least PRELIMINARY/
MODERATE, and prioritize the earlier tier-classifier concern (agreement/conflict
gates) and PreBreakout monotonicity — both of which this framework will quantify.

## Known limitations

- No live conclusions yet (data immaturity — the central point).
- Control records are compact (Run 47) → some per-signal analyses apply mainly to
  candidate/near-miss cohorts (which carry full features).
- Observational analysis is not causal; ablation-style reads are hypotheses.
