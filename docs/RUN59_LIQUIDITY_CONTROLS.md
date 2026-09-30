# Run 59 — Liquidity-comparable controls (pre-registered)

**Decision (owner, 2026-09-30): Option A — liquidity-comparable controls, new forward epoch.**

This decision was made from the Run 58 data-quality finding only. It was
**not decided from effectiveness results**: no candidate-vs-control returns,
win rates or other outcome statistics were looked at to choose it.

## Why

Run 58 (`docs/RUN58_MATURATION_PARITY.md`) found forward cohort maturation
parity CRITICAL (> 20 pp): at +60m about 88% of candidates matured versus
about 28% of controls. Root cause: `MARKET_DATA_AVAILABILITY_EFFECT`. Controls
were a seeded sample of the *whole* evaluated universe, whose median capture
dollar volume was 0.11% of candidates'; many such names have no IEX minute
bars. Candidates must pass the scan's liquidity floor, so the comparison was
between liquid breakouts and mostly illiquid names. Gate E (parity ≤ 10 pp)
would block any formal evaluation indefinitely.

Options considered: (A) liquidity-comparable controls, (B) SIP feed for
maturation, (C) wall-clock horizons, (D) pre-registered liquidity-matched
analysis of the existing design. A was chosen: it fixes the cause, costs
nothing, and keeps the comparison simple.

## What changes

1. **Control pool.** Controls are drawn only from evaluated non-candidates that
   pass the same point-in-time rules candidates must pass:
   - 20-day dollar volume ≥ the scan's `min_dollar_vol` (scheduled scans:
     `CRON_MIN_DOLLAR_VOL`, default $5,000,000), computed exactly as
     `scan/breakout.py` does (last close × mean volume of the 20 prior days,
     falling back to today's volume with < 21 bars) — `research_cohorts.dollar_vol20`;
   - last price within the scan's `min_price`–`max_price`.
   A symbol without a computable dollar volume or price is not eligible.
2. **Sampling.** Unchanged otherwise: the same seeded hash of
   (scan run id, symbol), the same size (`RESEARCH_CONTROL_N`, default 100),
   candidates excluded.
3. **Tagging.** New controls carry `market_context.control_design =
   "run59_liquidity_matched_v1"` and `selection_reason =
   "liquidity_matched_sample"`. Observation ids are unchanged (they depend on
   symbol, timestamp and context only). Legacy-design controls are unchanged.
4. **New forward epoch.** `FORWARD_EPOCH` now starts at
   **2026-10-01T12:00:00+00:00** (before the Thursday 8:35 AM ET scan). The Run 56
   epoch (2026-09-26T07:23:11Z) is recorded in `PREVIOUS_EPOCHS` and is never
   used for evidence again.
5. **No pooling of designs.** Inside the epoch, a CONTROL without the Run 59
   tag (e.g. captured before this change reached `main`) is excluded from the
   forward selection and from forward parity, and counted as
   `legacy_control_design_excluded`.

## What does not change

- Scoring, ranking, candidate and near-miss selection, universe, scan output
  (the research sidecar only gains a dollar-volume value and the floor it was
  judged against; `df.head(top_n)` production output is unchanged).
- Maturation, outcomes, horizons, retirement.
- **Gates A–H are unchanged** (≥10 trading days, ≥50 runs, ≥30 clusters per
  cohort, maturation ≥80%/70%, parity ≤10 pp, coverage ≥90%, ≥20 paired
  clusters, integrity). The forward clock restarts from zero.

## What to watch

- The cron log line `[research_cohorts] … control_pool=N liquidity-matched`:
  N should be in the hundreds or more per full-market scan.
- Forward parity on the System Health card should become measurable after a
  few trading days and is expected to fall well below 20 pp. If it stays above
  10 pp, the remaining gap is not liquidity composition and needs its own run.

## Deploy note

The epoch start assumes this reaches `main` before 2026-10-01 12:00 UTC. If it
lands later, controls captured in between are simply excluded (point 5); no
data is lost or mixed.
