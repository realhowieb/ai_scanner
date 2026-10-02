# Run 59B — Near-miss names excluded from controls (pre-registered)

**Decision (owner, 2026-10-02): exclude near-miss names from the control draw;
new control design and new forward epoch.**

This decision was made from a data-integrity gate only. It was
**not decided from effectiveness results**: no candidate, near-miss or control
returns, win rates or other outcome statistics were looked at to choose it.

## Why

Forward Evidence Readiness reported `H_research_integrity ❌ FAIL —
near-miss overlaps=22 (> 1%)` on the second day of the Run 59 epoch.

`scheduler/cron_runner._capture_research_cohorts` drew controls with
`exclude=candidate_symbols` only. Near-miss names (the ranked rows just below
the candidate cut) are evaluated, liquid symbols, so they are in the Run 59
liquidity-matched control pool. Before Run 59 that pool was the whole evaluated
universe (~11,000 names) and a near-miss was rarely drawn as a control. Run 59
restricted it to liquidity-matched names (about 180 eligible, 100 drawn), so a
symbol appearing as both NEAR_MISS and CONTROL in the same scan became routine.
The records are correct; the cohorts were simply not disjoint by construction.

Options considered: (A) exclude near-misses from the control draw and restart
the epoch, (B) keep the design and accept the overlap. A was chosen: it makes the
three cohorts disjoint by construction, and only two days of the Run 59 epoch
are lost.

## What changes

1. **Control draw.** Controls exclude candidates **and near-misses**:
   `select_control_symbols(pool, exclude=candidates + near_misses)`. Pool
   (Run 59 liquidity floor and price range), seeded hash and sample size
   (`RESEARCH_CONTROL_N`, default 100) are unchanged.
2. **Tagging.** New controls carry `market_context.control_design =
   "run59b_liquidity_matched_disjoint_v2"`. Observation ids are unchanged.
3. **New forward epoch.** `FORWARD_EPOCH` starts at
   **2026-10-05T12:00:00+00:00** (before the Monday 8:35 AM ET scan). The Run 59
   epoch (2026-10-01T12:00:00Z, `run59_liquidity_matched_v1`) is recorded in
   `PREVIOUS_EPOCHS` and is never used for evidence again.
4. **No pooling of designs.** A CONTROL without the Run 59B tag is excluded from
   the forward selection and counted as `legacy_control_design_excluded`.

## What does not change

- Scoring, ranking, candidate and near-miss selection, universe, scan output.
- Maturation, outcomes, horizons, retirement.
- **Gates A–H are unchanged** (including the 1% near-miss overlap limit in
  H_research_integrity). The forward clock restarts from zero.
- Records already captured are never rewritten.
