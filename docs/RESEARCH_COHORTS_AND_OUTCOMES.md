# Research cohorts & outcomes (Run 47)

A bounded, deterministic, point-in-time-safe research dataset that captures BOTH
what HSF **selected** (candidates) and a meaningful representation of what it
**rejected** (near-misses + controls), with future outcomes kept physically
separate — preparing for Run 48 signal-effectiveness analysis.

**No scoring/scanner/ranking/ML change.** Cohorts are labels over existing scan
outputs; production `results` (top-N) are byte-for-byte unchanged. Controls are
selected by a seeded hash of (scan_run_id, symbol) — never by future returns.

## Cohort architecture

Per scheduled US_MARKET scan (side-effect-only capture, guarded by
`HSF_RESEARCH_CAPTURE`, default on):

| Cohort | Definition | selection_reason | Features |
| --- | --- | --- | --- |
| **CANDIDATE** | the production top-N results | `top_n_candidate` | full |
| **NEAR_MISS** | ranked rows immediately **below** the TOP_N cut | `rank_below_cutoff` | **full** (same builder as candidates) |
| **CONTROL** | deterministic seeded sample of the broad evaluated non-candidate universe | `deterministic_sample` | compact (price/volume; richer technicals honestly absent — these were filtered before scoring) |

The engine exposes a `research_sink` (opt-in) that returns the near-miss slice +
the evaluated symbol list + a cheap price snapshot, **after** slicing production
`results` back to exactly `top_n`. No ranking/scoring is altered.

## Selection rules & bounds

- **NEAR_MISS:** request `top_n + RESEARCH_NEAR_MISS_N` rows from the breakout
  stage; candidates = first `top_n` (unchanged), near-misses = the next N.
- **CONTROL:** `select_control_symbols` sorts evaluated non-candidate symbols by
  `sha256(scan_run_id|symbol)` and takes the lowest-N — reproducible, seeded by
  point-in-time identity only, never outcomes.
- **Defaults:** `RESEARCH_NEAR_MISS_N=50`, `RESEARCH_CONTROL_N=100`, **hard cap
  500 each** — config can never accidentally persist the whole market.
- Typical per scan: 100 candidate + 50 near-miss + 100 control = **~250 research
  rows**.

## Feature fidelity

Candidates and near-misses share the same builder → identical feature schema for
fair comparison (market: price/volume; technical: rvol/vs_vwap/adx/gap/chg;
model: prebreakout probability; provenance/versions/data_quality). Run 46's
missing EMA/RSI/SuperTrend/EWO/VWAP remain **null** — they are not present in the
breakout candidate row and Run 47 does **not** recompute them or add provider
calls (documented limitation). Controls carry identity + price/volume only.

## Point-in-time safety (all cohorts)

Every cohort stores features known at T only. Run 46's leakage checks
(`check_point_in_time`) extend to near-miss and control records (tests enforce 0
violations). Cohort/selection_reason are point-in-time labels; `selection_reason`
never encodes future performance.

## Outcome architecture, horizons & formulas

Outcomes stay in the **separate** `hsf_observation_outcomes` table, joined by
`observation_id` — never inside the feature record. Horizons: **+5m / +15m / +30m
/ +60m / EOD**, each recorded `MATURED` / `PENDING` / `UNAVAILABLE`. Metrics
(computed in `analytics.observation_capture.compute_matured_outcomes`, reusing
`day_trade_validation`):
- `raw_return = (P_future − P_ref) / P_ref`
- `directional_return = raw_return` (LONG) or `−raw_return` (SHORT)
- `MFE` = max favorable excursion over the window (upward for LONG, downward for
  SHORT); `MAE` = max adverse excursion (direction-aware).
- `future_high/low/close` live only in the outcome record.

A `+60m` horizon is not mature until ≥ 60m (+ slack) have elapsed
(`horizon_eligibility`); the maturation worker measures from the precise
`scan_timestamp`, never the bucketed id, and the `build_outcome` guard rejects any
evaluation ≤ observation time.

## Session boundaries

Premarket/postmarket slots route to the session scan (not US_MARKET), so
US_MARKET research observations are regular-session. Intraday horizons that would
cross the session close simply have no bars → `PENDING`/`UNAVAILABLE`, never a
silently cross-session value. EOD maturation respects session close.

## Maturation

`scripts.mature_observations` matures candidate, near-miss, and control
observations identically (they share the store): per-horizon eligibility,
idempotent first-write-wins (`save_outcome`), symbol-grouped single fetch,
failure taxonomy, safe to rerun. `validate_outcome` flags missing entry price,
non-finite return, outcome-before-observation, invalid future price — never
silently zero-filled.

## Storage impact

- Per scan: ~250 research rows → ~750/day (3 regular slots) → ~189k/year.
- ~2 KB each ⇒ **~0.4 GB/year** — a large research-quality gain **without** the
  ~116× write increase / ~24 GB/year of full-universe capture (Run 46).

## Research export

`analytics.observation_integrity.research_export` /
`scripts.observation_health --export`: stable columns (incl. `research_cohort`,
`selection_reason`), deterministic ordering, **outcomes excluded by default**.
`include_outcomes=True` (explicit) joins matured outcomes under an obvious
`outcomes` key. Optional `cohorts`, `healthy_only`, date range, horizon selection.

## Backward compatibility

Legacy candidate observations without `research_cohort` are treated as
**CANDIDATE** (`cohort_of`) — no destructive migration, no rewrite of old
production records. Historical Replay, scanner-performance, outcome maturation,
and existing exports are unchanged.

## Known limitations

- Control records are **compact** (price/volume only) — the broad rejected
  universe never had full technicals computed.
- EMA/RSI/SuperTrend/EWO/VWAP remain null (not in the candidate row; not
  recomputed per Run 47 rules).
- Near-miss depth is bounded (default 50) — a small, deterministic slice below the
  cut, not the full ranked tail.

## Run 48 readiness

The dataset now contains selected (candidate), close-rejected (near-miss,
full-feature), and broad-rejected (control) cohorts with a deterministic,
point-in-time-safe schema and separate, direction-aware outcomes joinable by
`observation_id`. Run 48 can compare signal effectiveness across cohorts without
storing the whole market.
