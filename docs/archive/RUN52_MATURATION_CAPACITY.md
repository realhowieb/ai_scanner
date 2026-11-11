# Run 52 — maturation capacity and backlog audit

Can the observation-maturation pipeline keep up with the research data now
produced by full-market scheduled scans? **No scoring/ML/ranking/threshold/
outcome-definition change.** One low-risk correctness fix (outcome-aware loading)
plus the requested backlog telemetry were applied; everything else is measurement.

## Verdict: **CAPACITY RISK** (measured pre-fix) → dominant driver now fixed

Before this run the 400-symbols/run cap **bounded runtime but not the backlog**:
the ready set (738) exceeded the cap (400), ~73% of each run's fetch budget was
spent *re-processing already-matured observations*, un-priceable symbols were
retried every run forever, generation was climbing ~4.5×/day, and the worker was
actually firing ~3–4×/day (GitHub throttles the `*/30` schedule), far below the
nominal ~20. The applied fix removes the largest waste vector (redundant
re-processing); a second vector (infinite retry of no-data symbols) remains and is
the top recommendation. Re-classify to HEALTHY once the new telemetry confirms
`deferred_symbols≈0` on the next live runs.

## Measured evidence (production CI, mature-observations.yml)

| run | when (UTC) | scanned | ready_sym | deferred | +5m new/already/failed | +60m new/already/failed | attached |
|---|---|---:|---:|---:|---|---|---:|
| 35776071123 | 09-22 19:49 | 410 | — | — | 307 / 0 / 3 | 267 / 0 / 43 | 1182 |
| 35792923567 | 09-22 22:43 | 420 | — | — | 99 / **307** / 14 | 12 / 267 / 141 | 209 |
| 35896673367 | 09-23 17:35 | — | — | — | (cancelled — 15-min timeout, pre-Run-38B bump) | | — |
| 35911847604 | 09-23 19:50 (dry) | 1870 | **738** | **338** | 1030 / 0 / 229 | 943 / 0 / 316 | 3981* |

\*dry-run: would-be writes, nothing persisted. `attach_outcomes` was not yet live,
so `already` reads 0 in dry-runs even for matured work.

### What the numbers prove

1. **Generation is climbing.** Same time of day, `scanned` went 410 → 1870
   (~4.5×) in 24h as full-market scans ramped. Per scan ≤150 research
   observations (`near_miss` ≤50 + `control` ≤100), ~one distinct symbol each.
2. **Ready > capacity.** 738 ready symbols vs a 400 cap ⇒ **338 (46%) deferred
   every run.**
3. **~73% of work was redundant.** Run 35792923567 (real, not dry) shows
   `+5m already=307` vs `new=99`: 307 observations had already matured on a prior
   run but were re-fetched anyway. Root cause: `load_recent_observations` selected
   only `record` and never joined `hsf_observation_outcomes`, so the worker's
   `horizon_eligibility(..., o.get("outcomes"))` check always saw an empty set and
   re-marked matured horizons as `ready`.
4. **Oldest-first inverted priority.** Because matured observations stayed in the
   ready set and sort oldest-anchor-first, the 400 budget was spent on the oldest
   (most-likely-already-done) rows while the *newest, genuinely-fresh* observations
   were the ones deferred — i.e. fresh observations could be starved once ready
   persistently exceeded 400.
5. **Maturation is throttled, not hourly.** Observed cadence is ~3–4 runs/day
   (19:49, 17:35, prev-day 22:32, 19:47), not `*/30`. GitHub silently drops most
   scheduled ticks, so real capacity ≈ 4×400 = 1,600 symbol-fetches/day nominal,
   far less after redundant reprocessing.
6. **Un-maturable symbols retried forever.** `PRICE_DATA_UNAVAILABLE=716` +
   `INSUFFICIENT_FUTURE_BARS=339`. These never get an outcome row, so they never
   leave the ready set and are re-fetched every single run — permanent budget drain
   and a starvation vector independent of #3.
7. **Longer horizons mature worse.** +60m fails more than +5m (316 vs 229;
   141 vs 14) — later horizons need 60+ min of future bars that thin symbols and
   near-close anchors lack. Correctness is fine; completeness is horizon-skewed.

## Answers to the audit questions

| # | Question | Finding |
|---|---|---|
| 1 | Observations awaiting maturation | ~1,870 in-window on 09-23; not directly counted pre-fix (no telemetry) — now `ready_observations`/`ready_symbols` are reported. |
| 2 | Distinct symbols with ready horizons | **738** (latest measured). |
| 3 | Symbols processed / run | ≤ **400** (cap); ~73% were redundant pre-fix. |
| 4 | Symbols deferred by cap | **338/run** (46%). |
| 5 | Oldest unmatured age | Not measured pre-fix; now `oldest_pending_age_min` (oldest still-pending anchor after the run). |
| 6 | New observations / hour·day | ≤150/scan; ~4.5×/day growth trend; needs the new telemetry over several days to fix a rate. |
| 7 | Matured / hour·day | ~1,182 outcome-rows in a good run; effective fresh-symbol drain far below 400 pre-fix. |
| 8 | Net backlog growth/drain | **Growing** pre-fix (ready 738 > cap 400, generation rising, throttled runs, redundant reprocessing). |
| 9 | Time to clear backlog | `estimated_clearance_runs = ceil(ready/cap)` = **2 runs** *if* no redundant work and no new arrivals — not true pre-fix, so effectively unbounded. |
| 10 | Is 400/run sustainable? | **Not** while matured + no-data symbols stay in the ready set. Sustainable after the fix if fresh generation < cap × runs/day. |
| 11 | Symbols consuming excessive fetch time | The 716 no-data + 339 insufficient-bar symbols are re-fetched every run for zero yield. |
| 12 | Failures/retries → starvation | **Yes**, two vectors: matured-reprocessing (fixed) and no-data infinite retry (open). Plus a latent 5,000-row load-window cap (`--limit 5000`) that would evict the oldest once volume exceeds it. |
| 13 | All horizons mature correctly | Values correct; +60m completeness lags +5m due to insufficient future bars. |
| 14 | First-write-wins / idempotency | **Verified.** `already=307` = `save_outcome` ON CONFLICT DO NOTHING; matured labels are never rewritten. |
| 15 | Failed run safely resumes | **Verified.** The 17:35 timeout lost no data; the 19:49 run re-attempted cleanly. Maturation only writes `hsf_observation_outcomes`, idempotently. |

## Fix applied this run (smallest safe architectural fix)

**Outcome-aware loading.** `load_recent_observations(..., attach_outcomes=True)`
now attaches each observation's already-matured horizons via a single aggregate
LEFT JOIN (no N+1). The worker (`main()`) uses it, so matured horizons classify as
`already` and are **not re-fetched**. This is a query/wiring fix — the worker's
eligibility check was already written to read `o["outcomes"]`; the loader simply
never populated it. No outcome values, scoring, ranking, ML, or thresholds change.
Expected effect: ready-set drops by roughly the redundant fraction (~73% in the
worst observed run), bringing ready under the 400 cap and driving `deferred_symbols`
toward 0. It also relieves the 5,000-row window (matured rows no longer occupy
ready slots).

**Backlog telemetry** added to the report (`schema hsf-maturation-1.1`):
`ready_symbols`, `ready_observations`, `processed_symbols`, `deferred_symbols`,
`matured_observations`, `failed_symbols`, `oldest_pending_age_min`,
`estimated_clearance_runs`, `max_symbols` — printed each run and written to
`maturation_report.json`, so backlog growth/drain is now directly observable.

## Recommendations (ranked)

- **P0 (applied):** outcome-aware loading so matured work leaves the ready set.
- **P1:** stop infinite retries of un-maturable symbols. Smallest safe option:
  give up per (observation_id, horizon) once the anchor is older than a bound
  (e.g. > 1 trading day) and no bars exist — record it in a lightweight
  `maturation_attempts`/negative-cache table (NOT as a real outcome, to preserve
  outcome definitions), so 716+ dead symbols stop consuming budget every run.
- **P1:** raise the effective run cadence instead of the per-run cap — the cap is
  not the bottleneck, run frequency is. Trigger `mature-observations` from the same
  external scheduler that drives scans (cron-job.org `workflow_dispatch`) rather
  than relying on GitHub's throttled `*/30` schedule.
- **P2:** monitor the new telemetry; alert if `deferred_symbols > 0` for N
  consecutive runs or `oldest_pending_age_min` trends up. Revisit the `--limit
  5000` load window once daily volume approaches it (raise, or page by
  unmatured-only).

Do **not** raise `--max-symbols`: the evidence shows the cap was not the binding
constraint — wasted budget and throttled cadence were.

## Test / lint

Full suite **1,382 passed** / 9 skipped (2 new: outcome-aware skip + backlog
telemetry); ruff `E9,F,I` clean.
