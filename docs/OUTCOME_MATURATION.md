# Outcome maturation (Run 38B)

Completes the production feedback loop: **scheduled scan → canonical observation
→ durable storage → horizon maturation → outcome → scoreboard**. The maturation
worker attaches matured intraday outcomes to captured observations without ever
touching scans, scanners, models, or the observations' features.

## Worker architecture

`scripts/mature_observations.py` (orchestration) + `analytics.observation_capture`
(pure logic). Flow:
1. Load recent observations (`db.hsf_observations.load_recent_observations`).
2. Group by symbol → fetch each symbol's minute bars **once per run** (Task 15).
3. Per observation, compute per-horizon eligibility (`horizon_eligibility`).
4. For `ready` horizons, compute outcomes from future bars
   (`compute_matured_outcomes`, reuses `analytics.day_trade_validation`).
5. Attach idempotently (`save_outcome`, first-write-wins).
6. Emit a machine-readable `maturation_report.json` artifact + text summary.

`mature_observations(...)` takes injectable `fetch_bars`/`save_fn` so the full
pipeline is deterministically unit-tested without network/DB.

## Schedule

`.github/workflows/mature-observations.yml`: `*/30 13-22 * * 1-5` (every 30 min
during US market hours + a post-close sweep so the last hour's +60m horizons
mature), plus `workflow_dispatch` (with `slack_min` / `dry_run` inputs). 15-min
timeout, `concurrency` guard, report uploaded as an artifact. Reuses the same
`DATABASE_URL` / Alpaca secrets as the scheduled-scans workflow.

## Horizon semantics

Horizons `+5m / +15m / +30m / +60m` (minute-bar counts). A horizon is **eligible
only when `now ≥ observation_anchor + horizon + slack`** (`slack_min` default 15,
for provider bar-publish latency). Each horizon matures **independently** the
moment its own wall-clock elapses — partial maturation is normal:

```
14:00 obs → 14:25 run: +5m new; +15/30/60 not_ready
        → 15:20 run: +5m already; +15/30/60 new
```

`market_close` / `next_trading_day` are defined in the schema but not yet emitted
by this worker (they need daily-bar alignment) — documented as a follow-up.

## Price / bar alignment

- **Anchor** = the observation's precise `scan_timestamp` (UTC), never the
  hour-bucketed id timestamp.
- Bars are filtered to those **at or after** the anchor minute; `prices_after[0]`
  is the signal-time reference price; forward returns use strictly-later bars.
- All timestamps are parsed to timezone-aware UTC before comparison.
- **No fabricated prices:** if the provider has no bar at a horizon, that horizon
  is not attached and is counted `INSUFFICIENT_FUTURE_BARS`; it retries next run.
- Premarket/after-hours/illiquid: only real returned bars are used; gaps simply
  defer maturation. The structural no-lookahead guard in
  `hsf_observation.build_outcome` rejects any evaluation at/before the anchor.

## Outcome calculations

Per horizon: `raw_return` (from reference to the horizon bar),
`directional_return` (long = raw; short inverts — only where the scanner
expresses a direction), plus `mfe`/`mae` over the window. Raw market outcomes are
preserved independently of direction. Fields with no data are omitted, not zeroed.

## Failure taxonomy (Task 8)

`PRICE_DATA_UNAVAILABLE, INSUFFICIENT_FUTURE_BARS, INVALID_TIMESTAMP,
MARKET_CLOSED, PROVIDER_ERROR, DATABASE_ERROR, UNKNOWN, RATE_LIMITED`. Counted per
run; one bad symbol never blocks others (each symbol group is isolated in try/except).

Since the maturation-hardening run, `PRICE_DATA_UNAVAILABLE` means the provider
**answered** and had no bars (true missing data). HTTP 429 that persists through
bounded retries is `RATE_LIMITED` (symbol-level, retried next run) and a failed
request is `PROVIDER_ERROR` — neither is ever reported as missing data.

## Idempotency

`(observation_id, horizon)` is unique with first-write-wins
(`ON CONFLICT DO NOTHING`). Reruns skip already-matured horizons; nothing is
overwritten. Safe to run every 30 minutes indefinitely.

Ready horizons older than 6 days (4-day bounded fetch window plus 2-day grace)
are **retired**: they are skipped without fetching, because a retry cannot change
the result. This writes nothing. `--retire-after-days 0` re-attempts them for a
backfill. See `docs/MATURATION_RETRIEVAL_HARDENING.md`.

## Quality filtering & scoreboard integration

`analytics.scanner_performance.from_canonical_observations` now carries
`source`, `feature_completeness`, `fallback`, `stale`.
`filter_research_records(...)` defaults to **scheduled, non-fallback, non-stale,
completeness ≥ 0.5** — manual/test/reconstructed sources are never silently
mixed. `sample_readiness(...)` reports progress toward Run 38's gates
(`MIN_SAMPLE=30`, `STRONG_SAMPLE=100`) as a **readiness indicator only** (N ≥ 30
is not statistical significance).

## Operational load (estimates)

- Executions: ~18 runs/trading day (every 30 min, 13:00–22:00 UTC).
- Observations evaluated/run: bounded by `--limit` (default 5,000).
- **Price requests: one paginated multi-symbol series per 100-symbol batch**
  (`/v2/stocks/bars?symbols=…`, 10k bars/page shared across symbols), not one
  request per symbol. Each symbol's bars are retrieved once per run and reused by
  all its observations. See `docs/MATURATION_RETRIEVAL_HARDENING.md`.
- DB: reads = recent observations; writes = only new matured outcome rows.

## Retention

Observations are **never deleted** after maturation — history is an asset (Task
16). No automatic deletion exists. If growth becomes a concern later, archive
(cold storage / partition) rather than discard; do not add deletion.

## Audit

`scripts/observation_audit.py` (read-only) reports totals, today, by
source/scanner/session, quality distribution, and outcome maturation progress
(none / partial / full), writing `observation_audit.json`.
