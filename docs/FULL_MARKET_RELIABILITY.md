# Full-market scan reliability (Run 45)

Operational hardening for the Run 44 US_MARKET scheduled scanner. **No scoring,
signals, ML, ranking, or intelligence changed** — Run 45 makes full-market
operation measurable, repeatable, failure-aware, and safe. It reuses the Run 37
coverage/health system (`analytics.coverage`) rather than adding a second one.

## Run 44 baseline (reference, not a hardcoded expectation)

Live scheduled run 2026-09-22: provider assets 14,357 → eligible 11,827 →
attempted 11,826 → priced 11,631 (195 skipped, all intentional policy) → 100
candidates · coverage 98.3% · runtime 125s · ~94.6 symbols/sec · HEALTHY · source
LIVE. Normal listing/delisting/provider variation is expected between runs.

## Operational architecture

`analytics/scan_reliability.py` (pure, tested) + wiring in
`scheduler/cron_runner.py`:
- **Performance record** (`build_performance_record`) — one structured
  `hsf-scan-perf-1.0` record per scheduled scan (counts, timings, throughput,
  event counts, snapshot status). Unmeasurable stage timings are honestly `null`.
- **Health-gated snapshot safety** (`snapshot_decision`) — the critical gate.
- **Failure taxonomy** (`derive_event_counts`) — reuses `coverage.classify_failure`.
- **Repeatability** (`compare_to_recent`) — anomaly flags vs recent HEALTHY runs.
- **Human summary** (`render_run_summary`).
- **Overlap protection** (cron lockfile).

## Telemetry (per run)

Attached to the coverage artifact (`artifacts/automation/coverage_us_market.json`,
`performance` block) and appended to a rolling history
(`artifacts/automation/perf_history.jsonl`, last 60 runs): run_id, started/
completed, market_session, universe/source, provider_asset_count, eligible/
attempted/priced/skipped, candidate_count, coverage_percentage, coverage_health,
operational_state, timings (universe_load / total; per-stage where measurable,
else null), symbols_per_second, batch_size, batches_* (null — engine does not yet
expose batch counters), rate_limit/timeout/provider_error/policy_filtered/
provider_trouble events, failure_by_reason, snapshot_promoted,
snapshot_suppression_reason.

## Health classification

Reuses `coverage.classify_health` (FAILED > STALE > DEGRADED > HEALTHY) and folds
it to an operational triad (`to_operational_state`): STALE → DEGRADED.
- **HEALTHY** — enough of the intended universe processed to represent a
  full-market scan (coverage ≥ 0.95, fresh).
- **DEGRADED** — meaningful partial coverage / provider trouble (0.80–0.95, or
  stale).
- **FAILED** — cannot be trusted market-wide (zero eligible or zero priced).

## Snapshot safety (the critical guarantee)

`snapshot_decision(health, coverage_pct)`: **only a HEALTHY scan with coverage ≥
0.90 is promoted as the canonical daily snapshot.** DEGRADED / FAILED / STALE scans
are **not** promoted — their artifact and diagnostics are still written
(`snapshot_promoted=false` + `snapshot_suppression_reason`), so a bad scan can
never overwrite a known-good snapshot and failure evidence is never hidden. This
sits on top of the existing near-empty-result guard (`CRON_MIN_SAVE_ROWS`).

## Failure taxonomy

Per-symbol skips classify via the Run 37 taxonomy (`coverage.classify_failure`):
`RATE_LIMIT, TIMEOUT, PROVIDER_ERROR (API_ERROR), NO_PRICE_DATA,
FILTERED_BY_POLICY, INDICATOR_FAILURE, UNKNOWN`. Operational-level reasons add
`MALFORMED_RESPONSE, UNIVERSE_PROVIDER_FAILURE, OVERLAPPING_RUN`. The key
distinction: `policy_filtered_events` (intentional, e.g. the 195 yfinance-fallback
skips) are excluded from `provider_trouble_events`, so "195 intentionally skipped"
is never confused with "195 throttled".

## Overlap protection

A single-host lockfile (`artifacts/cron_scan.lock`): if a prior scheduled scan is
still running, the next invocation exits with `skip_reason=OVERLAPPING_RUN` (no
second concurrent whole-market sweep). A stale lock older than 30 min is reclaimed
so a crashed run never blocks forever. Lock IO is fail-open — it never prevents a
scan. Appropriate for the current single-runner GitHub Actions deployment (no
distributed infra).

## Batching / retry

`CRON_BATCH_SIZE` (Run 44) tunes the price-fetch batch size, resolved by
`scan.engine.resolve_chunk_size` (session > env > default) and clamped to
`[MIN, MAX]`. Batches process deterministically with per-chunk failure isolation
and per-symbol skip capture (one bad ticker/chunk never kills the scan);
concurrency stays bounded (`max_workers=4`). Do **not** change the production
default from synthetic tests alone — only from live evidence (Run 45 policy).

## Scheduled sessions

Unchanged from Run 44: 4 weekday slots (14:35 / 17:00 / 20:30 / 21:10 UTC).
`_resolve_session` routes premarket/postmarket slots to the session scan and
regular slots to the US_MARKET sweep + snapshot. Weekend/holiday skip
(`_skip_reason`, Alpaca calendar) and manual `force` override are preserved.

## Performance expectations

~11–12k attempted symbols in ~2 minutes at ~90+ symbols/sec is the working
baseline. Reliability outranks speed — do not trade provider stability, coverage,
or failure isolation for lower runtime.

## Troubleshooting / how to tell if a scan was healthy

Read `coverage_us_market.json`: `performance.operational_state == "HEALTHY"` **and**
`performance.snapshot_promoted == true` means it was trusted as a full-market
snapshot. If `snapshot_promoted == false`, read `snapshot_suppression_reason`.
Check `provider_trouble_events` (RATE_LIMIT/TIMEOUT spikes ⇒ throttling) vs
`policy_filtered_events` (intentional). `perf_history.jsonl` +
`compare_to_recent` flag universe-size / coverage / runtime / throughput /
provider-error anomalies against recent healthy runs.
