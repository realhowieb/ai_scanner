# Market coverage & data health

Run 37. Makes whole-market scan coverage, data failures, and staleness
**observable and measurable**. Answers: *when HSF says it scanned the market, how
much did it actually scan successfully?*

No DT/scanner/ML/UX changes. The only production wiring is additive and
backward-compatible (an opt-in `coverage_sink` and a best-effort report).

## 1. Universe pipeline

| Stage | Where | Notes |
| --- | --- | --- |
| Source | `sp500.txt`, `nasdaq.txt` (+ `sp600.txt`, `otc.txt`) | refreshed weekly |
| Scheduled refresh | `.github/workflows/refresh-universe.yml` | **Sundays 10:30 UTC**; `scripts/refresh_sp500.py` (SPDR holdings), `scripts/refresh_nasdaq.py` (nasdaqtrader) |
| Load | `scheduler.cron_runner._load_universe` | SP500 = full; NASDAQ = first `CRON_NASDAQ_LIMIT` (2000); COMBO = SP500 + NASDAQ[:2000], deduped |
| Normalize | `data.symbols.normalize_ticker` / `sanitize_ticker_list` | class-share separators → dash, collapse, provider-specific |
| Dedupe | `dedupe_preserve_order` | order-preserving |
| SPAC/units/warrants | `is_spac_unit_or_warrant` (drop by default) | in `sanitize_ticker_list` |
| Delisted/junk | `is_probably_delisted` (blocklist) | in `sanitize_ticker_list` |
| Tradability filter | `data.tradability.filter_tradable_tickers` | Alpaca active-tradable set; **fails open** (never empties universe) |
| Exchange/OTC | universe files are exchange-scoped; OTC kept separate | — |
| Scanner consumption | `scan.engine.run_breakout_scan` (+SPY for RS) | fetches OHLCV, runs breakout stage |
| Cache/storage | DB price cache (admin full-universe), `@st.cache_data` (snapshots/avg-vol/EMA), file snapshots | — |

## 2. Coverage funnel

`analytics.coverage.build_coverage_funnel` (pure) reports, per scan:

```
Expected            <universe size before eligibility>
Eligible            <after tradability/normalization>       (excluded = expected-eligible)
Attempted           <symbols price-fetch was attempted for>
Price success       <symbols with usable price data>        (price_failure = attempted-success)
Indicator complete  <optional; None until Run 38 wires it>
Results produced    <breakout candidates, top-N>
Coverage            price_success / eligible
```

Percentages: `eligible_pct`, `price_coverage_pct` (of eligible),
`price_coverage_vs_expected_pct`, `indicator_coverage_pct` (when measured).
Expected/eligible are supplied from the **live** universe — nothing is hardcoded.

**Note on "Results":** results = ranked breakout candidates (top-N passing
gap/volume/liquidity), **not** a coverage measure. Coverage is `price_success /
eligible`.

## 3. Failure taxonomy

Every skipped symbol is classified (`classify_failure`, built on
`data.provider_diagnostics.classify_skip_reason`):

`NO_PRICE_DATA, STALE_DATA, INSUFFICIENT_HISTORY, INVALID_SYMBOL, DELISTED,
UNSUPPORTED_SECURITY, INDICATOR_FAILURE, API_ERROR, RATE_LIMIT, TIMEOUT,
FILTERED_BY_POLICY, UNKNOWN`.

Provider→taxonomy map: rate_limited→RATE_LIMIT, timeout→TIMEOUT,
empty_response/duplicate_data→NO_PRICE_DATA, provider_missing/auth/download_error
→API_ERROR, invalid_data→INDICATOR_FAILURE, other/unknown→UNKNOWN. The funnel
reports counts per reason and the top 5 categories. No symbol is dropped
silently — every skip carries `(symbol, reason)`.

## 4. Data freshness & health

`classify_health` → one of `HEALTHY / DEGRADED / STALE / FAILED`
(precedence FAILED > STALE > DEGRADED > HEALTHY). Inputs it can measure:
universe age, latest price timestamp age, scan start/completion, duration,
market session, and (optionally) model-version age.

Thresholds are **documented defaults, grounded in current behavior**, all
overridable:
- **FAILED** — no eligible universe, or `price_success == 0` (mirrors the
  existing cron "near-empty large scan = throttled" guard and
  `provider_summary.severe`).
- **STALE** — `universe_age > 216h` (weekly Sunday refresh + ~1-day grace) or
  `price_age > 60m` (regular session). Ages left as `None` are not checked.
- **DEGRADED** — coverage below `healthy_floor` (0.95) but above 0; or model
  version very old.
- **HEALTHY** — coverage ≥ 0.95 and fresh.

These are starting points chosen to match observed behavior, not tuned against
outcomes; adjust as real coverage distributions accumulate.

## 5. Coverage report (artifact)

`coverage_report(funnel, health)` returns a bundle:
`{schema: "hsf-coverage-1.0", funnel, health, summary, text}` — machine-readable
JSON + a human-readable `text` block. The scheduled cron writes it to
`artifacts/automation/coverage_<universe>.json` (best-effort), uploaded by the
`scheduled-scans` workflow alongside the existing automation snapshot. It is
generated **even for throttled/partial runs**, so a degraded scan is visible.
The automation export's previously-empty `symbols_processed` / `symbols_skipped`
slots are now filled from the coverage sink.

## 6. Health summary

`health_summary` is a compact structure for a future admin/status surface:
`{state, universe_version, price_coverage_pct, indicator_coverage_pct, eligible,
price_success, stale_symbols, errors, top_failure_categories, duration_sec,
last_scan_completed_at}`. No UI was added this run.

## 7. Performance bottlenecks (ranked)

1. **Duplicate daily-bar fetches (highest impact).** The display enrichment path
   fetches daily bars twice per symbol: `fetch_daily_range_metrics` (150d, for
   ADX/SuperTrend/EWO/ATR/Donchian/Bollinger) and `fetch_ema_crosses` (90d, EMA
   cross). Candidate for a shared daily-frame fetch in Run 38 (not changed here —
   would touch scoring inputs and needs its own validation).
2. **Serial per-universe scans.** SP500, NASDAQ, COMBO run in sequence; COMBO
   overlaps SP500, so overlapping symbols are fetched again. A cross-run daily
   cache exists for admin full-universe runs; broadening it is a Run 38 item.
3. **Fetch concurrency.** `fetch_price_data_parallel` uses `max_workers=4`,
   `chunk_size=70` with per-chunk isolation. Reasonable; raising workers risks
   provider rate limits (already a tracked failure category).
4. **Indicator recomputation.** Indicators recompute each scan (no memoization
   across runs beyond price cache). Acceptable given per-scan freshness needs.

No optimizations were applied this run (mission: only safe, clearly justified
changes; these all touch scoring inputs or provider limits and need their own
validation).

## 8. Failure resilience (verified)

- **Per-symbol isolation:** provider skips are collected as `(symbol, reason)`
  and deduped (`data.prices._dedupe_skipped`); one bad ticker never aborts the
  scan.
- **Batch handling:** chunked fetches are wrapped in try/except; a failed chunk
  is logged and the scan continues.
- **Provider fallback:** parallel fetch → single-shot batch fetch fallback.
- **Rate-limit/timeout:** surfaced as `RATE_LIMIT` / `TIMEOUT` categories, not
  crashes.
- **Partial results:** the min-save guard (`CRON_MIN_SAVE_ROWS`) refuses to
  overwrite a good daily snapshot with a throttled near-empty run and reports the
  run failed.
- **Artifact after partial failure:** the coverage report is generated before the
  save guard, so even a throttled run produces an observable coverage artifact.

## 9. Known limitations

- **Indicator-coverage is not yet measured** (`indicator_complete=None`): the
  breakout stage silently drops symbols with insufficient history. Surfacing that
  count needs breakout-stage instrumentation (Run 38).
- Price/universe **age inputs are optional**; the cron path does not yet compute
  them, so health currently classifies on coverage alone until Run 38 feeds ages.
- Coverage is measured at the **price-fetch** boundary; downstream filter drops
  (gap/volume/liquidity) are intentionally not "failures".

## 10. Operational interpretation

- **HEALTHY** — coverage ≥ 95% of the eligible universe; act on results normally.
- **DEGRADED** — 80–95% (or an old model): results usable but a provider issue is
  eating coverage; check `top_failure_categories` (RATE_LIMIT/TIMEOUT ⇒ provider
  pressure; NO_PRICE_DATA ⇒ symbol/data issues).
- **STALE** — inputs too old; refresh the universe or re-run in-session.
- **FAILED** — zero coverage or no eligible universe; the scan did not represent
  the market — do not trust it.
