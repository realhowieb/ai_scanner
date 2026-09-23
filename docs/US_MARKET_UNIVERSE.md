# US_MARKET universe (Run 44)

`US_MARKET` is HSF's canonical full U.S.-listed tradable-equity universe for
scheduled scans, replacing the old `SP500 / NASDAQ[:2000] / COMBO` subsets that
scanned overlapping partial lists three times and capped NASDAQ at 2,000.

**This run changes market coverage only — no scanner/scoring/model/DT/Alert-
Priority logic changed.**

## What US_MARKET means

The set of currently active, tradable U.S.-listed equities that our market-data/
broker infrastructure (Alpaca) can process, normalized and deduplicated.

## Data source

`data/us_market_universe.py` → Alpaca `/v2/assets?status=active&asset_class=us_equity`
(the same authoritative endpoint `data.tradability` already uses). The network
call is isolated in `_fetch_us_equity_assets` so it is fully mockable/testable
offline.

## Included

- Active (`status=active`), tradable (`tradable=True`) `us_equity` assets on U.S.
  exchanges: NASDAQ, NYSE, NYSE American (AMEX), NYSEARCA/ARCA, BATS, IEX.
- Normalized (`data.symbols.normalize_ticker`), deduplicated, deterministically
  sorted.

## Excluded (with counted reasons)

- `inactive` — not active.
- `non_tradable` — `tradable=False`.
- `unsupported_asset_type` — non-equity class, or SPAC **units / warrants /
  rights** (via `is_spac_unit_or_warrant`) and delisted-pattern symbols.
- `malformed_symbol` — empty / non-alphanumeric / >10 chars.
- `wrong_exchange` — non-U.S. / unsupported venue.
- `duplicate` — repeated symbol.
- OTC, crypto, options, and non-U.S. assets are excluded by the endpoint query
  (`asset_class=us_equity`) itself.

**Limitation:** Alpaca does not expose a fine-grained security sub-type, so
**preferred shares** are only excluded when the symbol carries a recognizable
warrant/unit/rights marker. Some preferreds may remain; documented, not hidden.

## Refresh, caching & fallback (fail-safe)

`build_us_market_universe()` returns an explicit `source`:
- **live** — freshly fetched, filtered, and ≥ `_MIN_PLAUSIBLE` (1,000) symbols;
  the result is also written to the last-known-good cache
  (`artifacts/universe/us_market.json`).
- **cached** — the provider failed or returned an implausibly small set; the
  last-known-good cache is used and the result is marked `is_fallback=True` with
  a `fallback_reason`. **A cached universe can never masquerade as live.**
- **none** — neither live nor cache available; `symbols=[]`. The scheduled scan
  **aborts** (`run_and_save` raises) rather than substituting a partial universe.

HSF never reports full coverage when only a partial/legacy universe was used.

## Scheduled behavior

- Default scheduled universe is now **US_MARKET** (`_configured_universes`,
  `CRON_UNIVERSES` unset). `CRON_UNIVERSES=US_MARKET` also works explicitly.
- Legacy `SP500`, `NASDAQ`, `COMBO` remain fully supported (static files;
  `CRON_NASDAQ_LIMIT` still applies **only** to NASDAQ/COMBO). `US_MARKET` never
  honors `CRON_NASDAQ_LIMIT` and has no 2,000 cap.
- `CRON_TOP_N` limits only the **returned candidate count**, never the number of
  symbols evaluated (regression-tested).

## Coverage metrics & telemetry

Each `US_MARKET` run records (in the coverage artifact
`artifacts/automation/coverage_us_market.json`, alongside the Run 37 funnel):
`universe_name, universe_source, universe_generated_at, universe_symbol_count,
is_fallback, provider_assets, exclusions{…}, eligible_symbol_count,
attempted_symbol_count, successfully_priced_count, skipped_symbol_count,
result_count, universe_load_sec, scan_duration_sec, symbols_per_sec,
coverage_percentage`. Health (HEALTHY/DEGRADED/FAILED) comes from the existing
`analytics.coverage.classify_health` — a scan that prices only a fraction of the
eligible universe is **not** HEALTHY.

## Batching / scale

The existing `scan.engine.run_breakout_scan` price path already fetches in
deterministic chunks (`fetch_price_data_parallel`, `max_workers=4`,
`chunk_size=70`) with per-chunk failure isolation and per-symbol skip capture — a
single bad ticker never kills the scan. Run 44 does not redesign the engine; it
feeds it the larger universe unchanged. Concurrency stays bounded to avoid
provider rate limits.

## Observation capture (current behavior — a limitation)

Run 38A capture persists **only final scanner-result rows (candidates)**, not
every evaluated symbol (see `PRODUCTION_OBSERVATION_CAPTURE.md`). Scanning the
full market therefore does **not** proportionally increase observation writes —
capture volume tracks candidate count, not universe size. Capturing per-symbol
observations for the whole market would be a large, separately-measured change
and is intentionally **not** done here.

## Expected scale

A live U.S. equity universe is typically ~6,000–8,000 tradable symbols after
filtering. At the current price-fetch throughput this is minutes per scheduled
slot; monitor `symbols_per_sec` / `scan_duration_sec` in the coverage artifact to
confirm operational practicality.

## Troubleshooting

- **Scan aborted, source=none:** provider down and no cache — check Alpaca creds/
  status; the first successful run seeds the cache.
- **source=cached (FALLBACK):** provider was unavailable; coverage is from the
  last-known-good list — treat as degraded provenance.
- **Low coverage / DEGRADED-FAILED:** check the funnel `top_failure_categories`
  (RATE_LIMIT/TIMEOUT ⇒ provider pressure; NO_PRICE_DATA ⇒ symbol/data issues).
- **Universe looks small:** ensure `US_MARKET` (not a legacy universe) and that
  `CRON_NASDAQ_LIMIT` is not being misapplied (it does not affect US_MARKET).

## Asset-type handling (explicit)

| Type | Handling |
| --- | --- |
| Common stocks | **Included** |
| ETFs | **Included** (tradable `us_equity`; HSF treats them as scannable equities) |
| ADRs | **Included** (listed `us_equity` on a U.S. exchange) |
| Preferred shares | Mostly **included**; only excluded when the symbol carries a warrant/unit/rights marker (Alpaca exposes no fine sub-type — documented limitation) |
| Warrants / rights / units | **Excluded** (`is_spac_unit_or_warrant`) |
| OTC securities | **Excluded** by the endpoint (`asset_class=us_equity` returns exchange-listed only) |
| Test symbols | **Excluded** (`is_probably_delisted` blocklist / malformed filter) |
| Crypto | **Excluded** (not `us_equity`) |
| Options | **Excluded** (not `us_equity`) |

## Batching / CRON_BATCH_SIZE (Run 44)

The scheduled/headless price-fetch batch size is configurable via
`CRON_BATCH_SIZE` (env). Resolution precedence (`scan.engine.resolve_chunk_size`,
pure + tested): interactive Streamlit `price_fetch_chunk_size` → `CRON_BATCH_SIZE`
→ `PRICE_FETCH_CHUNK_SIZE` default, always clamped to
`[PRICE_FETCH_CHUNK_MIN, PRICE_FETCH_CHUNK_MAX]` (a bad value can never explode
memory or hammer the provider; invalid input falls back to the default). The
default behavior is unchanged when the env is unset. Batches are processed
deterministically with per-chunk failure isolation and per-symbol skip capture,
so one bad ticker or chunk never kills the whole scan; concurrency stays bounded
(`fetch_price_data_parallel`, `max_workers=4`) to respect provider rate limits.

## Live validation (2026-09-22, run 35758795541)

Non-mocked scheduled US_MARKET run on `dev`:
`universe_source=live` · provider_assets **14,357** → eligible **11,873**
(excluded: 878 non-tradable, 1,297 warrants/units/unsupported, 309 wrong-exchange)
→ eligible-after-tradability **11,827** → attempted **11,826** → priced **11,631**
(195 failures, all `FILTERED_BY_POLICY` = deliberate yfinance-fallback skips) →
**100 candidates** · coverage **98.3% · HEALTHY** · scan 125s · ~94.6 symbols/sec.
