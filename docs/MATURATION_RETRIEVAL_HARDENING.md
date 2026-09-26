# Outcome maturation: market-data retrieval hardening

Makes `scripts/mature_observations.py` turn the observation backlog into outcome
labels without being dominated by Alpaca HTTP 429. **No scoring, tier, ranking,
PreBreakout, or research-gate change.** The outcome math, horizon semantics,
schema, and first-write-wins persistence are unchanged.

## Diagnosis (Mature Observations #12, run 36202017693, 2026-09-25 23:41 UTC)

- 400 symbols processed → **one single-symbol request each** (`/v2/stocks/{sym}/bars`,
  window = earliest anchor date → now), sent back-to-back in ~60 s. That is ~400
  req/min against Alpaca's ~200 req/min basic quota.
- **141 of 400 symbols (≈35%) got HTTP 429.** There was no retry:
  `fetch_minute_bars` logged the error, returned `[]`, and the worker recorded
  every ready horizon as `PRICE_DATA_UNAVAILABLE`. That is roughly 600 of the 911.
  Throttling was indistinguishable from missing data.
- The remaining failures are real sparsity or no data: illiquid ETFs on IEX
  (BJAN, IAPR, BSJR, …) and preferreds captured before the Sep 23 US_MARKET
  exclusion (ATH.PRD, GAB.PRK, GNL.PRD, …).
- Observations per processed symbol ≈ 1.13 (451/400). Per-symbol reuse already
  existed, so batching is where the savings come from.

## Changes

| # | Change | Where |
|---|---|---|
| 1 | Multi-symbol retrieval: one paginated `/v2/stocks/bars?symbols=…` series per 100-symbol batch (10k bars/page shared), oldest-anchor-first so batch members share a start | `data/price_alpaca.fetch_minute_bars_multi`, `mature_observations._retrieve_bars` |
| 2 | Per-run cache: each symbol is fetched once per run, and every one of its observations reuses those bars (`cache_hits` / `cache_misses`) | worker pass 2a/2b |
| 3 | `_alpaca_get`: bounded retries (5) on 429/5xx/timeout/connection. For 429 it honours `Retry-After` (seconds or HTTP-date) and then `X-RateLimit-Reset`, capped at 60 s plus jitter. Other retries use exponential backoff with jitter (`[d/2, d]`, d = min(30, 2ⁿ)). 0.35 s pacing between requests keeps a run under ~170 req/min | `data/price_alpaca.py` |
| 4 | Circuit breaker: if 429 persists after retries, that batch and all later batches become `RATE_LIMITED` for this run (no hammering), and the next run retries them | `_retrieve_bars` |
| 5 | Ineligible symbols (preferred-share suffix, malformed) are dropped **before** the per-run cap and never fetched. Only the symbol-level US_MARKET rules are used; the SPAC U/W heuristic is excluded because it also matches MU, SNOW, NOW. ETFs are **not** excluded because no US_MARKET rule excludes them | `data/us_market_universe.symbol_exclusion_reason` |
| 6 | Bounded window: `[earliest anchor in batch (minute floor), min(now, latest anchor + 4 days)]`. The old window ran from the anchor date to now. Horizons count bars, so 4 days leaves sparse names time to reach 61 bars across a weekend or holiday. Outcomes are identical to the legacy fetch unless the 61st bar comes more than 4 days after the anchor | `FORWARD_WINDOW` |
| 7 | A mid-pagination failure raises. Partial bar series are never used | `fetch_minute_bars_multi` |

The per-run cap stays at **400 symbols**. The legacy injectable
`fetch_bars(symbol, start_date)` path still works, and a test asserts that it
produces outcomes identical to the batched path.

## Report fields (`maturation_report.json`, schema `hsf-maturation-1.2`)

`alpaca_requests`, `alpaca_429_count`, `alpaca_retry_count`,
`unique_symbols_requested`, `cache_hits`, `cache_misses`, `symbols_processed`,
`symbols_deferred`, `outcomes_matured`, `price_data_failures` (horizons),
`insufficient_future_bars` (horizons). The report also carries these fields:

- `price_data_unavailable_symbols`: the provider answered with no bars (true missing data).
- `rate_limited_symbols`: symbols blocked by persistent 429 (throttling).
- `provider_error_symbols`
- `ineligible {symbols, observations, reasons}`
- `requests_per_processed_symbol`
- `fetch_mode`

## Before / after estimate (400-symbol batch)

| | Before (#12, measured) | After (estimated) |
|---|---|---|
| Alpaca requests | ~400 (≈1.0 per symbol) | **~15–60** (4 batch series; bars ÷ 10k per page) ≈ 0.04–0.15 per symbol |
| Request rate | ~400/min, unpaced | ≤ ~170/min, paced |
| 429 rate | ~35% of requests, 0 retried | **≈0** expected; any 429 is retried with backoff |
| 429 → `PRICE_DATA_UNAVAILABLE` | ~600 horizons | 0 (reported as `RATE_LIMITED` if retries are exhausted) |
| Outcomes matured / run | 18 | **~150–400**. Run #11 (0 × 429) matured 384 from 511 eligible. The ~35% of the batch that was throttled gets real data |
| Runs to attempt every ready symbol once | 5 nominal, but ~35% of each run was wasted and requeued at the head of the queue | **5** (ceil(1613/400)). At the ~3–4 effective runs/day measured in Run 52, that is 1–2 trading days |

The request estimate assumes a ~2-day backlog window and 300–1,500 IEX bars per
symbol. Verify it against `alpaca_requests` and `requests_per_processed_symbol`
on the first live run.

## Remaining limiter (not changed here)

Symbols with sparse or no data stay "ready" forever (flagged in Run 52), and
oldest-first ordering keeps them at the head of each run's 400-symbol budget.
Requests now cost roughly bars instead of symbols. Once one live run confirms a
low `requests_per_processed_symbol` and short runtime, raising `--max-symbols` to
~2000 would clear the ready set in **one run** for about 50–150 requests.
A separate follow-up is to retire observations whose bounded window has fully
elapsed, because retrying them cannot change the result.
