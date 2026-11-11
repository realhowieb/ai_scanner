# Run 51 — full-market production health audit

Investigation of scheduled US_MARKET scan health after Runs 45–50 + the
preferred-share fix. **No scoring/threshold/ranking/ML change.** One low-risk
observability fix was applied (provider-timeout skips were mislabeled as policy);
everything else is measurement.

## Verdict: **CONDITIONAL PASS**

Full-market scanning is **reliable and self-protecting**: coverage sits 94–98%,
health classification and snapshot-gating work correctly (a real DEGRADED run was
suppressed live), the provider showed no rate-limiting, and the preferred-share
exclusion measurably improved coverage. The **conditional** is because (1) provider
read-timeouts on chunks were being **mislabeled as intentional "policy" skips** —
hiding provider instability in telemetry (now fixed), and (2) daytime coverage
varies with provider latency and should be monitored.

## Trend table (live production runs, 2026-09-23, UTC)

| Run | time | eligible | priced | coverage | policy | prov-trouble | dur(s) | sym/s | snapshot | health |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 35815769365 | 03:48 | 11,826 | 11,633 | 98.37% | 192 | 0 | 137 | 86 | ✅ | HEALTHY |
| 35817916450 | 04:21 | 11,826 | 11,633 | 98.37% | 192 | 0 | 145 | 81 | ✅ | HEALTHY |
| 35819912828 | 04:50 | 11,826 | 11,633 | 98.37% | 192 | 0 | 123 | 96 | ✅ | HEALTHY |
| 35868097202 | 13:35 | 11,844 | 11,487 | 96.99% | 356 | 0 | 183 | 65 | ✅ | HEALTHY |
| **35889771285** | **16:36** | **11,844** | **11,187** | **94.45%** | **656** | 0 | 182 | 65 | **❌ suppressed** | **DEGRADED** |
| 35903364523 | 18:34 | 11,844 | 11,485 | 96.97% | 358 | 0 | 194 | 61 | ✅ | HEALTHY |
| 35912400576 | 19:55 | 11,844 | 11,487 | 96.99% | 356 | 0 | 192 | 62 | ✅ | HEALTHY |
| **35915097802** | **20:3x (post-preferred-fix)** | **11,488** | **11,281** | **98.20%** | **206** | 0 | **115** | 98 | ✅ | HEALTHY |

## Before/after the preferred-share exclusion (commit a357829)

| | eligible | coverage | policy skips | runtime |
|---|---:|---:|---:|---:|
| Before (19:55) | 11,844 | 96.99% | 356 | 192s |
| **After (post-fix)** | **11,488** (−360 preferreds) | **98.20%** | **206** | **115s** |

Excluding **360 preferred shares** (`.PR<letter>`) removed illiquid non-target
symbols that couldn't be priced and dragged on runtime: **coverage +1.2pts, policy
skips −150, runtime −40%.** Clear improvement; still HEALTHY, snapshot promoted.

## Provider-failure taxonomy (the key finding)

Telemetry showed `provider_trouble_events = 0` on **every** run — including the
DEGRADED 94.45% run with 656 "policy" skips. Investigation of
`data/prices.py::_download_batch` found the cause: **when an Alpaca chunk request
fails (e.g. the observed `data.alpaca.markets Read timed out`), all ~50–70 symbols
in that chunk are labeled `skipped_yf_fallback` → `FILTERED_BY_POLICY`** (because
the yfinance fallback is disabled by policy), regardless of *why* Alpaca failed.
So genuine provider timeouts were counted as intentional policy skips, and the
coverage dips (98%→94.5%) — which correlate with slow runs (61 vs 96 sym/s) —
looked like "policy" rather than provider latency.

**Fix applied (observability only):** the batch-failure path now records the true
cause (`alpaca_timeout:<type>` / `alpaca_batch_error:<type>`), which the existing
coverage taxonomy classifies as **TIMEOUT / PROVIDER_ERROR** (→ `provider_trouble`),
not policy. Symbols Alpaca *successfully* returns-nothing-for stay `skipped_yf_fallback`
(genuine no-data). No symbol selection, scoring, or health threshold changed —
only the label. Regression-tested.

## Symbols/types responsible for repeated failures

- **Preferred shares** (`.PR<letter>`, e.g. PSA.PRF, PRIF.PRD) — illiquid, not on
  Alpaca's IEX feed → unpriceable + timeout-prone. **Now excluded** (−360).
- **Thin/micro-cap real equities** not on the IEX feed — the residual ~200 policy
  skips in HEALTHY runs are mostly these (genuine no-data, `skipped_yf_fallback`).
- **Chunk read-timeouts** during high-latency slots — previously hidden in policy;
  now surfaced as TIMEOUT.

## Remaining universe contamination (post-fix exclusions)

`non_tradable 878 · unsupported_asset_type 1298 (warrants/units/rights + delisted/
non-equity) · preferred_share 360 · wrong_exchange 308 (foreign venues) · duplicate 0`.
Warrants/units/rights, preferreds, and foreign structures are **excluded**. No
material non-target contamination remains; the residual skips are legitimate
thin-liquidity equities Alpaca's feed doesn't carry, not junk in the universe.

## Snapshot safety (verified live)

The 16:36 DEGRADED run (94.45% < 95% floor) had **`snapshot_promoted = false`** —
the health-gated promotion (Run 45) correctly refused to overwrite the known-good
snapshot, while retaining the coverage artifact + diagnostics. **A failed/degraded
scan cannot corrupt or replace the last known-good snapshot.**

## Top operational risks

1. **Provider single-point dependency + daytime latency** — coverage varies
   94.5–98% with Alpaca load; occasional chunk timeouts. (Now observable.)
2. **Thin-liquidity residue** — ~200 real symbols per scan lack IEX data; a floor,
   not a bug.
3. **No cross-run durable perf history** on the ephemeral runner (Run 50 P2).

## Recommended fixes (ranked)

- **P0:** none.
- **P1:** (applied) surface provider-timeout skips distinctly from policy skips so
  DEGRADED causes are diagnosable. Also: monitor `symbols_per_second` — sustained
  < ~70 signals provider degradation; consider a bounded retry/backoff on
  chunk-timeout before recording the skip (future, measured).
- **P2:** persist `perf_history.jsonl` to Neon for cross-run trend/anomaly
  detection; consider dropping additional non-target classes (rights `.R`, when-issued
  `.WI`) if they appear in exclusions telemetry.

## Test / lint

Full suite **1,380 passed** / 9 skipped; ruff `E9,F,I` clean.
