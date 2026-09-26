# Run 57 — Point-in-time research metadata

**Persist what the scanner knew. Do not change what the scanner does.**

> **RUN 57 DOES NOT CLAIM ANY TIER OR REGIME HAS PREDICTIVE VALUE.**
> It only records context so future evaluations can test such questions.

## Motivation (from Run 55)

Run 55 could not evaluate:
- **Tiers.** None were persisted: 0 of 913 primary observations carried one.
- **Market regime.** `market_context.market_regime` was never populated.
- **Several scanner features** it had to list as "not persisted". These include
  breakout distance, EMA 9/21 cross and trend, even though the scan computes them.

Run 57 adds what the scheduled scan genuinely holds at the scan timestamp to
**new** observations. The Run 56 forward experiment stays frozen and clean.

## Part 1 — Audit: what is persisted, what is lost

**Where observations are stored:** Neon `hsf_observations(observation_id,
context, timestamp, schema_version, record JSONB)`, with a SQLite fallback. The
record is the canonical `build_observation` dict (`hsf-obs-1.0`). Outcomes live
separately in `hsf_observation_outcomes`.

**Capture path:**
1. `scheduler/cron_runner.run_and_save` runs `scan.engine.run_breakout_scan`.
2. The engine calls `scan.breakout.run_breakout_scan`, which ranks by
   BreakoutScore and returns the top `top_n + near_miss_n` rows.
3. The engine splits that ranked frame:
   - **CANDIDATE** = the first `top_n` rows (the production results);
   - **NEAR_MISS** = the next `near_miss_n` rows (via `research_sink`);
   - **CONTROL** = a seeded sample of evaluated symbols with a price/volume
     snapshot (`select_control_symbols`).
4. `capture_scan_observations` and `_capture_research_cohorts` persist all three
   cohorts through `save_observations_batch`, first-write-wins.

| Group | Already persisted | Lost after the scan (before Run 57) |
|---|---|---|
| Identity | observation_id, symbol, timestamp (hour bucket), scan_timestamp, context, `market_context.scan_id`, research_cohort, selection_reason, session | rank order and the top_n cutoff |
| Signal | `scanners[]`: breakout score, direction (long), derived sub-scanner triggers, is_breakout / pattern (breakout meta) | — |
| Features | market.price and volume; indicators gap_pct, chg_pct, rvol, atr_pct | BreakoutPos20D, Trend20D%, Trend10D%, DollarVol20, RSvsSPY, EMACross; PatternTag outside breakout meta |
| Market context | coverage_health, source (`market_regime` exists but is always NULL) | — |
| Provenance | `versions` (obs schema, hsf_score 1.0, prebreakout / AI model ids, dt_score), breakout scanner `version` | commit SHA, effective scan config (top_n, price and liquidity floors, profile), per-symbol price provider and feed |

## Parts 2–3 — Tier and regime: honestly unavailable

**Tier: `TIER_NOT_EMITTED_BY_SCHEDULED_SCAN`.**
- The scheduled breakout scan emits **no tier**.
- The Strong / Developing / Weak tiers (`analytics/day_trade_intel.
  classify_setup_quality`) run only at UI render time, in Day Trader. They are
  computed from VWAP, ADX, SuperTrend and EWO, which the scheduled scan never
  fetches.
- Persisting a tier would mean *recomputing* one. Run 57 forbids that, so
  `tier_at_observation` is **NULL** for every cohort, with that reason recorded.

**Regime: `REGIME_CAPTURE_UNAVAILABLE`.**
- `ui.opportunities.classify_market_regime` exists (TRENDING BULLISH / RISK-ON /
  MIXED / CHOPPY / RISK-OFF / TRENDING BEARISH / HIGH VOLATILITY).
- It runs only in the Market Brief UI, from SPY/QQQ closes and sector ETFs
  (network calls through the evening-wrap helpers) and snapshot breadth.
- The scheduled scan loads none of those inputs. Capturing a regime would mean
  new inputs, new calls and a new pipeline, which is exactly what Run 57 must not
  invent.
- So `market_regime_at_observation` is **NULL**, with that reason recorded.
- The existing `market_context.market_regime` pass-through is unchanged, and a
  test shows it is still persisted exactly when a caller provides a value.

## Fields added: `record.research_metadata` (`hsf-research-meta-1.0`)

This is a new top-level JSONB key, written only for new captures. It sits
outside `market` and `indicators`, so **observation ids, `data_quality`
completeness and every existing field are byte-identical**.

| Field | Definition | Source (all in memory at the scan timestamp) | NULL when |
|---|---|---|---|
| `schema_version` | `hsf-research-meta-1.0` | constant | never |
| `tier_at_observation` | production tier | none exists in this path | always today |
| `tier_source` | why the tier is NULL | `TIER_NOT_EMITTED_BY_SCHEDULED_SCAN` | never |
| `tier_version` | tier logic version | — | always today |
| `market_regime_at_observation` | production regime | none exists in this path | always today |
| `market_regime_source` | why the regime is NULL | `REGIME_CAPTURE_UNAVAILABLE` | never |
| `market_regime_version` | regime logic version | — | always today |
| `scoring_version` | scoring system | the scanner's own `breakout-1` identifier | never |
| `ranking_rule` | how cohorts are cut | descriptive string (not an invented version number) | never |
| `feature_schema_version` | canonical feature set | `hsf-obs-1.0` | never |
| `scanner_commit_sha` | exact deployed code | `GITHUB_SHA` (or `HSF_COMMIT_SHA`), validated hex | outside CI |
| `universe_name` / `universe_version` | universe label | scan argument (e.g. `US_MARKET`) | — |
| `scan_mode` | how the scan ran | `scheduled` | — |
| `session` | market session | `_resolve_session()` | — |
| `scan_id` | scan run id | scan start timestamp | — |
| `scan_config` | effective parameters | top_n, near_miss_n, min/max price, min_dollar_vol, min_gap, profile, premarket/afterhours/unusual_volume, resolved once and passed unchanged to the scan | — |
| `rank_at_observation` | rank in the scanner's order | CANDIDATE = position 1..top_n; NEAR_MISS = top_n + position | CONTROL (unranked) |
| `price_provider` / `price_feed` | where this symbol's bars came from | the price frame's own `attrs["source"]` / `attrs["feed"]` (`alpaca_multi` / `rescue_single`; `iex` / `sip`) | untagged frames (e.g. the yfinance path) |
| `row_features` | ranked-row columns the canonical record drops | breakout_pos_20d, trend_20d_pct, trend_10d_pct, dollar_vol_20, pattern_tag, rs_vs_spy, ema_cross | CONTROL (never scored) |

**Cohort comparability (Part 8).** CANDIDATE, NEAR_MISS and CONTROL all receive
the same run-level block: tier/regime status, scoring, commit, universe, mode,
session and config, plus the per-symbol provider tag. Controls were filtered
before scoring, so rank and row features do not exist for them; they are left
empty rather than synthesized.

**Only code change on the scanner side.** In `scan/engine.py`, inside the opt-in
`research_sink` block that runs *after* ranking, the price snapshot now also
copies each frame's existing `source` / `feed` tags. The cron resolves the same
scan parameters once and passes them unchanged. Neither change affects `results`.

## Point-in-time guarantee (Part 6)

- Every value is either held by the scan process at the scan timestamp or is a
  constant.
- `analytics/research_metadata.assert_point_in_time` rejects any metadata key
  resembling returns, outcomes, MFE/MAE, horizons, evaluation or maturation
  state, or hindsight labels.
- Row features are copied only from a fixed whitelist of the ranked row's
  columns.
- `classify_market_regime` is provably never called (test).

## Schema evolution and legacy rows (Part 7)

- The change is purely additive JSONB; no DDL or migration is needed.
- **Historical rows are not backfilled.** They simply lack `research_metadata`,
  which means NULL.
- First-write-wins means a later write cannot add metadata to an existing id
  (tested).
- Backfilling tiers from current rules, or a regime from later data, would be
  synthetic history. A historical NULL is the scientifically correct value.

## Run 56 compatibility (Part 10)

- **The forward epoch is NOT reset.** It stays `2026-09-26T07:23:11Z`: the new
  fields are additive metadata, not an experimental change.
- The pre-registered gates A–H are unchanged.
- `forward_evidence_readiness.json` gains an informational
  `metadata_completeness` block per cohort: block, tier, regime,
  scoring-version, commit and provider coverage percentages. It is not a gate,
  because tier and regime are NULL by design and gating on them would block
  forever.
- The anti-peeking guard and the invariance-to-returns test still pass.
- `scripts/audit_research_metadata.py` reads **observations only**, never
  outcomes, and writes `artifacts/research/research_metadata_audit.{json,md}`.
  It reports coverage by cohort and direction, row-feature coverage, and
  provenance value counts. The daily readiness workflow runs it.

## Verification: frozen scanner fixture (Part 12)

The real `scan.engine.run_breakout_scan` was run on fixed synthetic prices
(40 symbols, top_n 10, near_miss_n 5, 8 controls) twice:
- at the pre-Run-57 commit `ebd00a6`, in a temporary worktree;
- with Run 57 applied.

Symbol order, scores (10), near-miss membership (5), control membership (8) and
all 23 observations (every field outside `research_metadata`) were
**identical**.

`tests/test_research_metadata.py` (20 tests) keeps this protected:
- results are byte-identical with and without the research sink;
- observations built with metadata equal observations built without it once the
  block is removed;
- ranks follow the scanner order;
- tier and regime are NULL and the regime classifier is never called;
- metadata attachment and context building are non-fatal;
- hindsight keys are rejected;
- legacy rows stay readable and un-backfilled;
- the Run 56 output stays clean and the epoch is unchanged.

## Remaining gaps, and what this enables

**Still unavailable:**
- tier and regime (see above);
- the Day Trader intraday fields (VWAP, ADX, SuperTrend, EWO), which also drive
  the conflicts and agreement signals;
- RSI;
- PreBreakout and AI-confidence scores on scheduled observations (their model
  versions are recorded, their scores are not);
- a provider tag on the yfinance path;
- the commit SHA outside GitHub Actions.

**Future questions these fields make testable** (in a formal evaluation, not in
monitoring):
- whether rank within the candidate set matters beyond score;
- whether the row features the canonical record used to drop (breakout
  distance, 10- and 20-day trend, EMA cross, RS vs SPY) carry information;
- whether provider or feed differences explain maturation gaps;
- whether results differ across commits or scan configurations.

**Recommended Run 58 scope** (separate, design-reviewed, and applied to new
observations only):
1. Decide whether the scheduled scan should compute the existing
   `classify_market_regime` at scan time from point-in-time inputs (SPY/QQQ
   intraday, sector ETFs, breadth from the evaluated universe). That adds calls,
   so it needs its own review.
2. Decide whether to compute Day Trader tier and conflicts in the scheduled
   path. That needs intraday indicators the scan does not fetch today.
3. Address control maturation parity if Run 56 reports DATA_PIPELINE_BIAS. For
   example, liquidity-stratified controls or the SIP feed for maturation, as a
   separate experiment-design change.

Until the Run 56 gates pass, none of these should change which symbols are
selected.
