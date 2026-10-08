# Point-in-time feature audit (research dataset, feature schema v1)

Scope: every feature the HSF research dataset can serve for a historical
observation, and every feature research would want that HSF never stored.
Code: `analytics/research_schema.py` (schemas, record types),
`analytics/research_dataset.py` (snapshot, temporal join, builder),
`db/research_datasets.py` (bulk reads, version registry). Design notes:
`docs/RESEARCH_DATASET.md`.

Classifications:

- **SAFE_STORED**: frozen at observation time in an immutable row and read back as is.
- **SAFE_RECONSTRUCTABLE**: not stored, but recomputable using only information that existed at `observed_at`. The rule is documented below.
- **UNSAFE_CURRENT_VALUE**: the only available value is today's (mutable) value. Joining it to history leaks.
- **MISSING**: never computed or never persisted for scheduled observations.
- **UNKNOWN**: a value exists, but what it means at the time can't be established.

Historical market data existing is not enough. A value is safe only when its
computation used nothing after `observed_at`.

## The boundary

`observed_at` = `signal_outcomes.fired_at` of the frozen opportunity row. That is
the snapshot time the opportunity was computed from. Features are read from:

1. the opportunity row's frozen payload (`raw_signal`, `indicators`,
   `setup_score`, `prebreakout_prob`), written once at fire time and never updated, and
2. one scheduled-scan record in `hsf_observations`, chosen by the backward
   as-of join below.

Labels (`return_*`, `mfe_5d`, `mae_5d`, `benchmark_return_*`, `excess_return_*`)
live only in `OutcomeRecord`, built from the Outcome Intelligence canonical record.

## Temporal join rules

| Join | Rule | Why |
|---|---|---|
| opportunity -> scan record | same ticker; context `scheduled:*`; `scan_timestamp <= observed_at` AND row `created_at` (write time) `<= observed_at`; `observed_at - scan_timestamp <= 3 h`; latest scan wins; ties prefer `scheduled:us_market`, then the lowest record id | `created_at` proves the values existed before the observation. A scan that started before but finished after `observed_at` is refused. |
| opportunity -> snapshot rank | rows with exactly the same `fired_at` | All were frozen at the same instant. |
| opportunity -> outcomes | by row id (same row) | Labels only. They never enter the snapshot. |
| ticker -> sector / fundamentals / universe list | **not performed** | Only current values exist (UNSAFE_CURRENT_VALUE). |
| ticker -> latest scan / price cache | **not performed** | `price_data_cache` and the daily snapshot row are overwritten. |

Every snapshot exposes its join result as `join.status`: `MATCHED`,
`NO_SCAN_WITHIN_LAG`, `NO_SCAN_RECORD`, or `ONLY_LATER_SCANS` (records exist for
the ticker, but all were written after the observation). Scan features are null
unless the status is `MATCHED`.

## Feature classification (schema v1)

| Feature | Source | Classification | Notes |
|---|---|---|---|
| `hsf_score` | signal_outcomes.raw_signal | SAFE_STORED | |
| `hsf_score_version` | raw_signal.score_version | SAFE_STORED | "1.0" spans formula-input changes (see findings) |
| `hsf_signals_component` | raw_signal.score_components | SAFE_STORED | |
| `hsf_model_component` | raw_signal.score_components | SAFE_STORED | input model changed 2026-10-07/08 |
| `hsf_momentum_component` | raw_signal.score_components | SAFE_STORED | |
| `hsf_fading_penalty` | raw_signal.score_components | SAFE_STORED | |
| `primary_setup` | raw_signal | SAFE_STORED | |
| `hsf_status` | raw_signal | SAFE_STORED | |
| `signals` | indicators.signals | SAFE_STORED | sorted list |
| `n_signals` | indicators | SAFE_STORED | |
| `fading` | indicators | SAFE_STORED | |
| `chg_pct` | indicators | SAFE_STORED | null when not in a gapper/mover list |
| `gap_pct` | indicators | SAFE_STORED | null when not a gapper |
| `breakout_score` | setup_score column | SAFE_STORED | top setups only |
| `prebreakout_prob` | prebreakout_prob column | SAFE_STORED | value as shown; served model version not stored |
| `snapshot_rank` | same-instant frozen rows | SAFE_RECONSTRUCTABLE | see rule below |
| `snapshot_size` | same-instant frozen rows | SAFE_RECONSTRUCTABLE | |
| `price` | hsf_observations.market.price | SAFE_STORED | via join |
| `volume` | hsf_observations.market.volume | SAFE_STORED | via join |
| `rvol_20` | hsf_observations.indicators.rvol | SAFE_STORED | via join |
| `volatility_20d_pct` | hsf_observations.indicators.atr_pct | SAFE_STORED | scanner `Volatility20D%` |
| `scan_gap_pct` | hsf_observations.indicators.gap_pct | SAFE_STORED | via join |
| `scan_chg_pct` | hsf_observations.indicators.chg_pct | SAFE_STORED | via join |
| `scanner_breakout_score` | hsf_observations.scanners | SAFE_STORED | via join |
| `is_breakout` | hsf_observations.scanners meta | SAFE_STORED | via join |
| `trend_10d_pct` | research_metadata.row_features | SAFE_STORED | Run 57+ rows only |
| `trend_20d_pct` | research_metadata.row_features | SAFE_STORED | Run 57+ rows only |
| `breakout_pos_20d` | research_metadata.row_features | SAFE_STORED | Run 57+ rows only |
| `dollar_vol_20` | research_metadata.row_features | SAFE_STORED | Run 57+ rows only |
| `rs_vs_spy` | research_metadata.row_features | SAFE_STORED | Run 57+ rows only |
| `ema_cross` | research_metadata.row_features | SAFE_STORED | tag only, no EMA values |
| `pattern_tag` | research_metadata.row_features | SAFE_STORED | Run 57+ rows only |
| `scanner_rank` | research_metadata.rank_at_observation | SAFE_STORED | Run 57+ rows only |

### SAFE_RECONSTRUCTABLE rules

`snapshot_rank` / `snapshot_size`

- Source data: `signal_outcomes` rows with `source='opportunity'` and the same `fired_at`.
- Timestamp rule: only rows frozen at exactly that instant. They are all written by the same freeze call.
- Calculation: sort by `hsf_score` descending, then ticker, then id. Rank 1 = highest. Size = row count.
- Lookback: none.
- Session behavior: none. The freeze happens after a scan completes, whatever the session.
- Caveat: the UI's own list order was not stored, so ties may differ from what a user saw. Only the top 5 per snapshot are ever frozen, so rank is within the top 5 only.

## Features not in schema v1

| Feature | Classification | Reason |
|---|---|---|
| rsi_14 | MISSING | Scheduled scans never compute RSI. Day Trader computes it at render time and doesn't store it. It is reconstructable from raw daily bars, but none are persisted. |
| ema9 / ema21 values | MISSING | Only the cross tag is stored. |
| adx / vwap / supertrend / ewo / day_trader_tier | MISSING | Computed only when the Day Trader page renders. |
| prev_close / vol_avg_20 / high_20 / spark_10d | SAFE_STORED | In per-scan `runs.results_json` only, pruned after 90 days. Readable via the per-scan run (never the daily snapshot row). Not read by v1. |
| prebreakout_model_version (served) | UNKNOWN | The served bundle version isn't frozen. `hsf_observations.versions.prebreakout_model` is the code constant, which can differ from the served champion. |
| sector / industry / market_cap / fundamentals | UNSAFE_CURRENT_VALUE | Only current metadata exists. |
| market_regime / breadth | MISSING | Computed in the Market Brief UI only (`REGIME_CAPTURE_UNAVAILABLE`). |
| universe membership list | UNSAFE_CURRENT_VALUE | Universe files are refreshed in place. Only the universe name is frozen. |

### If these are ever reconstructed (not done in this run)

RSI/EMA/ATR-style features would be SAFE_RECONSTRUCTABLE only with:

- Source data: raw (`adjustment=raw`) daily bars from a persisted store, not a live provider call per build.
- Timestamp rule: only bars whose session closed at or before `observed_at`. Use the ET session close from `analytics.market_calendar`, early closes included. The training pipeline's current "+1 day at UTC midnight" rule is conservative: a post-close scan doesn't see that day's bar. That is safe, but it doesn't match what the live scan saw.
- Calculation and lookback: exactly the scanner's (e.g. EMA span 9/21 with `adjust=False`, 14-period Wilder RSI), with at least 3x the span of warm-up bars.
- Session behavior: an intraday observation must not use the current day's partial bar unless it was the bar the scan actually fetched. Unadjusted bars break across splits, so flag split dates rather than adjust them with later-known factors.

## Market-context readiness (P1)

| Context | Classification | Notes |
|---|---|---|
| SPY trend | NOT_AVAILABLE | Not stored per observation. SAFE_TO_DERIVE from raw SPY daily bars with the close rule above (needs persisted bars). |
| QQQ trend | NOT_AVAILABLE | Same as SPY. |
| market volatility (VIX or SPY realized vol) | NOT_AVAILABLE | SPY realized vol is SAFE_TO_DERIVE from bars. VIX is not fetched anywhere. |
| breadth | SAFE_TO_DERIVE | From the same scan's persisted rows (`runs` per-scan results or CONTROL/NEAR_MISS cohort records). This is the candidate set, not the whole market. |
| sector performance | NOT_AVAILABLE | Sector ETF closes are only loaded by the Market Brief UI. |
| sector relative strength | UNSAFE | Needs a ticker->sector map, and only the current map exists. |
| stock vs SPY RS | AVAILABLE_STORED | `rs_vs_spy` (Run 57+ scan records). |
| stock vs QQQ RS | NOT_AVAILABLE | SAFE_TO_DERIVE from bars. |
| stock vs sector RS | UNSAFE | Current sector map only. |

## Findings that matter for ML v3

1. **Daily snapshot rows are mutable.** `save_daily_snapshot` overwrites `results_json` and resets `created_at` during the day. Training (`ml_prebreakout.load_run_history`) reads snapshot and per-scan rows together, so the last scan of each day is counted twice. Only per-scan rows are safe for research, and they are pruned after 90 days.
2. **Score version doesn't mark input changes.** `hsf_score_version` stays "1.0" across PreBreakout entering scheduled scans (2026-10-07, PR #14) and the train/serve skew fix (2026-10-08, PR #26, `_live_feature_frame`). Before that fix, live PreBreakout inputs lacked timestamp, daily bars and SPY/QQQ, and `prebreakout_prob` and `hsf_model_component` were near-flat. Split walk-forward folds at those dates. Don't treat same-version rows as one distribution.
3. **No served model version on history.** `model_version` is null for every observation. It is not inferred from today's deployment.
4. **Opportunity provenance gap.** `fired_at` is the newest snapshot of any universe, and no run id is stored. The scan join recovers the scan instead (by time, with lag and write-time checks).
5. **Training imputes zeros.** `build_ml_dataset` does `X.fillna(0.0)`. The research layer preserves nulls. Imputation belongs in a versioned preprocessing step.
6. **Overlap.** The cron freezes a ticker in many snapshots per day, and all of them share one entry close and outcome window. Rows expose `overlap.group` (ticker + entry trading day), `group_size`, `first_in_group` and `overlapping_entry_days` for purge/embargo. Nothing is removed.
7. **Selection.** Only the top 5 opportunities per snapshot are frozen, so the dataset is conditional on being top 5. It is not a market sample.

Live coverage for these features is in the PR report (measured by `scripts/research_dataset.py` through the Diagnostics workflow).
