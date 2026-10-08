# ML v3 point-in-time and leakage audit

Features come only from the canonical FeatureSnapshot (research schema v1). Nothing is rebuilt from current scanner state. Classes: SAFE_STORED (frozen at observation), SAFE_ASOF_JOIN (backward join to a scan written before the observation), REVIEW (point-in-time but with a provenance gap), LEAKAGE (would carry post-observation information), MISSING (never stored).

| FEATURE | SOURCE | SCHEMA | COVERAGE | POINT-IN-TIME | LEAKAGE RISK | NOTES |
|---|---|---|---|---|---|---|
| hsf_score | signal_outcomes(source='opportunity') | 1 | 100.0% | SAFE_STORED | LOW | value shown at fire time (frozen); formula version 1.0 throughout |
| hsf_score_version | signal_outcomes(source='opportunity') | 1 | 100.0% | SAFE_STORED | LOW |  |
| hsf_signals_component | signal_outcomes(source='opportunity') | 1 | 100.0% | SAFE_STORED | LOW |  |
| hsf_model_component | signal_outcomes(source='opportunity') | 1 | 100.0% | REVIEW | MEDIUM | max(BreakoutScore, PreBreakout %): inherits the PreBreakout version gap |
| hsf_momentum_component | signal_outcomes(source='opportunity') | 1 | 100.0% | SAFE_STORED | LOW |  |
| hsf_fading_penalty | signal_outcomes(source='opportunity') | 1 | 100.0% | SAFE_STORED | LOW |  |
| primary_setup | signal_outcomes(source='opportunity') | 1 | 100.0% | SAFE_STORED | LOW |  |
| hsf_status | signal_outcomes(source='opportunity') | 1 | 100.0% | SAFE_STORED | LOW |  |
| signals | signal_outcomes(source='opportunity') | 1 | 69.2% | SAFE_STORED | LOW |  |
| n_signals | signal_outcomes(source='opportunity') | 1 | 100.0% | SAFE_STORED | LOW |  |
| fading | signal_outcomes(source='opportunity') | 1 | 100.0% | SAFE_STORED | LOW |  |
| chg_pct | signal_outcomes(source='opportunity') | 1 | 81.6% | SAFE_STORED | LOW |  |
| gap_pct | signal_outcomes(source='opportunity') | 1 | 72.1% | SAFE_STORED | LOW |  |
| breakout_score | signal_outcomes(source='opportunity') | 1 | 97.0% | SAFE_STORED | LOW |  |
| prebreakout_prob | signal_outcomes(source='opportunity') | 1 | 56.9% | REVIEW | MEDIUM | stored value as shown is point-in-time, but the served model version was never recorded and the model's inputs changed 2026-10-07/08 (train/serve skew fix) |
| snapshot_rank | derived from the same frozen snapshot | 1 | 100.0% | SAFE_STORED | LOW | reconstructed from rows frozen at the same instant only |
| snapshot_size | derived from the same frozen snapshot | 1 | 100.0% | SAFE_STORED | LOW | reconstructed from rows frozen at the same instant only |
| price | hsf_observations(context='scheduled:*') | 1 | 22.4% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| volume | hsf_observations(context='scheduled:*') | 1 | 22.4% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| rvol_20 | hsf_observations(context='scheduled:*') | 1 | 22.4% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| volatility_20d_pct | hsf_observations(context='scheduled:*') | 1 | 22.4% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| scan_gap_pct | hsf_observations(context='scheduled:*') | 1 | 22.4% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| scan_chg_pct | hsf_observations(context='scheduled:*') | 1 | 22.4% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| scanner_breakout_score | hsf_observations(context='scheduled:*') | 1 | 22.4% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| is_breakout | hsf_observations(context='scheduled:*') | 1 | 22.4% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| trend_10d_pct | hsf_observations.research_metadata (Run 57+) | 1 | 12.3% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| trend_20d_pct | hsf_observations.research_metadata (Run 57+) | 1 | 12.3% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| breakout_pos_20d | hsf_observations.research_metadata (Run 57+) | 1 | 12.3% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| dollar_vol_20 | hsf_observations.research_metadata (Run 57+) | 1 | 12.3% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| rs_vs_spy | hsf_observations.research_metadata (Run 57+) | 1 | 12.3% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| ema_cross | hsf_observations.research_metadata (Run 57+) | 1 | 0.8% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| pattern_tag | hsf_observations.research_metadata (Run 57+) | 1 | 12.3% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| scanner_rank | hsf_observations.research_metadata (Run 57+) | 1 | 12.3% | SAFE_ASOF_JOIN | LOW | backward as-of join: scan written (created_at) <= observed_at, lag <= 3 h |
| rsi_14 | - | - | 0.0% | MISSING | N/A | Scheduled scans never compute RSI; Day Trader computes it at render time and doesn't store it. Reconstructable from raw daily bars, but no historical bars are persisted. |
| ema9 / ema21 values | - | - | 0.0% | MISSING | N/A | Only the EMA cross tag is stored. Values are reconstructable from raw bars, which aren't persisted. |
| adx / vwap / supertrend / ewo / day_trader_tier | - | - | 0.0% | MISSING | N/A | Computed only when the Day Trader page renders; never persisted for scheduled scans. |
| prev_close / vol_avg_20 / high_20 / spark_10d | - | - | 0.0% | MISSING | N/A | Stored in per-scan runs.results_json for 90 days only (then pruned). Not read by schema v1. |
| prebreakout_model_version (served) | - | - | 0.0% | MISSING | N/A | The served model version lives in the loaded bundle and is not frozen on observations. hsf_observations.versions holds the code constant, which can differ from the served champion. |
| sector / industry / market_cap / fundamentals | - | - | 0.0% | LEAKAGE | HIGH (if joined) | Only current metadata exists. Joining it to history would leak today's values. |
| market_regime / breadth | - | - | 0.0% | MISSING | N/A | Regime is computed only in the Market Brief UI and never frozen (REGIME_CAPTURE_UNAVAILABLE). |
| universe membership list | - | - | 0.0% | LEAKAGE | HIGH (if joined) | Universe files are overwritten by refresh jobs; only the universe name is frozen. |

## Automated checks

- Scan joins: 233 matched, 0 written after the observation (must be 0).
- Outcome-like names in the feature vector: none.
- Single-feature in-sample AUC >= 0.85 (leak symptom): none.

### Single-feature in-sample AUC (primary horizon, primary label)

| FEATURE | N | AUC |
|---|---|---|
| price | 60 | 0.6702 |
| volume | 60 | 0.3676 |
| scan_gap_pct | 60 | 0.4095 |
| gap_pct | 50 | 0.4103 |
| rvol_20 | 60 | 0.4106 |
| scan_chg_pct | 60 | 0.4129 |
| volatility_20d_pct | 60 | 0.5814 |
| hsf_signals_component | 134 | 0.45 |
| n_signals | 134 | 0.45 |
| hsf_score | 134 | 0.4508 |
| chg_pct | 86 | 0.4536 |
| is_breakout | 60 | 0.4581 |
| snapshot_size | 134 | 0.5253 |
| breakout_score | 114 | 0.4809 |
| scanner_breakout_score | 60 | 0.4842 |
| hsf_model_component | 134 | 0.4845 |
| snapshot_rank | 134 | 0.4845 |
| hsf_momentum_component | 134 | 0.4876 |
| hsf_fading_penalty | 134 | 0.5068 |
| fading | 134 | 0.5068 |

## Explicit inspection list

| ITEM | FINDING |
|---|---|
| future price / volume / returns | Not in the snapshot. Labels live only in OutcomeRecord; FeatureSnapshot refuses outcome-like keys (tests/test_ml_v3_audit.py). |
| MFE / MAE / outcome / maturity fields | Labels only; never encoded as features. |
| future benchmark values | Benchmark returns are labels; regime labels use SPY closes strictly before the entry day. |
| post-observation setup labels | primary_setup/status/signals are the frozen fire-time values. |
| current HSF score / rank | hsf_score and snapshot_rank are the frozen values at fire time; scanner_rank is the scan's own rank. |
| current sector metadata / universe / fundamentals | Not joined (UNSAFE_CURRENT_VALUE). |
| normalization across future samples | Medians, scalers and category vocabularies are fitted on the training fold only. |
| target-derived variables | None in research schema v1. The legacy July recipes used one (see ml_v3_walk_forward_summary.md, legacy section). |
