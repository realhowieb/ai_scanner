# ML v3 data quality gate

Dataset `hsf-ml-2026-10-08-v1` · fingerprint `sha256:527bbd62ae511db111679f77949be0acd9bbed06ca1ac0a83196c1ad1e7886cc` · audit slice `sha256:5afdb61396903bd9f70b1a3bfec69d0c415d3f6dd9c898e7a238c0382503d065`

## Verdict: **FAIL**

Blocks definitive validation (diagnostic analysis continues where safe):

- only 134 matured, certified signal-days at the primary 5-day horizon (definitive validation needs >= 300)

Warnings:

- benchmark missing on 128 of 337 matured 5-day rows
- served model version unknown on all 1041 observations
- 629 observations have unavailable outcomes (mostly the 2026-09-12/13 burst); excluded by maturity, not deleted
- 656 repeated same-day snapshots collapsed to one signal-day each
- overlapping label windows across entry days: handled by purge + embargo

## Checks

| CHECK | RESULT |
|---|---|
| observations / signal-days | 1041 / 385 |
| duplicate observations (same ticker + instant) | 0 |
| same-ticker same-time duplicates | 0 |
| same-ticker same-day repeated snapshots | 656 |
| rows in multi-row ticker-day groups | 894 |
| rows with overlapping label windows | 387 |
| missing timestamps | 0 |
| invalid (future) timestamps | 0 |
| missing prices (no matched scan) | 808 |
| zero/negative prices | 0 |
| missing scores | 0 |
| scores outside 0-100 | 0 |
| 5d outcome pending / unavailable / invalid | 75 / 629 / 0 |
| matured 5d observations | 337 |
| missing benchmark among matured 5d | 128 |
| missing MFE/MAE among matured 5d | 0 |
| unknown model versions | 1041 |
| unknown scoring versions | 0 |
| scoring versions | {"1.0": 1041} |
| temporal join status | {"MATCHED": 233, "NO_SCAN_RECORD": 95, "NO_SCAN_WITHIN_LAG": 10, "ONLY_LATER_SCANS": 703} |
| look-ahead join violations (scan written after observation) | 0 |
| as-of join lag seconds (min/p25/median/p75/p95/max) | 126 / 174 / 206 / 245 / 1813 / 2248 |

## Label balance (primary label: return > 0, matured + certified signal-days)

| HORIZON | N | POSITIVE | RATE |
|---|---|---|---|
| 1d | 134 | 74 | 55.2% |
| 3d | 134 | 71 | 53.0% |
| 5d | 134 | 74 | 55.2% |

## Concentration (matured 5d signal-days)

| MEASURE | VALUE |
|---|---|
| rows | 134 |
| unique tickers | 69 |
| top ticker share | 6.0% |
| top-10 ticker share | 33.6% |
| entry days | 13 |
| largest single-day share | 14.2% |
| setups | {"Breakout": 108, "Gapper": 16, "Golden Cross": 10} |

## Feature nullness (observation unit)

| FEATURE | NULL RATE |
|---|---|
| hsf_score | 0.0% |
| hsf_score_version | 0.0% |
| hsf_signals_component | 0.0% |
| hsf_model_component | 0.0% |
| hsf_momentum_component | 0.0% |
| hsf_fading_penalty | 0.0% |
| primary_setup | 0.0% |
| hsf_status | 0.0% |
| signals | 30.8% |
| n_signals | 0.0% |
| fading | 0.0% |
| chg_pct | 18.4% |
| gap_pct | 27.9% |
| breakout_score | 3.0% |
| prebreakout_prob | 43.1% |
| snapshot_rank | 0.0% |
| snapshot_size | 0.0% |
| price | 77.6% |
| volume | 77.6% |
| rvol_20 | 77.6% |
| volatility_20d_pct | 77.6% |
| scan_gap_pct | 77.6% |
| scan_chg_pct | 77.6% |
| scanner_breakout_score | 77.6% |
| is_breakout | 77.6% |
| trend_10d_pct | 87.7% |
| trend_20d_pct | 87.7% |
| breakout_pos_20d | 87.7% |
| dollar_vol_20 | 87.7% |
| rs_vs_spy | 87.7% |
| ema_cross | 99.2% |
| pattern_tag | 87.7% |
| scanner_rank | 87.7% |

## Observations per UTC day

| DAY | OBSERVATIONS |
|---|---|
| 2026-09-12 | 593 |
| 2026-09-13 | 15 |
| 2026-09-14 | 27 |
| 2026-09-15 | 25 |
| 2026-09-16 | 30 |
| 2026-09-17 | 35 |
| 2026-09-18 | 35 |
| 2026-09-21 | 20 |
| 2026-09-22 | 45 |
| 2026-09-23 | 40 |
| 2026-09-24 | 20 |
| 2026-09-25 | 15 |
| 2026-09-27 | 5 |
| 2026-09-28 | 27 |
| 2026-09-29 | 16 |
| 2026-09-30 | 18 |
| 2026-10-01 | 15 |
| 2026-10-02 | 10 |
| 2026-10-05 | 10 |
| 2026-10-06 | 24 |
| 2026-10-07 | 10 |
| 2026-10-08 | 6 |
