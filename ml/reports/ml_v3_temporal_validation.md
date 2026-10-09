# ML v3 temporal validation design

## Algorithm

1. Unit: one row per (ticker, UTC observation day), the day's first frozen observation (Outcome Intelligence `signal_day`). Population: matured AND certified at horizon h.
2. Each row gets `entry_day` (first trading day on/after the UTC fire date) and `window_end[h]` (the trading day h bars later: the last bar its label reads).
3. Validation blocks are consecutive entry days holding >= 20 rows and >= 5 of each class. Blocks are chosen from actual coverage, not calendar quarters.
4. Training rows for a block starting on day V: entry_day < V, and window_end[h] < V (purge), and window_end[h] < V minus 1 trading day(s) (embargo). Training needs >= 30 rows with both classes; otherwise the block start moves forward a day.
5. Expanding window: every later fold trains on all earlier, purged history.
6. Final holdout: the most recent block of >= 100 rows over >= 5 days is reserved first, only if at least 3 walk-forward folds remain without it. Otherwise no holdout is created.
7. Score inputs (HSF Score, stored model %) get probabilities from a train-fold logistic map.

Deterministic tests: `tests/test_ml_v3_audit.py` (chronological order, purge, embargo, overlapping outcomes, no training label reading a validation-period bar, fold reproducibility).

## 1-day horizon: 134 rows, positive rate 55.2%

Final holdout: not created: reserving 108 rows from 2026-09-17 leaves 0 walk-forward folds (need >= 3)

| FOLD | TRAIN START | TRAIN END | VAL START | VAL END | TRAIN N | VAL N | VAL POS RATE | PURGED | EMBARGOED | VAL TICKERS | VAL SETUPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-14 | 2026-09-17 | 2026-09-22 | 2026-09-23 | 37 | 31 | 51.6% | 9 | 14 | 24 | {"Breakout": 25, "Golden Cross": 5, "Gapper": 1} |
| 2 | 2026-09-14 | 2026-09-21 | 2026-09-24 | 2026-09-28 | 60 | 26 | 53.8% | 12 | 19 | 18 | {"Breakout": 24, "Gapper": 2} |

## 3-day horizon: 134 rows, positive rate 53.0%

Final holdout: not created: reserving 108 rows from 2026-09-17 leaves 0 walk-forward folds (need >= 3)

| FOLD | TRAIN START | TRAIN END | VAL START | VAL END | TRAIN N | VAL N | VAL POS RATE | PURGED | EMBARGOED | VAL TICKERS | VAL SETUPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-14 | 2026-09-17 | 2026-09-24 | 2026-09-28 | 37 | 26 | 46.2% | 40 | 14 | 18 | {"Breakout": 24, "Gapper": 2} |

## 5-day horizon: 134 rows, positive rate 55.2%

Final holdout: not created: reserving 108 rows from 2026-09-17 leaves 0 walk-forward folds (need >= 3)

| FOLD | TRAIN START | TRAIN END | VAL START | VAL END | TRAIN N | VAL N | VAL POS RATE | PURGED | EMBARGOED | VAL TICKERS | VAL SETUPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-14 | 2026-09-17 | 2026-09-28 | 2026-09-30 | 37 | 25 | 60.0% | 58 | 14 | 17 | {"Breakout": 22, "Gapper": 2, "Golden Cross": 1} |

Market conditions per entry day (point-in-time SPY regime) are in ml_v3_market_context.md.

