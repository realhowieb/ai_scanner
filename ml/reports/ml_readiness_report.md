# ML v4 data readiness report (production snapshot)

Production run 2026-10-09 01:32 UTC, Diagnostics `script=ml_readiness.py` on `claude/project-thread-wl8nou`
(read-only; job 113626290471). Re-score dry run: Diagnostics `script=rescore_premature_outcomes.py` (job 113626297211).
Reproduce with the same Diagnostics run; the full generated report (every table below plus per-day collection,
distributions and the JSON) prints in the job log. The System Health run writes the same report as
`ml_readiness_monitor.md` to its step summary twice per trading day.

Unit: signal-day (first observation per ticker per UTC day), 5-day primary horizon, the ML v3 audit's unit.

## Status: **NOT_READY** → recommendation **ML_V4_NOT_READY**

## Gates

| GATE | KIND | VALUE | NEEDS | STATUS |
|---|---|---|---|---|
| matured_signal_days | ACCUMULATING | 134 | >= 300 | FAIL |
| entry_days | ACCUMULATING | 13 | >= 30 | FAIL |
| entry_weeks | ACCUMULATING | 3 | >= 6 | FAIL |
| max_entry_day_share | ACCUMULATING | 14.2% | <= 10% | FAIL |
| unique_symbols | ACCUMULATING | 69 | >= 100 | FAIL |
| top10_symbol_share | ACCUMULATING | 33.6% | <= 30% | FAIL |
| class_balance | ACCUMULATING | 44.8% minority | >= 20% | PASS |
| walk_forward_folds | ACCUMULATING | 0 | >= 3 | FAIL |
| final_holdout | ACCUMULATING | not valid | valid | FAIL |
| collection_active | STRUCTURAL | 1 trading day since last | <= 2 | PASS |
| recent_outcome_coverage | STRUCTURAL | 34.9% | >= 90% | FAIL |
| maturation_backlog | STRUCTURAL | 0 overdue | 0 | PASS |
| benchmark_coverage | STRUCTURAL | 76.1% (102 of 134) | >= 90% | FAIL |
| point_in_time_integrity | STRUCTURAL | 0 violations | 0 | PASS |
| scan_feature_join_rate | STRUCTURAL | 22.4% | >= 80% | FAIL |
| served_model_version | STRUCTURAL | 0% | >= 95% | FAIL |
| ai_confidence_frozen | STRUCTURAL | 0% | >= 95% | FAIL |

The reason for every threshold is in `analytics/ml_readiness.py` (`GATES`). Walk-forward folds use the
pre-registered ML v4 spec (>= 60 train, >= 30 validation, purge = label window, 1-day embargo); the ML v3 audit's
smaller 30/20 sizing found 1 fold on the same data.

## Dataset

| METRIC | VALUE |
|---|---|
| observations | 1,041 (385 signal-days, 261 tickers) |
| observation dates / entry days | 22 / 19 |
| matured observations (5d) | 337 (134 signal-days, 32.4% of observations) |
| immature (waiting for window) | 75 observations, 48 signal-days |
| failed maturation | 629 observations |
| effective independent samples (no same-ticker window overlap) | 78 |
| range | 2026-09-12 06:54Z to 2026-10-08 13:40Z |
| labels, 5d (return > 0) | 74 positive / 60 negative (55.2%) |
| labels, 5d beat SPY | 51.0% of 102 with a benchmark |
| 10/15/20-bar outcomes | not collected (never derived) |

## Outcome maturation integrity

| CATEGORY | OBSERVATIONS | SIGNAL-DAYS |
|---|---|---|
| TRAINING_ELIGIBLE | 337 | 134 |
| WAITING_FOR_WINDOW | 75 | 48 |
| PREMATURE_LABEL_WRITE | 613 | 199 |
| DATA_PROVIDER_FAILURE / MISSING_PRICE_DATA (EA) | 16 | 4 |
| every other failure category | 0 | 0 |

**Root cause of the 613:** the outcome cron picks rows 8 calendar days after `fired_at`. A weekend snapshot enters
on Monday, so its 5-day window closes on the following Monday's close, but the 8-day age is reached on Sunday; the
Monday-morning cron found 4 of 5 sessions, `score_signal` returned None, and the cron wrote an all-empty label, which
is final. All 613 are scorable from today's bars (dry run). 608 of them enter on 2026-09-14 (the 2026-09-12 weekend
freeze burst), 5 on 2026-09-28. EA (16 rows) still can't be scored.

Fixed in this PR: the cron leaves a row pending while its window is open (writes an empty label only once the window
is complete and still unscorable, or after 21 days) and drops today's unfinished daily bar before scoring.

## Collection

- Expected: 6 slots x top 5 = 30 observations per trading day (upper bound). Actual: 22.5 per trading day overall,
  15.6 over the last 10 trading days; 9.7 and 8.3 signal-days.
- Snapshots by slot (19 trading days): 13:35 UTC 16, 16:35 UTC 17, 19:35 UTC 17, and **0** at 12:35, 20:35 and 21:35
  (the pre-market and after-close slots never freeze opportunities). 35 snapshots were off-schedule (manual runs).
- Duplicates: 0 exact; 662 same-ticker same-day repeats (63.6% of observations); 39.6% of a day's tickers were also
  there the previous trading day. More snapshots of the same top 5 would mostly add repeats.
- 613 observations were frozen on non-trading days; 46 were frozen more than an hour after their snapshot.
- Market regime (derived from SPY prior closes): 375 signal-days sideways/low-vol, 10 bearish/low-vol.

## Point-in-time and feature integrity

- 0 look-ahead joins, 0 labels at/before the observation, 0 labels from an unfinished bar, 0 rows recorded before
  their timestamp.
- Scan-feature join 22.4%: 703 rows only have scan records written later. The nearest same-ticker scan record is a
  median 11.5 days after the observation (minimum 2.5 hours), so this is not a write-lag race: the scheduled capture
  does not record the opportunity tickers at that scan.
- Served PreBreakout model version and AI Confidence are frozen on 0 rows.

## Growth projection

At the measured rate (8.3 signal-days per trading day, 34.9% maturation success) the accumulating targets need about
58 trading days. With the maturation fix (success near 100%), the same rate reaches 300 matured signal-days, 30 entry
days and 100 tickers in about 21 to 23 trading days. Structural gates are not projected: each needs a pipeline change.
