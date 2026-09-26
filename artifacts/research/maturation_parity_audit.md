# Maturation Parity & Missingness Audit

Generated 2026-09-26T08:13:57.542686+00:00 · schema `hsf-maturation-parity-1.0` · READ-ONLY · completeness only (no outcome values)

## Root cause: **MARKET_DATA_AVAILABILITY_EFFECT** (confidence HIGH)

- Material mechanisms at +60m: ['MARKET_DATA_AVAILABILITY_EFFECT']
- Components (control − candidate missing rate, pp): {'HISTORICAL_SCHEDULER_STARVATION': -15.71, 'MARKET_DATA_AVAILABILITY_EFFECT': 58.38, 'COHORT_COMPOSITION_EFFECT': 1.06, 'RETIREMENT_POLICY_EFFECT': 0.0, 'PIPELINE_BUG': 0.0, 'UNATTRIBUTED': 0.0, 'EXPECTED_TEMPORAL_MISSINGNESS': 3.57}
- Control/candidate median capture dollar-volume ratio: 0.0011
- Basis: traced

Historical scheduler starvation: **True** · current 2,000 cap binding: **False** · fix: **AUDIT_ONLY** (current 2,000-symbol cap does not bind and no same-anchor sharing mismatch exists; historical starvation affected pre-epoch data only) · forward parity status: **NO_FORWARD_DATA**

## Run 55 snapshot

Observations 3713 · cohorts {'CANDIDATE': 1400, 'NEAR_MISS': 713, 'CONTROL': 1600}

Run 55 primary population (regular-session anchors):

| Horizon | Candidate | Near-miss | Control | Gap (pp) | Class |
|---|---|---|---|---|---|
| +5m | 54.75% | 45.68% | 14.62% | 40.13 | CRITICAL |
| +15m | 54.08% | 44.21% | 11.85% | 42.23 | CRITICAL |
| +30m | 49.83% | 40.78% | 9.92% | 39.91 | CRITICAL |
| +60m | 47.58% | 37.85% | 8.62% | 38.96 | CRITICAL |

All sessions:

| Horizon | Cohort | Eligible | Matured | Unmatured | Maturation % | Projected % (after backlog) |
|---|---|---|---|---|---|---|
| +5m | CANDIDATE | 1400 | 857 | 543 | 61.21% | — |
| +5m | NEAR_MISS | 713 | 365 | 348 | 51.19% | — |
| +5m | CONTROL | 1600 | 301 | 1299 | 18.81% | — |
| +15m | CANDIDATE | 1400 | 849 | 551 | 60.64% | — |
| +15m | NEAR_MISS | 713 | 356 | 357 | 49.93% | — |
| +15m | CONTROL | 1600 | 242 | 1358 | 15.12% | — |
| +30m | CANDIDATE | 1400 | 798 | 602 | 57.0% | — |
| +30m | NEAR_MISS | 713 | 335 | 378 | 46.98% | — |
| +30m | CONTROL | 1600 | 205 | 1395 | 12.81% | — |
| +60m | CANDIDATE | 1400 | 768 | 632 | 54.86% | — |
| +60m | NEAR_MISS | 713 | 315 | 398 | 44.18% | — |
| +60m | CONTROL | 1600 | 178 | 1422 | 11.12% | — |

| Horizon | Candidate | Near-miss | Control | Gap (pp) | Class | Projected gap | Projected class |
|---|---|---|---|---|---|---|---|
| +5m | 61.21% | 51.19% | 18.81% | 42.4 | CRITICAL | — | — |
| +15m | 60.64% | 49.93% | 15.12% | 45.52 | CRITICAL | — | — |
| +30m | 57.0% | 46.98% | 12.81% | 44.19 | CRITICAL | — | — |
| +60m | 54.86% | 44.18% | 11.12% | 43.74 | CRITICAL | — | — |

Missingness at +60m (count · % of eligible · % of missing):

| Reason | CANDIDATE | NEAR_MISS | CONTROL |
|---|---|---|---|
| PENDING_BACKLOG | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% |
| SCHEDULER_DEFERRED | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% |
| RATE_LIMITED | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% |
| PRICE_DATA_UNAVAILABLE | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% |
| INSUFFICIENT_FUTURE_BARS | 89 · 6.36% · 14.08% | 50 · 7.01% · 12.56% | 121 · 7.56% · 8.51% |
| RETIRED_WINDOW_CLOSED | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% |
| INELIGIBLE_SYMBOL | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 17 · 1.06% · 1.2% |
| INVALID_ANCHOR | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% |
| PIPELINE_ERROR | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% |
| UNKNOWN | 543 · 38.79% · 85.92% | 348 · 48.81% · 87.44% | 1284 · 80.25% · 90.3% |

Timing (ET):

| Cohort | P25 | Median | P75 | +60m crosses close | PRE (<09:30) | 09:30-10:30 | 10:30-12:00 | 12:00-14:00 | 14:00-15:00 | 15:00-16:00 | POST (>=16:00) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CANDIDATE | 09:36 | 14:34 | 15:36 | 21.43% | 100 | 300 | 0 | 300 | 300 | 300 | 100 |
| NEAR_MISS | 09:36 | 14:34 | 15:36 | 22.86% | 50 | 150 | 0 | 150 | 150 | 163 | 50 |
| CONTROL | 09:36 | 14:34 | 15:36 | 25.0% | 200 | 300 | 0 | 300 | 300 | 400 | 100 |

Symbol composition:

| Cohort | Unique symbols | Obs/symbol | Median capture $-vol (M) | Median price | Never-matured symbols % | Exclusion | Format |
|---|---|---|---|---|---|---|---|
| CANDIDATE | 261 | 5.36 | 12.776 | 103.35 | 27.97% | {'eligible': 1400} | {'PLAIN': 1400} |
| NEAR_MISS | 277 | 2.57 | 11.311 | 99.21 | 38.27% | {'eligible': 713} | {'PLAIN': 713} |
| CONTROL | 1496 | 1.07 | 0.014 | 25.44 | 81.08% | {'eligible': 1583, 'preferred_share': 17} | {'PLAIN': 1572, 'FIVE_LETTER_SPECIAL_SUFFIX': 11, 'CLASS_OR_SUFFIX': 17} |

Symbol overlap: {'candidate_control': 24, 'candidate_near_miss': 170, 'near_miss_control': 35}

Symbol sharing mismatches: {'DIFFERENT_SCAN_RUN': 4478}

Scheduler cap allocation (share of each cohort's ready work reached by the worker's own ordering):

- cap 400: ready symbols 1610, deferred 1210, reach gap 28.75pp — CANDIDATE 0.04%, NEAR_MISS 0.0%, CONTROL 28.75%
- cap 2000: ready symbols 1610, deferred 0, reach gap 0.0pp — CANDIDATE 100.0%, NEAR_MISS 100.0%, CONTROL 100.0%

Historical replay (cap 400, 10 real maturation runs): never-attempted share of still-unmatured eligible work:

- +5m: CANDIDATE 543/543 (100.0%), NEAR_MISS 340/348 (97.7%), CONTROL 969/1299 (74.6%)
- +15m: CANDIDATE 550/551 (99.82%), NEAR_MISS 349/357 (97.76%), CONTROL 970/1358 (71.43%)
- +30m: CANDIDATE 600/602 (99.67%), NEAR_MISS 369/378 (97.62%), CONTROL 974/1395 (69.82%)
- +60m: CANDIDATE 628/632 (99.37%), NEAR_MISS 389/398 (97.74%), CONTROL 979/1422 (68.85%)

Retirement: active=False · CANDIDATE 0 obs (0.0%), NEAR_MISS 0 obs (0.0%), CONTROL 0 obs (0.0%)

Static scheduler audit:

- **loader query**: ORDER BY timestamp DESC LIMIT 5000 (hour-bucketed timestamp; all cohorts of a scan share it) — cohort-neutral: yes, except arbitrary tie order in the one boundary scan; risk: ~1238 research observations/day -> the 5,000 window covers ~4.0 days, shorter than the 6-day retirement window; older unmatured rows stop being retried
- **grouping**: observations grouped by symbol regardless of cohort; one bar series per symbol serves every cohort's observations — cohort-neutral: yes (shared bars); risk: none when every ready symbol is processed
- **ordering + cap**: symbols ordered by earliest PENDING anchor, then the first max_symbols processed — cohort-neutral: only when the cap does not bind; risk: when the cap binds, persistently failing old symbols stay at the head and a recurring symbol's newer observations ride along with its oldest pending one; CANDIDATE symbols recur far more than CONTROL symbols
- **per-run cap**: 400 symbols until b81daaf (2026-09-26), 2,000 after — cohort-neutral: capacity is counted in symbols, and CONTROL contributes ~1 symbol per observation; risk: CONTROL-heavy symbol counts make the cap bind on CONTROL first
- **horizons**: all ready horizons of an observation are processed together — cohort-neutral: yes; risk: none
- **retirement**: time-based (anchor + 6 days), cohort-agnostic code path — cohort-neutral: yes by construction; risk: inherits any earlier starvation
- **batching / caching / 429**: 100-symbol batches in the same oldest-first order; persistent 429 trips a circuit breaker for the remaining batches — cohort-neutral: yes, but the batches cut off by a breaker are the later ones in the ordering; risk: same as cap
- **failure retry**: failed symbols are retried every run until retirement — cohort-neutral: yes; risk: no-data symbols consume capacity each run

## Current historical

Observations 3713 · cohorts {'CANDIDATE': 1400, 'NEAR_MISS': 713, 'CONTROL': 1600}

Run 55 primary population (regular-session anchors):

| Horizon | Candidate | Near-miss | Control | Gap (pp) | Class |
|---|---|---|---|---|---|
| +5m | 54.75% | 45.68% | 14.62% | 40.13 | CRITICAL |
| +15m | 54.08% | 44.21% | 11.85% | 42.23 | CRITICAL |
| +30m | 49.83% | 40.78% | 9.92% | 39.91 | CRITICAL |
| +60m | 47.58% | 37.85% | 8.62% | 38.96 | CRITICAL |

All sessions:

| Horizon | Cohort | Eligible | Matured | Unmatured | Maturation % | Projected % (after backlog) |
|---|---|---|---|---|---|---|
| +5m | CANDIDATE | 1400 | 857 | 543 | 61.21% | 99.93% |
| +5m | NEAR_MISS | 713 | 365 | 348 | 51.19% | 100.0% |
| +5m | CONTROL | 1600 | 301 | 1299 | 18.81% | 50.94% |
| +15m | CANDIDATE | 1400 | 849 | 551 | 60.64% | 98.79% |
| +15m | NEAR_MISS | 713 | 356 | 357 | 49.93% | 98.32% |
| +15m | CONTROL | 1600 | 242 | 1358 | 15.12% | 41.25% |
| +30m | CANDIDATE | 1400 | 798 | 602 | 57.0% | 91.71% |
| +30m | NEAR_MISS | 713 | 335 | 378 | 46.98% | 91.02% |
| +30m | CONTROL | 1600 | 205 | 1395 | 12.81% | 34.0% |
| +60m | CANDIDATE | 1400 | 768 | 632 | 54.86% | 87.57% |
| +60m | NEAR_MISS | 713 | 315 | 398 | 44.18% | 86.4% |
| +60m | CONTROL | 1600 | 178 | 1422 | 11.12% | 28.12% |

| Horizon | Candidate | Near-miss | Control | Gap (pp) | Class | Projected gap | Projected class |
|---|---|---|---|---|---|---|---|
| +5m | 61.21% | 51.19% | 18.81% | 42.4 | CRITICAL | 49.06 | CRITICAL |
| +15m | 60.64% | 49.93% | 15.12% | 45.52 | CRITICAL | 57.54 | CRITICAL |
| +30m | 57.0% | 46.98% | 12.81% | 44.19 | CRITICAL | 57.71 | CRITICAL |
| +60m | 54.86% | 44.18% | 11.12% | 43.74 | CRITICAL | 59.45 | CRITICAL |

Missingness at +60m (count · % of eligible · % of missing):

| Reason | CANDIDATE | NEAR_MISS | CONTROL |
|---|---|---|---|
| PENDING_BACKLOG | 458 · 32.71% · 72.47% | 301 · 42.22% · 75.63% | 272 · 17.0% · 19.13% |
| SCHEDULER_DEFERRED | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% |
| RATE_LIMITED | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% |
| PRICE_DATA_UNAVAILABLE | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 403 · 25.19% · 28.34% |
| INSUFFICIENT_FUTURE_BARS | 174 · 12.43% · 27.53% | 97 · 13.6% · 24.37% | 730 · 45.62% · 51.34% |
| RETIRED_WINDOW_CLOSED | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% |
| INELIGIBLE_SYMBOL | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 17 · 1.06% · 1.2% |
| INVALID_ANCHOR | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% |
| PIPELINE_ERROR | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% |
| UNKNOWN | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% | 0 · 0.0% · 0.0% |

Timing (ET):

| Cohort | P25 | Median | P75 | +60m crosses close | PRE (<09:30) | 09:30-10:30 | 10:30-12:00 | 12:00-14:00 | 14:00-15:00 | 15:00-16:00 | POST (>=16:00) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CANDIDATE | 09:36 | 14:34 | 15:36 | 21.43% | 100 | 300 | 0 | 300 | 300 | 300 | 100 |
| NEAR_MISS | 09:36 | 14:34 | 15:36 | 22.86% | 50 | 150 | 0 | 150 | 150 | 163 | 50 |
| CONTROL | 09:36 | 14:34 | 15:36 | 25.0% | 200 | 300 | 0 | 300 | 300 | 400 | 100 |

Symbol composition:

| Cohort | Unique symbols | Obs/symbol | Median capture $-vol (M) | Median price | Never-matured symbols % | Exclusion | Format |
|---|---|---|---|---|---|---|---|
| CANDIDATE | 261 | 5.36 | 12.776 | 103.35 | 27.97% | {'eligible': 1400} | {'PLAIN': 1400} |
| NEAR_MISS | 277 | 2.57 | 11.311 | 99.21 | 38.27% | {'eligible': 713} | {'PLAIN': 713} |
| CONTROL | 1496 | 1.07 | 0.014 | 25.44 | 81.08% | {'eligible': 1583, 'preferred_share': 17} | {'PLAIN': 1572, 'FIVE_LETTER_SPECIAL_SUFFIX': 11, 'CLASS_OR_SUFFIX': 17} |

Symbol overlap: {'candidate_control': 24, 'candidate_near_miss': 170, 'near_miss_control': 35}

Symbol sharing mismatches: {'DIFFERENT_SCAN_RUN': 4478}

Scheduler cap allocation (share of each cohort's ready work reached by the worker's own ordering):

- cap 400: ready symbols 1610, deferred 1210, reach gap 28.75pp — CANDIDATE 0.04%, NEAR_MISS 0.0%, CONTROL 28.75%
- cap 2000: ready symbols 1610, deferred 0, reach gap 0.0pp — CANDIDATE 100.0%, NEAR_MISS 100.0%, CONTROL 100.0%

Historical replay (cap 400, 10 real maturation runs): never-attempted share of still-unmatured eligible work:

- +5m: CANDIDATE 543/543 (100.0%), NEAR_MISS 340/348 (97.7%), CONTROL 969/1299 (74.6%)
- +15m: CANDIDATE 550/551 (99.82%), NEAR_MISS 349/357 (97.76%), CONTROL 970/1358 (71.43%)
- +30m: CANDIDATE 600/602 (99.67%), NEAR_MISS 369/378 (97.62%), CONTROL 974/1395 (69.82%)
- +60m: CANDIDATE 628/632 (99.37%), NEAR_MISS 389/398 (97.74%), CONTROL 979/1422 (68.85%)

Retirement: active=False · CANDIDATE 0 obs (0.0%), NEAR_MISS 0 obs (0.0%), CONTROL 0 obs (0.0%)

Static scheduler audit:

- **loader query**: ORDER BY timestamp DESC LIMIT 5000 (hour-bucketed timestamp; all cohorts of a scan share it) — cohort-neutral: yes, except arbitrary tie order in the one boundary scan; risk: ~1238 research observations/day -> the 5,000 window covers ~4.0 days, shorter than the 6-day retirement window; older unmatured rows stop being retried
- **grouping**: observations grouped by symbol regardless of cohort; one bar series per symbol serves every cohort's observations — cohort-neutral: yes (shared bars); risk: none when every ready symbol is processed
- **ordering + cap**: symbols ordered by earliest PENDING anchor, then the first max_symbols processed — cohort-neutral: only when the cap does not bind; risk: when the cap binds, persistently failing old symbols stay at the head and a recurring symbol's newer observations ride along with its oldest pending one; CANDIDATE symbols recur far more than CONTROL symbols
- **per-run cap**: 400 symbols until b81daaf (2026-09-26), 2,000 after — cohort-neutral: capacity is counted in symbols, and CONTROL contributes ~1 symbol per observation; risk: CONTROL-heavy symbol counts make the cap bind on CONTROL first
- **horizons**: all ready horizons of an observation are processed together — cohort-neutral: yes; risk: none
- **retirement**: time-based (anchor + 6 days), cohort-agnostic code path — cohort-neutral: yes by construction; risk: inherits any earlier starvation
- **batching / caching / 429**: 100-symbol batches in the same oldest-first order; persistent 429 trips a circuit breaker for the remaining batches — cohort-neutral: yes, but the batches cut off by a breaker are the later ones in the ordering; risk: same as cap
- **failure retry**: failed symbols are retried every run until retirement — cohort-neutral: yes; risk: no-data symbols consume capacity each run

## Forward epoch

Status: **NO_FORWARD_DATA**
