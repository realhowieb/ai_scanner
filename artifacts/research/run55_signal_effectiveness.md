# Run 55 — Research Evidence & Signal Effectiveness

Generated 2026-09-26T07:23:11.062400+00:00 · schema `hsf-run55-signal-effectiveness-1.0` · READ-ONLY

## 1. Executive verdict

**Overall verdict: INSUFFICIENT_EVIDENCE** · evidence quality **INSUFFICIENT**

- Reason: CANDIDATE vs CONTROL positive at 1/4 horizons; only 1/4 horizons powered to rule out a 0.25% effect
- Readiness: **CONDITIONALLY_READY**
- Higher scores predict better outcomes: **INCONCLUSIVE** (Spearman CI > 0 at 4/4 horizons; scored n 803)
- Tiers separate outcome quality: **INCONCLUSIVE**
- Run 56 recommendation: **F. COLLECT_MORE_DATA** — CANDIDATE vs CONTROL positive at 1/4 horizons; only 1/4 horizons powered to rule out a 0.25% effect

Verdict rules were fixed in `analytics/signal_evidence.py` (`CRITERIA`, `primary_verdict`) before the production data was analyzed.

## 2. Dataset / readiness

| Metric | Value |
|---|---|
| total_observations | 4233 |
| explicitly_tagged | 3713 |
| legacy_inferred | 520 |
| unique_symbols | 1823 |
| unique_scan_runs | 16 |
| matured_observations | 1523 |
| unmatured_observations | 2190 |
| retirement_eligible_unmatured | 0 |
| matured_outcome_records | 5569 |
| directional_return_stored_coverage | 0.5473 |
| mfe_coverage | 0.5473 |
| mae_coverage | 0.5473 |
| duplicate_outcomes | 0 |
| conflicting_outcomes | 0 |
| orphan_outcomes | 0 |
| point_in_time_violations | 0 |
| invalid_observations | 0 |
| invalid_outcomes | 0 |
| extreme_returns_abs_gt_50pct | 8 |
| cohort_overlap_dropped | 5 |

Explicit observations by cohort: {'CANDIDATE': 1400, 'NEAR_MISS': 713, 'CONTROL': 1600} · all (incl. legacy): {'CONTROL': 1600, 'CANDIDATE': 1400, 'NEAR_MISS': 713, 'LEGACY_INFERRED': 520}
Primary-population N at +60m: {'CANDIDATE': 571, 'NEAR_MISS': 232, 'CONTROL': 110} · direction N: {'LONG': 803, 'SHORT': 0, 'UNKNOWN_(CONTROL)': 110}

Matured observations by cohort × horizon (explicit):

| Cohort | +5m | +15m | +30m | +60m |
|---|---|---|---|---|
| CANDIDATE | 857 | 849 | 798 | 768 |
| NEAR_MISS | 365 | 356 | 335 | 315 |
| CONTROL | 301 | 242 | 205 | 178 |

Readiness gates:

| Gate | Passed | If failed | Detail |
|---|---|---|---|
| modern_matured_observations | ✅ | INSUFFICIENT | 1523 >= 300 |
| cohort_primary_min | ✅ | INSUFFICIENT | min cohort n at +60m = 110 >= 30 |
| point_in_time | ✅ | INSUFFICIENT | 0 violations |
| conflicting_outcomes | ✅ | INSUFFICIENT | 0 conflicting duplicates |
| invalid_values | ✅ | INSUFFICIENT | invalid rate 0.0 <= 0.01 |
| cohort_primary_powered | ✅ | CONDITIONALLY_READY | min cohort n at +60m >= 100 |
| scan_runs | ❌ | CONDITIONALLY_READY | 16 distinct scan runs >= 20 |
| stored_directional_return | ❌ | CONDITIONALLY_READY | stored coverage 0.5473 (derived from raw_return otherwise) |
| excursion_coverage | ❌ | CONDITIONALLY_READY | MFE/MAE coverage 0.5473 |
| short_sample | ❌ | CONDITIONALLY_READY | SHORT n at +60m = 0 |

## 3. Cohort comparison

Directional returns, winsorized per horizon at pooled 0.5/99.5 percentiles (limits: {'+5m': [-0.03951, 0.036125], '+15m': [-0.038698, 0.101112], '+30m': [-0.086778, 0.139431], '+60m': [-0.107713, 0.140162]}). Per-group CIs are i.i.d. approximations; differences use a cluster bootstrap over scan runs.

### +5m

| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff |
|---|---|---|---|---|---|---|---|
| CANDIDATE | 657 | 47.9% [44.1%, 51.8%] | +0.037% [-0.009%, +0.083%] | +0.000% | +0.597% | -0.191% / +0.215% | 1.264 |
| NEAR_MISS | 280 | 35.4% [30.0%, 41.1%] | -0.027% [-0.070%, +0.015%] | -0.025% | +0.363% | -0.175% / +0.064% | 1.231 |
| CONTROL | 188 | 34.0% [27.7%, 41.1%] | -0.219% [-0.405%, -0.033%] | -0.102% | +1.302% | -0.503% / +0.176% | 1.059 |

| Comparison | N (A vs B) | Scan runs | Δ mean [95% cluster CI] | Δ win rate [CI] | MDE (80%) | Powered | + N/arm needed | Evidence |
|---|---|---|---|---|---|---|---|---|
| CANDIDATE − CONTROL | 657 vs 188 | 11 | +0.256% [+0.104%, +0.380%] | +13.9% [+5.4%, +27.3%] | +0.198% | yes | 0 | MODERATE_EVIDENCE |
| CANDIDATE − NEAR_MISS | 657 vs 280 | 11 | +0.064% [+0.006%, +0.149%] | +12.6% [+6.6%, +19.7%] | +0.103% | yes | 0 | MODERATE_EVIDENCE |
| NEAR_MISS − CONTROL | 280 vs 188 | 11 | +0.192% [+0.014%, +0.312%] | +1.3% [-8.2%, +13.8%] | +0.213% | yes | 0 | MODERATE_EVIDENCE |

### +15m

| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff |
|---|---|---|---|---|---|---|---|
| CANDIDATE | 649 | 47.3% [43.5%, 51.1%] | +0.032% [-0.057%, +0.121%] | -0.014% | +1.159% | -0.455% / +0.327% | 1.183 |
| NEAR_MISS | 271 | 41.3% [35.6%, 47.3%] | -0.060% [-0.130%, +0.010%] | -0.024% | +0.588% | -0.287% / +0.149% | 0.917 |
| CONTROL | 152 | 35.5% [28.4%, 43.4%] | -0.057% [-0.420%, +0.305%] | -0.165% | +2.281% | -0.755% / +0.185% | 1.607 |

| Comparison | N (A vs B) | Scan runs | Δ mean [95% cluster CI] | Δ win rate [CI] | MDE (80%) | Powered | + N/arm needed | Evidence |
|---|---|---|---|---|---|---|---|---|
| CANDIDATE − CONTROL | 649 vs 152 | 11 | +0.090% [-0.186%, +0.249%] | +11.8% [+8.3%, +22.8%] | +0.311% | no | 84 | WEAK_EVIDENCE |
| CANDIDATE − NEAR_MISS | 649 vs 271 | 11 | +0.092% [-0.010%, +0.187%] | +6.0% [-2.4%, +12.2%] | +0.141% | yes | 0 | WEAK_EVIDENCE |
| NEAR_MISS − CONTROL | 271 vs 152 | 11 | -0.002% [-0.352%, +0.239%] | +5.8% [-0.8%, +21.2%] | +0.422% | no | 281 | WEAK_EVIDENCE |

### +30m

| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff |
|---|---|---|---|---|---|---|---|
| CANDIDATE | 598 | 45.3% [41.4%, 49.3%] | -0.023% [-0.176%, +0.130%] | -0.089% | +1.903% | -0.668% / +0.470% | 1.135 |
| NEAR_MISS | 250 | 36.4% [30.7%, 42.5%] | -0.233% [-0.406%, -0.060%] | -0.111% | +1.395% | -0.656% / +0.104% | 0.817 |
| CONTROL | 127 | 29.1% [21.9%, 37.6%] | -0.140% [-0.713%, +0.434%] | -0.252% | +3.298% | -1.069% / +0.186% | 2.011 |

| Comparison | N (A vs B) | Scan runs | Δ mean [95% cluster CI] | Δ win rate [CI] | MDE (80%) | Powered | + N/arm needed | Evidence |
|---|---|---|---|---|---|---|---|---|
| CANDIDATE − CONTROL | 598 vs 127 | 10 | +0.117% [-0.662%, +0.763%] | +16.2% [+3.9%, +28.6%] | +1.018% | no | 1980 | WEAK_EVIDENCE |
| CANDIDATE − NEAR_MISS | 598 vs 250 | 9 | +0.210% [+0.012%, +0.417%] | +8.9% [-1.4%, +18.6%] | +0.289% | no | 85 | INSUFFICIENT |
| NEAR_MISS − CONTROL | 250 vs 127 | 10 | -0.093% [-1.067%, +0.575%] | +7.3% [-9.7%, +24.4%] | +1.173% | no | 2667 | WEAK_EVIDENCE |

### +60m

| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff | MFE mean / median | MAE mean / median |
|---|---|---|---|---|---|---|---|---|---|
| CANDIDATE | 571 | 44.0% [39.9%, 48.1%] | -0.086% [-0.269%, +0.097%] | -0.081% | +2.228% | -0.835% / +0.594% | 1.073 | +1.355% / +0.706% | -1.281% / -0.804% |
| NEAR_MISS | 232 | 39.7% [33.6%, 46.1%] | -0.283% [-0.477%, -0.090%] | -0.143% | +1.504% | -0.890% / +0.205% | 0.745 | +0.620% / +0.244% | -1.197% / -0.676% |
| CONTROL | 110 | 34.5% [26.3%, 43.8%] | -0.569% [-1.273%, +0.135%] | -0.471% | +3.766% | -1.335% / +0.261% | 1.079 | +4.034% / +0.270% | -2.185% / -1.184% |

| Comparison | N (A vs B) | Scan runs | Δ mean [95% cluster CI] | Δ win rate [CI] | MDE (80%) | Powered | + N/arm needed | Evidence |
|---|---|---|---|---|---|---|---|---|
| CANDIDATE − CONTROL | 571 vs 110 | 9 | +0.483% [-0.131%, +0.935%] | +9.4% [-3.6%, +25.4%] | +0.761% | no | 910 | INSUFFICIENT |
| CANDIDATE − NEAR_MISS | 571 vs 232 | 8 | +0.197% [-0.080%, +0.545%] | +4.3% [-6.8%, +15.7%] | +0.446% | no | 507 | INSUFFICIENT |
| NEAR_MISS − CONTROL | 232 vs 110 | 9 | +0.285% [-0.560%, +0.670%] | +5.1% [-7.1%, +21.9%] | +0.878% | no | 1248 | INSUFFICIENT |

## 4. Horizon comparison

| Horizon | CAND−CTRL Δ mean | CI sign | Evidence | CAND−NM Δ mean | CI sign | NM−CTRL Δ mean | CI sign |
|---|---|---|---|---|---|---|---|
| +5m | +0.256% | POSITIVE | MODERATE_EVIDENCE | +0.064% | POSITIVE | +0.192% | POSITIVE |
| +15m | +0.090% | NONE | WEAK_EVIDENCE | +0.092% | NONE | -0.002% | NONE |
| +30m | +0.117% | NONE | WEAK_EVIDENCE | +0.210% | POSITIVE | -0.093% | NONE |
| +60m | +0.483% | NONE | INSUFFICIENT | +0.197% | NONE | +0.285% | NONE |

## 5. Score monotonicity

**DO HIGHER SCORES PREDICT BETTER OUTCOMES? INCONCLUSIVE** — Spearman CI > 0 at 4/4 horizons; scored n 803

Population: CANDIDATE + NEAR_MISS with BreakoutScore. Bucket mode: **quintile** (Q1 [-inf, 12.3), Q2 [12.3, 15.1), Q3 [15.1, 19.4), Q4 [19.4, 26.9), Q5 [26.9, +inf)); scored N = 803; score quantiles [min, p10, p25, p50, p75, p90, max] = [8.184, 11.079, 12.854, 17.339, 24.744, 32.969, 73.041].

### +5m — Spearman 0.1199 [+0.049, +0.188] (POSITIVE), Pearson 0.1449, adjacent violations 0 of 4

| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff |
|---|---|---|---|---|---|---|---|
| Q1 [-inf, 12.3) | 184 | 35.3% [28.8%, 42.5%] | -0.037% [-0.092%, +0.019%] | -0.015% | +0.385% | -0.189% / +0.065% | 1.124 |
| Q2 [12.3, 15.1) | 190 | 35.3% [28.8%, 42.3%] | -0.017% [-0.078%, +0.043%] | -0.048% | +0.427% | -0.212% / +0.092% | 1.51 |
| Q3 [15.1, 19.4) | 189 | 43.9% [37.0%, 51.0%] | -0.015% [-0.074%, +0.044%] | -0.011% | +0.415% | -0.163% / +0.145% | 1.04 |
| Q4 [19.4, 26.9) | 178 | 49.4% [42.2%, 56.7%] | +0.036% [-0.055%, +0.126%] | +0.000% | +0.617% | -0.179% / +0.231% | 1.182 |
| Q5 [26.9, +inf) | 196 | 56.6% [49.6%, 63.4%] | +0.118% [+0.014%, +0.222%] | +0.044% | +0.742% | -0.175% / +0.277% | 1.322 |

### +15m — Spearman 0.0849 [+0.009, +0.146] (POSITIVE), Pearson 0.1859, adjacent violations 1 of 4

| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff |
|---|---|---|---|---|---|---|---|
| Q1 [-inf, 12.3) | 179 | 40.2% [33.3%, 47.5%] | -0.054% [-0.138%, +0.030%] | -0.010% | +0.576% | -0.277% / +0.120% | 0.951 |
| Q2 [12.3, 15.1) | 185 | 40.0% [33.2%, 47.2%] | -0.104% [-0.206%, -0.002%] | -0.076% | +0.707% | -0.467% / +0.219% | 0.929 |
| Q3 [15.1, 19.4) | 186 | 46.8% [39.7%, 53.9%] | -0.073% [-0.159%, +0.013%] | -0.015% | +0.599% | -0.466% / +0.256% | 0.782 |
| Q4 [19.4, 26.9) | 177 | 48.0% [40.8%, 55.4%] | -0.064% [-0.206%, +0.078%] | -0.039% | +0.965% | -0.468% / +0.326% | 0.876 |
| Q5 [26.9, +inf) | 193 | 52.3% [45.3%, 59.3%] | +0.303% [+0.062%, +0.545%] | +0.036% | +1.711% | -0.319% / +0.570% | 1.823 |

### +30m — Spearman 0.0956 [+0.051, +0.134] (POSITIVE), Pearson 0.1059, adjacent violations 2 of 4

| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff |
|---|---|---|---|---|---|---|---|
| Q1 [-inf, 12.3) | 174 | 37.9% [31.1%, 45.3%] | -0.133% [-0.256%, -0.010%] | -0.072% | +0.829% | -0.523% / +0.098% | 0.896 |
| Q2 [12.3, 15.1) | 170 | 32.4% [25.8%, 39.7%] | -0.330% [-0.608%, -0.051%] | -0.270% | +1.853% | -0.967% / +0.158% | 1.038 |
| Q3 [15.1, 19.4) | 172 | 45.4% [38.1%, 52.8%] | -0.048% [-0.206%, +0.110%] | -0.072% | +1.059% | -0.570% / +0.370% | 1.042 |
| Q4 [19.4, 26.9) | 163 | 44.8% [37.4%, 52.4%] | -0.129% [-0.372%, +0.115%] | -0.107% | +1.589% | -0.809% / +0.439% | 0.957 |
| Q5 [26.9, +inf) | 169 | 53.2% [45.7%, 60.6%] | +0.216% [-0.208%, +0.640%] | +0.048% | +2.812% | -0.439% / +0.518% | 1.195 |

### +60m — Spearman 0.0528 [+0.015, +0.095] (POSITIVE), Pearson 0.0613, adjacent violations 2 of 4

| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff | MFE mean / median | MAE mean / median |
|---|---|---|---|---|---|---|---|---|---|
| Q1 [-inf, 12.3) | 161 | 44.1% [36.7%, 51.8%] | -0.205% [-0.370%, -0.041%] | -0.034% | +1.064% | -0.607% / +0.285% | 0.683 | +0.594% / +0.240% | -0.950% / -0.495% |
| Q2 [12.3, 15.1) | 160 | 35.0% [28.0%, 42.7%] | -0.311% [-0.605%, -0.018%] | -0.279% | +1.894% | -1.111% / +0.214% | 1.031 | +0.829% / +0.354% | -1.340% / -1.125% |
| Q3 [15.1, 19.4) | 161 | 42.9% [35.5%, 50.6%] | -0.047% [-0.279%, +0.184%] | -0.108% | +1.497% | -0.713% / +0.530% | 1.179 | +0.967% / +0.440% | -0.960% / -0.721% |
| Q4 [19.4, 26.9) | 160 | 41.2% [33.9%, 49.0%] | -0.188% [-0.473%, +0.097%] | -0.111% | +1.837% | -0.797% / +0.496% | 1.002 | +1.322% / +0.714% | -1.302% / -0.868% |
| Q5 [26.9, +inf) | 161 | 50.3% [42.7%, 57.9%] | +0.035% [-0.469%, +0.538%] | +0.026% | +3.260% | -0.812% / +0.731% | 0.977 | +1.925% / +0.967% | -1.721% / -0.821% |

## 6. Tier effectiveness

**DO CURRENT TIERS SEPARATE OUTCOME QUALITY? INCONCLUSIVE**

Not measurable: production Day Trader tiers (Strong/Developing/Weak) are computed at render time from VWAP/ADX/SuperTrend/EWO and are not persisted on observations; 0 of 913 primary observations carry a tier.

## 7. LONG vs SHORT

**INSUFFICIENT** — SHORT n at +60m = 0

SHORT transform check (stored directional_return == −raw for SHORT, == raw for LONG): 2451 checked, 0 mismatches; stored 3040, derived 2513.

### +5m

| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff |
|---|---|---|---|---|---|---|---|
| LONG | 937 | 44.2% [41.0%, 47.4%] | +0.018% [-0.017%, +0.052%] | -0.010% | +0.539% | -0.186% / +0.164% | 1.295 |
| SHORT | 0 | — — | — — | — | — | — / — | — |

### +15m

| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff |
|---|---|---|---|---|---|---|---|
| LONG | 920 | 45.5% [42.4%, 48.8%] | +0.005% [-0.061%, +0.072%] | -0.015% | +1.025% | -0.410% / +0.268% | 1.141 |
| SHORT | 0 | — — | — — | — | — | — / — | — |

### +30m

| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff |
|---|---|---|---|---|---|---|---|
| LONG | 848 | 42.7% [39.4%, 46.0%] | -0.085% [-0.204%, +0.034%] | -0.096% | +1.770% | -0.665% / +0.367% | 1.072 |
| SHORT | 0 | — — | — — | — | — | — / — | — |

### +60m

| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff | MFE mean / median | MAE mean / median |
|---|---|---|---|---|---|---|---|---|---|
| LONG | 803 | 42.7% [39.3%, 46.2%] | -0.143% [-0.285%, -0.002%] | -0.096% | +2.047% | -0.842% / +0.470% | 1.008 | +1.124% / +0.525% | -1.255% / -0.788% |
| SHORT | 0 | — — | — — | — | — | — / — | — | — / — | — / — |

## 8. Time-of-day

**NOT_MATERIAL** — candidate mean spread +0.2070% between 14:00-15:00 and 15:00-16:00 (overlapping CIs)

Population: explicit observations, all sessions. N per bucket at +60m: {'PRE (<09:30)': 162, '09:30-10:30': 222, '10:30-12:00': 0, '12:00-14:00': 176, '14:00-15:00': 170, '15:00-16:00': 113, 'POST (>=16:00)': 99}

### +60m — CANDIDATE by bucket

| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff | MFE mean / median | MAE mean / median |
|---|---|---|---|---|---|---|---|---|---|
| PRE (<09:30) | 99 | 35.4% [26.6%, 45.2%] | -0.679% [-1.155%, -0.203%] | -0.572% | +2.416% | -2.208% / +0.489% | 0.8 | +16.644% / +16.644% | +0.000% / +0.000% |
| 09:30-10:30 | 184 | 47.3% [40.2%, 54.5%] | -0.026% [-0.368%, +0.316%] | -0.057% | +2.365% | -1.317% / +0.991% | 1.046 | +1.826% / +1.167% | -1.090% / -0.885% |
| 10:30-12:00 | 0 | — — | — — | — | — | — / — | — | — / — | — / — |
| 12:00-14:00 | 149 | 37.6% [30.2%, 45.6%] | -0.107% [-0.348%, +0.133%] | -0.119% | +1.498% | -0.569% / +0.268% | 1.187 | +1.018% / +0.603% | -0.818% / -0.480% |
| 14:00-15:00 | 138 | 44.9% [36.9%, 53.2%] | -0.206% [-0.545%, +0.133%] | -0.051% | +2.031% | -0.714% / +0.345% | 0.776 | +0.989% / +0.536% | -1.212% / -0.651% |
| 15:00-16:00 | 100 | 46.0% [36.6%, 55.7%] | +0.001% [-0.593%, +0.594%] | -0.016% | +3.026% | -1.027% / +1.132% | 1.153 | +1.455% / +0.768% | -1.820% / -1.108% |
| POST (>=16:00) | 98 | 62.2% [52.4%, 71.2%] | +0.735% [+0.237%, +1.233%] | +0.312% | +2.515% | -0.264% / +1.971% | 1.639 | +1.774% / +1.014% | -0.947% / -0.486% |

### +60m — CONTROL by bucket

| Group | N | Win rate [95% CI] | Mean [95% CI] | Median | SD | P25 / P75 | Payoff | MFE mean / median | MAE mean / median |
|---|---|---|---|---|---|---|---|---|---|
| PRE (<09:30) | 63 | 39.7% [28.5%, 52.0%] | -0.651% [-1.588%, +0.286%] | -0.298% | +3.794% | -1.818% / +0.579% | 0.822 | +1.639% / +0.644% | -3.225% / -2.166% |
| 09:30-10:30 | 38 | 31.6% [19.1%, 47.5%] | -0.851% [-2.143%, +0.441%] | -0.543% | +4.064% | -1.302% / +0.349% | 1.024 | +4.244% / +0.279% | -2.276% / -0.987% |
| 10:30-12:00 | 0 | — — | — — | — | — | — / — | — | — / — | — / — |
| 12:00-14:00 | 27 | 48.1% [30.7%, 66.0%] | -0.507% [-1.598%, +0.584%] | -0.143% | +2.892% | -0.878% / +0.206% | 0.509 | +0.994% / +0.251% | -1.880% / -0.857% |
| 14:00-15:00 | 32 | 28.1% [15.6%, 45.4%] | -0.810% [-2.123%, +0.504%] | -0.571% | +3.791% | -1.706% / +0.251% | 1.159 | +4.156% / +0.462% | -2.418% / -1.405% |
| 15:00-16:00 | 13 | 30.8% [12.7%, 57.6%] | +0.720% [-1.738%, +3.178%] | -0.670% | +4.522% | -1.071% / +0.200% | 4.071 | +9.064% / +0.133% | -1.974% / -1.166% |
| POST (>=16:00) | 1 | 100.0% [20.6%, 100.0%] | +1.445% — | +1.445% | — | +1.445% / +1.445% | — | +2.966% / +2.966% | -0.342% / -0.342% |

| Comparison | N (A vs B) | Scan runs | Δ mean [95% cluster CI] | Δ win rate [CI] | MDE (80%) | Powered | + N/arm needed | Evidence |
|---|---|---|---|---|---|---|---|---|
| PRE (<09:30): CAND − CTRL | 99 vs 63 | 2 | -0.028% [-0.579%, -0.028%] | -4.3% [-14.6%, -4.3%] | +0.393% | no | 93 | INSUFFICIENT |
| 09:30-10:30: CAND − CTRL | 184 vs 38 | 3 | +0.825% [+0.224%, +2.513%] | +15.7% [+3.7%, +60.0%] | +1.635% | no | 1587 | INSUFFICIENT |
| 10:30-12:00: CAND − CTRL | 0 vs 0 | 0 | — — | — — | — | no | None | INSUFFICIENT |
| 12:00-14:00: CAND − CTRL | 149 vs 27 | 2 | +0.399% [+0.385%, +0.726%] | -10.6% [-15.0%, +42.9%] | +0.243% | no | 0 | INSUFFICIENT |
| 14:00-15:00: CAND − CTRL | 138 vs 32 | 2 | +0.603% [+0.535%, +0.603%] | +16.8% [+16.8%, +16.9%] | +0.049% | no | 0 | INSUFFICIENT |
| 15:00-16:00: CAND − CTRL | 100 vs 13 | 2 | -0.719% [-0.853%, -0.719%] | +15.2% [+12.7%, +15.2%] | +0.096% | no | 0 | INSUFFICIENT |
| POST (>=16:00): CAND − CTRL | 98 vs 1 | 1 | -0.710% — | -37.8% — | — | no | None | INSUFFICIENT |

## 9. Market regime

**REGIME ANALYSIS UNAVAILABLE** (point-in-time regime present on 0.0% of primary observations). market_context.market_regime is not populated at capture (scheduler passes no point-in-time regime); not reconstructed to avoid leakage

## 10. Feature / conflict diagnostics

Population: CANDIDATE + NEAR_MISS. 14 features x 4 horizons; expect ~1 in 20 spurious CI exclusions. Diagnostic only.

| Feature | Class | Spearman +5m [CI] | Spearman +15m [CI] | Spearman +30m [CI] | Spearman +60m [CI] | +60m winners mean / losers mean |
|---|---|---|---|---|---|---|
| score | USEFUL | +0.120 [+0.053, +0.187] (n=937) | +0.085 [+0.010, +0.145] (n=920) | +0.096 [+0.052, +0.135] (n=848) | +0.053 [+0.013, +0.094] (n=803) | 20.729993 / 19.691913 |
| rvol | NEUTRAL | +0.004 [-0.057, +0.145] (n=935) | +0.041 [-0.027, +0.124] (n=918) | +0.064 [-0.064, +0.176] (n=846) | +0.027 [-0.137, +0.158] (n=802) | 0.711905 / 0.763285 |
| gap_pct | NEUTRAL | -0.050 [-0.216, +0.093] (n=937) | -0.008 [-0.163, +0.111] (n=920) | +0.026 [-0.103, +0.132] (n=848) | +0.028 [-0.086, +0.132] (n=803) | -0.163906 / -0.208964 |
| chg_pct | NEUTRAL | +0.075 [-0.021, +0.185] (n=937) | +0.117 [+0.001, +0.221] (n=920) | +0.042 [-0.048, +0.143] (n=848) | -0.004 [-0.081, +0.088] (n=803) | 0.432819 / 0.36193 |
| atr_pct | NEUTRAL | +0.050 [-0.065, +0.159] (n=935) | +0.012 [-0.126, +0.137] (n=918) | -0.008 [-0.076, +0.047] (n=846) | +0.005 [-0.055, +0.052] (n=802) | 4.992211 / 5.028177 |
| price | USEFUL | +0.047 [-0.015, +0.129] (n=937) | +0.038 [-0.046, +0.136] (n=920) | +0.041 [-0.037, +0.137] (n=848) | +0.074 [+0.008, +0.177] (n=803) | 169.033848 / 147.736478 |
| volume | NEUTRAL | -0.041 [-0.109, +0.064] (n=937) | +0.008 [-0.077, +0.098] (n=920) | +0.051 [-0.083, +0.175] (n=848) | +0.018 [-0.154, +0.132] (n=803) | 318331.48105 / 326449.667391 |
| dollar_volume | NEUTRAL | -0.018 [-0.079, +0.085] (n=937) | +0.025 [-0.057, +0.137] (n=920) | +0.055 [-0.079, +0.191] (n=848) | +0.051 [-0.104, +0.185] (n=803) | 30862782.915575 / 21836557.068491 |
| trigger_count | NEUTRAL | +0.033 [-0.027, +0.094] (n=937) | +0.030 [-0.029, +0.080] (n=920) | +0.007 [-0.041, +0.051] (n=848) | -0.064 [-0.099, +0.000] (n=803) | 4.029155 / 4.058696 |

| Flag (with − without) | Class | +5m | +15m | +30m | +60m |
|---|---|---|---|---|---|
| is_breakout | INSUFFICIENT | -0.030% [-0.126%, +0.111%] (69/868) | -0.013% [-0.152%, +0.306%] (68/852) | -0.112% [-0.347%, +0.155%] (62/786) | -0.419% [-0.594%, +0.075%] (56/747) |
| gap_up | NEUTRAL | -0.047% [-0.174%, +0.035%] (383/554) | -0.015% [-0.185%, +0.100%] (378/542) | -0.148% [-0.389%, +0.013%] (359/489) | -0.197% [-0.503%, +0.039%] (345/458) |
| gap_down | NEUTRAL | +0.054% [-0.032%, +0.183%] (547/390) | +0.023% [-0.089%, +0.188%] (535/385) | +0.153% [-0.012%, +0.395%] (483/365) | +0.198% [-0.044%, +0.505%] (454/349) |
| unusual_vol | INSUFFICIENT | +0.146% [+0.024%, +0.430%] (88/849) | +0.437% [+0.165%, +1.142%] (88/832) | +0.450% [-0.104%, +1.675%] (86/762) | +0.239% [-0.485%, +2.306%] (81/722) |
| momentum | INSUFFICIENT | +0.049% [-0.022%, +0.163%] (829/108) | +0.082% [+0.013%, +0.133%] (812/108) | +0.109% [-0.092%, +0.463%] (744/104) | +0.059% [-0.128%, +0.390%] (707/96) |

Not measurable (not persisted at observation time):

- **rsi**: not captured by the scheduled scan observation
- **ema_alignment**: not captured
- **ema9_ema21**: ema_cross is computed by scan/breakout.py but not mapped into the observation
- **breakout_distance**: BreakoutPos20D is not mapped into the observation
- **confirmation_count / agreement**: Day Trader intel is computed at render time, never persisted
- **conflict_count / conflict flags**: Day Trader conflicts are computed at render time, never persisted
- **prebreakout**: scheduled observations carry no PreBreakout score
- **vwap / adx / supertrend / ewo**: not in the scheduled breakout result frame

## 11. MFE / MAE quality

Horizon +60m. +60m records only: shorter-horizon MFE/MAE windows vary with batch composition.

| Group | N | MFE mean / median | MAE mean / median | MFE ÷ abs(MAE) | Strong move → reversal | Early adverse → winner | Early favourable → loser |
|---|---|---|---|---|---|---|---|
| CANDIDATE | 571 | +1.355% / +0.706% | -1.281% / -0.804% | 1.058 | 5.8% | 16.3% | 19.8% |
| NEAR_MISS | 232 | +0.620% / +0.244% | -1.197% / -0.676% | 0.518 | 2.2% | 17.2% | 16.0% |
| CONTROL | 110 | +4.034% / +0.270% | -2.185% / -1.184% | 1.846 | 7.3% | 18.2% | 16.4% |
| score Q1 [-inf, 12.3) | 161 | +0.594% / +0.240% | -0.950% / -0.495% | 0.625 | 1.2% | 17.4% | 13.7% |
| score Q2 [12.3, 15.1) | 160 | +0.829% / +0.354% | -1.340% / -1.125% | 0.619 | 3.8% | 16.9% | 18.1% |
| score Q3 [15.1, 19.4) | 161 | +0.967% / +0.440% | -0.960% / -0.721% | 1.007 | 3.1% | 17.4% | 21.7% |
| score Q4 [19.4, 26.9) | 160 | +1.322% / +0.714% | -1.302% / -0.868% | 1.016 | 8.8% | 17.5% | 21.2% |
| score Q5 [26.9, +inf) | 161 | +1.925% / +0.967% | -1.721% / -0.821% | 1.119 | 6.8% | 13.7% | 18.6% |

## 12. Statistical power

Meaningful effect (pre-declared): 0.25%. MDE is the difference detectable with 80% power at α = 0.05, from the cluster-bootstrap SE. “+ N/arm needed” scales the smaller arm by (MDE / meaningful effect)².

| Horizon | CAND−CTRL MDE | Powered | + N/arm | CAND−NM MDE | Powered | + N/arm |
|---|---|---|---|---|---|---|
| +5m | +0.198% | yes | 0 | +0.103% | yes | 0 |
| +15m | +0.311% | no | 84 | +0.141% | yes | 0 |
| +30m | +1.018% | no | 1980 | +0.289% | no | 85 |
| +60m | +0.761% | no | 910 | +0.446% | no | 507 |

## 13. Limitations

- Horizons count minute bars, not wall-clock minutes; sparse symbols' “+60m” can span hours.
- CONTROL rows have no direction and are measured long; every CANDIDATE/NEAR_MISS row's first scanner is `breakout` (long), so SHORT is effectively unobserved.
- Outcomes matured before Run 53A lack stored directional_return/MFE/MAE; directional_return is derived from raw_return for those, MFE/MAE cannot be recovered.
- Candidates and near-misses are adjacent in score rank (top-N vs just below), which restricts the score range within the scored population.
- Observations of the same symbol in consecutive runs overlap in time; the cluster bootstrap treats scan runs, not symbols, as independent.
- Tiers, conflicts, EMA/RSI and PreBreakout are not persisted on observations (Parts 4 and 8).
- Point-in-time market regime is not captured (Part 9).

## 14. Run 56 recommendation

**F. COLLECT_MORE_DATA** — CANDIDATE vs CONTROL positive at 1/4 horizons; only 1/4 horizons powered to rule out a 0.25% effect

- Parts 4 and 8 (tiers, conflicts, EMA/RSI/PreBreakout) cannot be evaluated until the scheduled capture persists those point-in-time fields.

Not implemented in Run 55.
