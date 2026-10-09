# ML v3 final report and ML v4 recommendations

Dataset `hsf-ml-2026-10-08-v1`, fingerprint `sha256:527bbd62ae511db111679f77949be0acd9bbed06ca1ac0a83196c1ad1e7886cc`
(frozen 2026-10-08 21:04:21Z, reproducible on rebuild). Audit slice `sha256:5afdb61396903bd9f70b1a3bfec69d0c415d3f6dd9c898e7a238c0382503d065`,
identical on both production audit runs. Audit code at git `63f3dbe`. Every number below comes from the files in this
folder; nothing is illustrative.

## Verdict

**E — INSUFFICIENT TRUSTWORTHY DATA.**

The canonical point-in-time dataset holds 134 matured, certified signal-days over 13 entry days (2026-09-14 to
2026-09-30), all in one sideways, low-volatility market. That supports one or two purged walk-forward folds of about
25 validation rows. No model, including the HSF Score, can be told apart from a coin flip at that size: every 5-day
AUC sits between 0.47 and 0.51, with confidence intervals spanning roughly 0.0 to 0.6.

Secondary finding, recorded so it isn't lost: the **historical ~0.963 AUC was materially inflated (the evidence for D).**
It comes from retired models validated with a random split on a label that is close to the current scanner state.
Re-running the same recipe on 149,416 current scan rows gives 0.887 with a random split and 0.723 once the split is
chronological, purged and free of snapshot copies. The current IsBreakout flag on its own scores 0.758. The active
AI Confidence recipe drops from its reported 0.580 (0.591 reproduced) to 0.528 with a purge and 0.543 in a 5-fold
walk-forward. E is the primary verdict because the question the spec asks, whether today's signal is real, can't be
answered on today's data. D answers a different question, about the old headline number.

## ML v4 challenger training: **NO-GO**

Training a challenger on the canonical dataset now would fit about 100 rows and validate on about 25. Any result would
be noise, and a holdout isn't possible (reserving 108 rows leaves 0 folds). The data and infrastructure work below
comes first. Once each item holds, this becomes GO with the pre-registered spec in section 30-34.

Required first:

1. **At least 300 matured, certified signal-days** at the 5-day horizon, spread over at least 30 entry days, enough
   for 3 or more purged folds plus a holdout. This is inferred from the current rate: about 10 matured signal-days per
   trading day means roughly 17 more trading days of clean scheduled scans, plus 5 days to mature.
2. **SPY benchmark coverage of at least 90%** of matured rows. It is 21.6% now (209 of 966 scored rows). Labels C, D
   and E and every excess-return column depend on it.
3. **Record the served model version on every observation.** It is UNKNOWN on all 1,041. The code constant says
   `prebreakout-xgb-v16` while the active registry row is v9.
4. **Freeze the AI Confidence output (and its inputs) per observation.** It is not stored anywhere research can read,
   so the audit could score it point-in-time on only 65 rows, and on 0 in a validation fold.
5. **Raise the scan-feature join rate.** It is 22.4%. 703 observations only have scans written after them, so their
   price, volume and trend features are missing at observation time. Either freeze the inputs on the observation row
   or write the scan before the opportunity snapshot.
6. **Store a 10/15/20-bar outcome** if those horizons matter. None exists, and this audit did not invent one.

## Final report (items 1-35)

1. **Dataset version:** `hsf-ml-2026-10-08-v1` (feature schema 1, label schema 1).
2. **Fingerprint:** `sha256:527bbd62ae511db111679f77949be0acd9bbed06ca1ac0a83196c1ad1e7886cc`.
3. **Date range:** observations 2026-09-12 06:54Z to 2026-10-08 13:40Z. Matured entry days 2026-09-14 to 09-30.
4. **Total observations:** 1,041 (385 signal-days after collapsing 656 repeated same-day snapshots).
5. **Matured:** 337 observations (all certified), 134 signal-days. 75 pending, 629 unavailable (mostly the
   2026-09-12/13 burst), all kept in the dataset and excluded only by maturity.
6. **Feature coverage:** HSF score components 100%, BreakoutScore 97%, prebreakout_prob 57%, scan-joined features
   (price, volume, RVOL, volatility) 22%, trend, rank and rs_vs_spy 12%. RSI, EMA values and market regime 0%.
7. **Outcome coverage:** 34.9% of scored rows at 1, 3 and 5 days.
8. **Benchmark coverage:** 21.6%.
9. **MFE/MAE coverage:** 34.9% (all matured rows).
10. **Current production model:** PreBreakout `prebreakout-xgb-v9` (XGBoost, 75 features) feeds the HSF Score through
    `max(BreakoutScore/60, PreBreakout%/100) × 38`. AI Confidence `ai-confidence-xgb-v1` (XGBoost, 6 features) is a
    separate column. The HSF Score is a hand-weighted heuristic, not a probability. See `ml_v3_production_model.md`.
11. **Current target:** PreBreakout predicts FutureQualitySetupHit (a quality setup within 3 sessions that then hits
    +4% before −2%). AI Confidence predicts +4% before −2% within 5 days. Users are shown return > 0 and beat-SPY.
12. **Historical reported AUC:** 0.962 / 0.956 / 0.936 (`prebreakout-xgb-v1`, Jun 30 to Jul 14), 0.968
    (`prebreakout-xgb-v5`, replaced 26 minutes later by 0.592), 0.998 (July AI Confidence). Active models report 0.668
    and 0.580.
13. **Reproduced legacy AUC:** PreBreakout v1 recipe 0.887 with a random split, 0.714 chronological, 0.723 purged with
    no snapshot copies. AI Confidence v1 (same-row label) 0.999 random, 0.979 chronological. Current AI Confidence
    recipe 0.591 as validated, 0.796 random, 0.528 purged, 0.543 walk-forward.
14. **Leakage-safe walk-forward AUC (5-day, research dataset):** HSF Score 0.507, logistic 0.493, both XGBoost
    challengers 0.500 (they predict a constant on about 100 training rows), random 0.467, majority 0.500. One fold,
    N = 25. Served PreBreakout % and AI Confidence: not evaluable (59 and 39 unscorable rows).
15. **Leakage-safe PR-AUC:** 0.58 to 0.62 against a 0.60 base rate.
16. **Fold stability:** not measurable. 1 fold at 3 and 5 days, 2 at 1 day.
17. **Baseline comparison:** nothing beats the majority or random baselines. Model complexity adds nothing measurable.
18. **HSF score monotonicity:** none, on any metric. Spearman rho of score against 5-day return is −0.128 (p 0.14,
    N 134).
19. **Inversions:** 50-59 has a higher win rate than 60-69, and 70-79 higher than 80-89. 80-89 is the worst bucket
    (42.9% win, −2.22% median). 90-100 has N = 1 and is not interpretable.
20. **Best-supported horizon:** none. 10/15/20 bars have no outcomes. Among 1/3/5 days no horizon is distinguishable,
    and none was picked.
21. **Horizon stability:** INSUFFICIENT at every horizon (fewer than 3 evaluable periods).
22. **Feature importance:** all features LOW VALUE. The XGBoost challengers learned no splits, so gain, permutation and
    SHAP are all 0. This reflects sample size, not proof that the features are useless.
23. **Ablation:** no group changes the XGBoost result. For the logistic model, removing momentum (−0.053) or
    PreBreakout (−0.027) lowers AUC slightly on 25 rows. That is within noise.
24. **Leakage findings:** no look-ahead joins (0 violations, median join lag 206 s). PreBreakout % and the HSF model
    component are REVIEW because the served version is unrecorded and the inputs changed on 2026-10-07/08. Sector,
    fundamentals and universe membership would leak if joined, and are not used. In production training: the
    PreBreakout purge (5 days) is shorter than its label (about 8 days); training reads daily-snapshot copies; AI
    Confidence has no purge and calibrates on its own validation set.
25. **Market-context gaps:** SPY/QQQ trend and volatility are safe to derive. rs_vs_spy is stored but 88% null.
    Breadth has insufficient history. Sector features are unsafe (only today's map). Stock-vs-QQQ is missing.
26. **Regimes:** 12 of 13 entry days are sideways/low-volatility and 1 bearish, so there is nothing to compare.
27. **Ticker concentration:** 69 tickers, top ticker VICR 6.0%, top 10 33.6%. Excluding the top 10 leaves 13
    validation rows, too few for an AUC.
28. **Probability calibration:** Brier 0.240 to 0.243 for the HSF map and XGBoost (equal to predicting the base rate),
    ECE 0.14 for the HSF Score map. Platt and isotonic need a calibrator fitted on earlier folds, and with one fold
    they could not be run. They were not fitted on the evaluation fold.
29. **Final holdout:** not valid. No holdout was created.
30. **Recommended ML v4 target:** `excess_return_5d > 0` (label C, beat SPY), the metric users see, with
    `return_5d > 0` (A) as a co-reported secondary. Report F (MFE ≥ +4%, MAE > −2%) as a path-quality diagnostic, not
    as the training target. Pre-register these before training; don't choose after seeing results.
31. **Recommended ML v4 features:** only SAFE_STORED or SAFE_ASOF_JOIN fields frozen on the observation: HSF score
    components, BreakoutScore, chg_pct, gap_pct, RVOL, 20-day volatility, trend 10/20-day, rs_vs_spy, setup and
    signal flags, plus SPY trend and volatility from prior closes.
32. **Remove or review:** PreBreakout % and `hsf_model_component` until the served version is frozen per row.
    Same-row IsBreakout and anything derived from future scan rows. `fillna(0)` on missing scan features (use explicit
    missing indicators). Never join sector, fundamentals or universe membership.
33. **Calibration strategy:** fit isotonic (or Platt when a fold has under about 200 rows) on the out-of-fold
    predictions of earlier folds only. Report Brier, ECE and the reliability table on later folds. Never calibrate on
    the fold an AUC is reported on.
34. **Champion/challenger design:** the framework in `analytics/ml_v3_audit.py` is ready. Champion = HSF Score as
    served. Challengers = logistic and leakage-safe XGBoost (fixed hyperparameters, no search). Unit = signal-day.
    Expanding folds of at least 60 training and 30 validation rows, purge = label window, 1-day embargo, entry-day block
    bootstrap CIs. Final holdout = the last 5+ entry days, evaluated once. Each run writes an experiment record
    (`ml_v3_walk_forward_results.json → experiments`) with dataset version, fingerprint, schema versions, windows, purge
    and git revision. A challenger is promotable only if its lower 95% AUC bound beats the champion's point AUC on the
    holdout and its top-third beat-SPY rate is higher.
35. **Production HSF scoring should stay unchanged** while ML v4 is developed. Nothing here shows the score adds
    signal, but nothing shows an alternative does either, and changing it now would break the frozen observation
    series that ML v4 needs. The 80-89 inversion should be re-checked once there are at least 300 signal-days.

## Mandatory tables

The comparison, horizon and score tables are in `ml_v3_walk_forward_summary.md`, generated from the run. The rows
behind them are in `ml_v3_model_baselines.csv`, `ml_v3_horizon_comparison.csv` and `ml_v3_score_calibration.csv`.

## How to re-run

Dispatch the **Diagnostics** workflow with `script = ml_v3_audit.py` on this branch. It reuses the frozen dataset
version (it never creates a second one once a version exists), applies the as-of rule so later outcome backfills don't
change the slice, and prints the reports as a bundle in the log.
