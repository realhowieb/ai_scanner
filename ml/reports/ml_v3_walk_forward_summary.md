# ML v3 walk-forward summary

Dataset `hsf-ml-2026-10-08-v1` (`sha256:527bbd62ae511db111679f77949be0acd9bbed06ca1ac0a83196c1ad1e7886cc`), audit slice `sha256:5afdb61396903bd9f70b1a3bfec69d0c415d3f6dd9c898e7a238c0382503d065`, git `63f3dbebc8c349dceeecfeb463692e738e1b1fbf`. Primary horizon 5 trading days and primary label return > 0 were fixed before any result was seen (Outcome Intelligence defaults). Financial columns are Outcome Intelligence metrics on each model's top third of validation predictions per fold. Benchmark-based columns are n/a when SPY coverage is missing.

## Mandatory comparison table (5-day horizon)

| MODEL | VALIDATION | N | ROC-AUC [95% CI] | PR-AUC | BRIER | WIN RATE | MEDIAN EXCESS RETURN | BENCHMARK BEAT RATE | MFE | MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| PreBreakout v1 (production 2026-06-30), as reported | random stratified 80/20, own label | n/a | 0.962 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| PreBreakout v1 recipe, reproduced on current runs | random stratified 80/20 | 149416 | 0.887 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| PreBreakout v1 recipe, same rows | chronological + purge, no snapshot copies |  | 0.723 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| XGBoost retrained, production features (research) | legacy-style: chrono 80/20, all observations, no purge/dedup | 68 | 0.387 [0.174, 0.677] | 0.538 | 0.413 | 56.5% | -7.09% | 0.0% | 13.09% | -4.54% |
| XGBoost, leakage-safe subset (research) | legacy-style: chrono 80/20, all observations, no purge/dedup | 68 | 0.588 [0.425, 0.870] | 0.625 | 0.272 | 69.6% | -6.45% | 0.0% | 15.29% | -2.94% |
| Logistic regression (research) | legacy-style: chrono 80/20, all observations, no purge/dedup | 68 | 0.472 [0.399, 0.745] | 0.591 | 0.360 | 56.5% | -5.21% | 0.0% | 13.09% | -3.48% |
| Majority-class baseline | walk-forward, purged, 1 folds | 25 | 0.500 [0.500, 0.500] | 0.600 | 0.240 | 87.5% | n/a | n/a | 11.16% | -1.70% |
| Random probability baseline | walk-forward, purged, 1 folds | 25 | 0.467 [0.000, 0.642] | 0.583 | 0.388 | 50.0% | 5.32% | 100.0% | 7.75% | -8.44% |
| HSF Score (production heuristic) | walk-forward, purged, 1 folds | 25 | 0.507 [0.071, 0.583] | 0.621 | 0.243 | 75.0% | -2.67% | 0.0% | 11.16% | -1.91% |
| PreBreakout % as served (production XGBoost) | walk-forward, purged, 0 folds |  | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Logistic regression (research) | walk-forward, purged, 1 folds | 25 | 0.493 [0.000, 0.583] | 0.609 | 0.301 | 50.0% | n/a | n/a | 11.16% | -5.78% |
| XGBoost retrained, production features (research) | walk-forward, purged, 1 folds | 25 | 0.500 [0.500, 0.500] | 0.600 | 0.240 | 87.5% | n/a | n/a | 11.16% | -1.70% |
| XGBoost, leakage-safe subset (research) | walk-forward, purged, 1 folds | 25 | 0.500 [0.500, 0.500] | 0.600 | 0.240 | 87.5% | n/a | n/a | 11.16% | -1.70% |
| AI Confidence (production XGBoost, scored point-in-time) | walk-forward, purged, 0 folds |  | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

AI Confidence rows scored point-in-time: 65 (observed_at > trained_at and all six inputs stored at observation time).

## Fold-level ROC-AUC per model (5-day)

| MODEL | FOLDS | MEAN | MEDIAN | STD | WORST | BEST | ROWS NOT SCORABLE |
|---|---|---|---|---|---|---|---|
| Majority-class baseline | 1 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0 |
| Random probability baseline | 1 | 0.467 | 0.467 | 0.000 | 0.467 | 0.467 | 0 |
| HSF Score (production heuristic) | 1 | 0.507 | 0.507 | 0.000 | 0.507 | 0.507 | 0 |
| PreBreakout % as served (production XGBoost) | 0 | n/a | n/a | n/a | n/a | n/a | 59 |
| Logistic regression (research) | 1 | 0.493 | 0.493 | 0.000 | 0.493 | 0.493 | 0 |
| XGBoost retrained, production features (research) | 1 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0 |
| XGBoost, leakage-safe subset (research) | 1 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0 |
| AI Confidence (production XGBoost, scored point-in-time) | 0 | n/a | n/a | n/a | n/a | n/a | 39 |

## Horizon table (HSF Score heuristic and leakage-safe XGBoost)

### HSF Score (production heuristic)

| HORIZON | N | ROC-AUC | PR-AUC | WIN RATE | MEDIAN RETURN | MEDIAN EXCESS | BEAT RATE | MFE | MAE | STABILITY |
|---|---|---|---|---|---|---|---|---|---|---|
| 1d | 57 | 0.454 [0.328, 0.512] | 0.484 | 42.1% | -0.19% | 0.52% | 57.1% | n/a | n/a | INSUFFICIENT (fewer than 3 evaluable periods) |
| 3d | 26 | 0.414 [0.250, 0.536] | 0.420 | 44.4% | -0.87% | n/a | n/a | n/a | n/a | INSUFFICIENT (fewer than 3 evaluable periods) |
| 5d | 25 | 0.507 [0.071, 0.583] | 0.621 | 75.0% | 7.90% | -2.67% | 0.0% | 11.16% | -1.91% | INSUFFICIENT (fewer than 3 evaluable periods) |
| 10-bar | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | NOT AVAILABLE (no outcome exists) |
| 15-bar | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | NOT AVAILABLE (no outcome exists) |
| 20-bar | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | NOT AVAILABLE (no outcome exists) |

### XGBoost, leakage-safe subset (research)

| HORIZON | N | ROC-AUC | PR-AUC | WIN RATE | MEDIAN RETURN | MEDIAN EXCESS | BEAT RATE | MFE | MAE | STABILITY |
|---|---|---|---|---|---|---|---|---|---|---|
| 1d | 57 | 0.426 [0.343, 0.529] | 0.472 | 26.3% | -0.71% | 0.50% | 60.0% | n/a | n/a | INSUFFICIENT (fewer than 3 evaluable periods) |
| 3d | 26 | 0.500 [0.500, 0.500] | 0.462 | 55.6% | 1.23% | n/a | n/a | n/a | n/a | INSUFFICIENT (fewer than 3 evaluable periods) |
| 5d | 25 | 0.500 [0.500, 0.500] | 0.600 | 87.5% | 9.59% | n/a | n/a | 11.16% | -1.70% | INSUFFICIENT (fewer than 3 evaluable periods) |
| 10-bar | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | NOT AVAILABLE (no outcome exists) |
| 15-bar | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | NOT AVAILABLE (no outcome exists) |
| 20-bar | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | NOT AVAILABLE (no outcome exists) |

### Logistic regression (research)

| HORIZON | N | ROC-AUC | PR-AUC | WIN RATE | MEDIAN RETURN | MEDIAN EXCESS | BEAT RATE | MFE | MAE | STABILITY |
|---|---|---|---|---|---|---|---|---|---|---|
| 1d | 57 | 0.302 [0.260, 0.362] | 0.484 | 31.6% | -0.71% | 0.26% | 50.0% | n/a | n/a | INSUFFICIENT (fewer than 3 evaluable periods) |
| 3d | 26 | 0.470 [0.214, 0.650] | 0.484 | 44.4% | -0.12% | n/a | n/a | n/a | n/a | INSUFFICIENT (fewer than 3 evaluable periods) |
| 5d | 25 | 0.493 [0.000, 0.583] | 0.609 | 50.0% | 0.67% | n/a | n/a | 11.16% | -5.78% | INSUFFICIENT (fewer than 3 evaluable periods) |
| 10-bar | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | NOT AVAILABLE (no outcome exists) |
| 15-bar | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | NOT AVAILABLE (no outcome exists) |
| 20-bar | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | NOT AVAILABLE (no outcome exists) |

## Score table (HSF Score buckets, 5-day, all matured signal-days)

| HSF SCORE | N | WIN RATE [95% CI] | MEDIAN RETURN | MEDIAN EXCESS | BEAT RATE | MFE | MAE |
|---|---|---|---|---|---|---|---|
| 0-49 | 63 | 57.1% [44.9%, 68.6%] | 0.32% | 1.19% | 55.0% | 5.53% | -3.80% |
| 50-59 | 18 | 61.1% [38.6%, 79.7%] | 0.69% | -2.68% | 25.0% | 11.24% | -3.59% |
| 60-69 | 19 | 52.6% [31.7%, 72.7%] | 0.69% | -3.51% | 33.3% | 8.11% | -5.71% |
| 70-79 | 19 | 52.6% [31.7%, 72.7%] | 0.32% | 0.59% | 50.0% | 5.26% | -5.75% |
| 80-89 | 14 | 42.9% [21.4%, 67.4%] | -2.22% | -5.16% | 28.6% | 3.16% | -5.39% |
| 90-100 | 1 | 100.0% [20.7%, 100.0%] | 7.26% | 5.32% | 100.0% | 24.12% | 5.08% |

Monotonicity (buckets with >= 10 matured): {"win_rate": false, "median_return": false, "median_excess_return": false, "benchmark_beat_rate": false, "median_mfe": false}. Inversions: win_rate: 50-59 (0.6111) > 60-69 (0.5263); win_rate: 70-79 (0.5263) > 80-89 (0.4286); median_return: 50-59 (0.006906) > 60-69 (0.006874); median_return: 60-69 (0.006874) > 70-79 (0.003211); median_return: 70-79 (0.003211) > 80-89 (-0.022174); median_excess_return: 0-49 (0.011949) > 60-69 (-0.035142); benchmark_beat_rate: 0-49 (0.55) > 60-69 (0.3333); median_mfe: 50-59 (0.112444) > 60-69 (0.081087); median_mfe: 60-69 (0.081087) > 70-79 (0.052632); median_mfe: 70-79 (0.052632) > 80-89 (0.031624). Spearman(score, 5d return): {"rho": -0.1277, "p_value": 0.1414, "n": 134}.

## Final holdout

Not valid: reserving 108 rows from 2026-09-17 leaves 0 walk-forward folds (need >= 3). No holdout was created.

## Legacy metric reproduction

```json
{
 "ai_confidence_current_recipe": {
  "split_diagnostics": {
   "daily_snapshot_rows": 18003,
   "exact_duplicate_feature_rows": 54742,
   "positive_rate_train": 0.2546,
   "positive_rate_validation": 0.2709,
   "preprocessing": "fillna(0.0) per column (no statistics fitted, so nothing crosses the split); isotonic calibration map fitted on the SAME validation predictions the AUC is reported on",
   "rows": 139013,
   "rows_per_symbol_day": 6.26,
   "split_method": "chronological 80/20 by row Timestamp, no purge, no embargo, no dedup",
   "train_range": [
    "2026-07-10T21:32:48.497387+00:00",
    "2026-09-14T19:36:16.452496+00:00"
   ],
   "train_rows": 111210,
   "train_rows_label_window_overlaps_validation": 18193,
   "validation_range": [
    "2026-09-14T19:36:16.452496+00:00",
    "2026-10-01T21:38:26.656038+00:00"
   ],
   "validation_rows": 27803,
   "validation_rows_identical_to_a_train_row": 88,
   "validation_rows_whose_symbol_day_is_in_train": 1212
  },
  "variants": [
   {
    "roc_auc": 0.5908,
    "train_n": 111210,
    "validation_n": 27803,
    "variant": "legacy_reproduction (chronological 80/20, all rows)"
   },
   {
    "roc_auc": 0.796,
    "train_n": 111210,
    "validation_n": 27803,
    "variant": "random 80/20 split (reference only, NOT a benchmark)"
   },
   {
    "roc_auc": 0.5281,
    "train_n": 90662,
    "validation_n": 27803,
    "variant": "chronological 80/20 + purge (label window) + 1-day embargo"
   },
   {
    "roc_auc": 0.5518,
    "train_n": 11523,
    "validation_n": 6816,
    "variant": "... + one row per symbol/entry day (no daily-snapshot copies)"
   },
   {
    "fold_summary": {
     "best": 0.5821,
     "folds": 5,
     "mean": 0.5431,
     "median": 0.5332,
     "std": 0.0231,
     "worst": 0.5175
    },
    "folds": [
     {
      "fold": 1,
      "roc_auc": 0.5552,
      "train_n": 1267,
      "validation_end": "2026-08-05",
      "validation_n": 3136,
      "validation_start": "2026-07-23"
     },
     {
      "fold": 2,
      "roc_auc": 0.5274,
      "train_n": 4686,
      "validation_end": "2026-08-19",
      "validation_n": 2495,
      "validation_start": "2026-08-06"
     },
     {
      "fold": 3,
      "roc_auc": 0.5175,
      "train_n": 7398,
      "validation_end": "2026-09-02",
      "validation_n": 2591,
      "validation_start": "2026-08-20"
     },
     {
      "fold": 4,
      "roc_auc": 0.5332,
      "train_n": 9946,
      "validation_end": "2026-09-17",
      "validation_n": 3735,
      "validation_start": "2026-09-03"
     },
     {
      "fold": 5,
      "roc_auc": 0.5821,
      "train_n": 12689,
      "validation_end": "2026-10-01",
      "validation_n": 5807,
      "validation_start": "2026-09-18"
     }
    ],
    "roc_auc": 0.5431,
    "train_n": null,
    "validation_n": 17764,
    "variant": "expanding walk-forward (5 folds), deduped, purge + embargo"
   }
  ]
 },
 "registry_legacy_auc": {
  "all_v1_aucs": [
   0.9617857044217968,
   0.9557634094012094,
   0.9361987948998047,
   0.5449126886679385
  ],
  "auc": 0.9617857044217968,
  "model": "prebreakout-xgb-v1",
  "registry_id": 1,
  "trained_at": "2026-06-30 16:27:35.598429+00:00"
 },
 "v1_recipes": {
  "ai_confidence_v1": {
   "chronological_80_20_auc": 0.9787,
   "label": "IsBreakout of the same row",
   "positive_rate": 0.0616,
   "reproduced_random_split_auc": 0.9991,
   "single_feature_auc_BreakoutScore": 0.7357
  },
  "median_minutes_between_scans_of_a_symbol": 38.4,
  "prebreakout_v1": {
   "auc_of_current_IsBreakout_flag_alone": 0.7584,
   "chronological_80_20_auc": 0.7139,
   "chronological_purged_auc": 0.7134,
   "chronological_purged_no_snapshot_copies_auc": 0.7226,
   "label": "IsBreakout in any of the next 3 scan rows",
   "positive_rate": 0.0883,
   "positives_from_cross_symbol_bleed": 451,
   "purged_training_rows": 3357,
   "reproduced_random_split_auc": 0.8873,
   "share_of_positives_already_breakout_now": 0.5327,
   "single_feature_auc_BreakoutScore": 0.6433
  },
  "range": [
   "2026-07-10T21:32:48.497387+00:00",
   "2026-10-08T20:40:33.543486+00:00"
  ],
  "rows": 149416,
  "split_facts": {
   "daily_snapshot_rows": 19669,
   "train_rows": 119532,
   "validation_rows": 29884,
   "validation_start": "2026-09-16T19:49:26.237062+00:00"
  },
  "symbols": 4270
 }
}
```

