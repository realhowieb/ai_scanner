# ML v3: what "the production model" actually is

Sources: the code on main at audit time, plus the model registry tables `prebreakout_models` and
`ai_confidence_models`, read with SELECT only (`ml/reports/ml_v3_model_registry.json`,
`scripts/ml_v3_model_registry.py`).

## The chain from raw features to ranking

```
Raw scanner features (BreakoutScore, Trend10D%, VolRel20, ... from each scan)
    │
    ├──► PreBreakout XGBoost (prebreakout-xgb-v9, 75 features) ──► raw prob ──► isotonic map ──► PreBreakout %
    │                                                                                │
    ├──► BreakoutScore (rule-based scanner score, not ML) ──────────────────────────┤
    │                                                                                ▼
    │                                                    model_component = max(BreakoutScore/60, PreBreakout %/100) × 38
    │                                                                                │
    ├──► signals present (breakout, golden_cross, prebreakout, gapper, gainer) ──► signals_component = 12 each, cap 48
    ├──► chg_pct ──► momentum_component = clamp(chg, 0, 6)/6 × 14
    └──► loser/fading tag ──► fading_penalty = 20
                                                                                     ▼
                         HSF Score = round(clamp(signals + model + momentum − fading, 0, 100))   (HSF_SCORE_VERSION 1.0)
                                                                                     ▼
                         Ranking: top opportunities ordered by HSF Score (ui/opportunities.py); top 5 per snapshot frozen

    └──► AI Confidence XGBoost (ai-confidence-xgb-v1, 6 features) ──► probability column on scan rows.
         It is NOT an input to the HSF Score. It's a separate ranking column.
```

**The HSF Score is not a model probability.** It is a fixed, hand-weighted heuristic that was never trained. Machine
learning reaches it only through the PreBreakout % inside `model_component`, and only when that % is larger
than the scaled BreakoutScore.

## Production artifacts (active rows in the registry)

| | PreBreakout | AI Confidence |
|---|---|---|
| Model type | XGBClassifier (binary:logistic) | XGBClassifier (binary:logistic) |
| Artifact | Neon `prebreakout_models` id 13, `is_active = true` (local fallback `prebreakout_model.pkl`) | Neon `ai_confidence_models` id 3, `is_active = true` (fallback `models/xgb_breakout_model.joblib`) |
| Model version | `prebreakout-xgb-v9`. The code constant says `prebreakout-xgb-v16`; v16 (id 14) was trained but is **not** active | `ai-confidence-xgb-v1` |
| Trained | 2026-09-10 16:46 UTC | 2026-09-09 20:55 UTC |
| Training window | `load_run_history(days_back=90)` scan rows up to the train date | `load_run_history(days_back=90, max_runs=2000)` |
| Rows | 21,694 candidate rows, 3,392 positive | 43,932 rows, 9,878 positive |
| Features | 75 (scanner fields, OHLCV-derived compression/structure/higher-low features, SPY/QQQ regime and relative strength; order = `feature_names` in the registry row) | 6, in this order: Trend10D%, Trend20D%, VolRel20, DollarVol20, BreakoutScore, GapPct |
| Target | `FutureQualitySetupHit`. The row must be a candidate (not a breakout, BreakoutScore < 8, below the 20-day high). The label is 1 if the same symbol produces a high-quality setup 1-3 trading days later **and** that setup then hits +4% before −2% within 5 trading days | `ForwardReturnHit`: +4% before −2% within 5 trading days of the scan date (fallback return_5d ≥ +4%) |
| Label horizon | up to ~8 trading days (3-day lead + 5-day outcome) | 5 trading days |
| Validation | expanding window, 5 folds, purge 5 trading days | one chronological 80/20 split, no purge |
| Reported AUC | 0.668 (fold std 0.023) | 0.580 |
| Hyperparameters | stored in the bundle (Run #17 search); n/a in the registry summary | n_estimators 400, max_depth 5, learning_rate 0.05, subsample 0.9, colsample_bytree 0.9, hist, seed 42 |
| Preprocessing | `_apply_preprocessing_plan` (bundle), then `fillna(0.0)` | `fillna(0.0)` |
| Calibration | isotonic map from validation buckets (`calibration_map`) | isotonic map fitted on the same validation predictions the AUC is reported on |
| Served output | `PreBreakoutProb%` (calibrated × 100) on picks, frozen on opportunity rows as `prebreakout_prob` | `AI Confidence` column on scan rows, **not frozen** anywhere research can read |
| What is recorded | the shown % only. The served model version is never frozen | nothing per observation |

### What the models predict

- **PreBreakout** predicts a future *scanner state* (a quality setup appearing within 3 sessions) combined with a
  price outcome after that setup. It does not predict the observation's own forward return.
- **AI Confidence** predicts a path-based price outcome (+4% before −2%) over 5 sessions.
- **The HSF Score** predicts nothing by construction. It is a confluence ranking, and Outcome Intelligence measures it
  against forward returns (`return_h > 0`, excess vs SPY).

## Findings about the production setup

1. **The headline ~0.96 AUC belongs to retired models.** `prebreakout-xgb-v1` (2026-06-30 to 07-14) reported
   0.962, 0.956 and 0.936. `prebreakout-xgb-v5` (2026-09-10 04:41) reported 0.968 with only 547 positives and was
   replaced 26 minutes later by a run on the same rows with 3,381 positives (AUC 0.592). The July AI Confidence
   models reported 0.998. None of these is active. See the legacy section of `ml_v3_walk_forward_summary.md`.
2. **The PreBreakout purge is shorter than its label.** It purges 5 trading days, but the label reads up to about
   8 trading days ahead (3-day lead + 5-day outcome). Training rows just before a fold boundary can see validation
   prices. This is a small, real leak in the 0.668 figure.
3. **Training reads mutable daily-snapshot rows.** `load_run_history` explodes every `runs` row, including
   `daily_snapshot` copies (also noted by the research-dataset audit), so the same scan rows are counted twice.
4. **AI Confidence's reported AUC uses no purge**, and its calibration map is fitted on the same validation
   predictions the AUC and calibration buckets come from.
5. **The served PreBreakout version is not recorded per observation.** The code constant (v16) differs from the
   active champion (v9). The live input pipeline changed on 2026-10-07/08 (train/serve skew fix), while the
   score version stayed 1.0.
6. **Live skew (earlier runs 25-27, same model):** the stored live scores had AUC 0.52 on 8,742 matured setups,
   against 0.69 with the training pipeline's inputs. This audit's own point-in-time read of the stored % is in the
   walk-forward summary.
