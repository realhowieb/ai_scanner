# HSF ML v4: multi-model engine runbook

ML v4 trains XGBoost, LightGBM and CatBoost challengers on the same finalized research dataset, evaluates them leakage-safely, tracks them in a model registry and tests them forward in shadow mode. **Production AI Confidence is unchanged**: it keeps using `ai_confidence_models`. Wiring a registry champion into AI Confidence is a separate, explicit decision that ML v4 does not make.

## Pieces

| What | Where |
|---|---|
| Providers, targets, feature sets, artifacts | `analytics/ml_v4_providers.py` |
| Walk-forward evaluation (purge + embargo, holdout once) | `analytics/ml_v4_evaluation.py` |
| Calibration (sigmoid / isotonic, earlier folds only) | `analytics/ml_v4_calibration.py` |
| Ensembles (equal, weighted, best, stacking) | `analytics/ml_v4_ensemble.py` |
| Lifecycle rules | `analytics/ml_v4_lifecycle.py` |
| Registry tables `ml_models`, `ml_model_events` | `db/ml_model_registry.py` |
| Shadow scoring, summaries, promotion gates | `analytics/ml_v4_shadow.py`, `scripts/ml_v4_shadow.py` |
| Shadow table `ml_shadow_predictions` (180 days) | `db/ml_shadow_predictions.py` |
| Monitoring | `analytics/ml_v4_monitoring.py` |
| Admin API `/v1/admin/ml/*` | `api/routes/ml_models.py` |
| Admin page | Admin → Models (`web/src/features/AdminModels.tsx`) |
| Readiness check (read-only) | `scripts/ml_v4_readiness_audit.py` (Diagnostics workflow) |

Target: **+4% before −2% within 5 trading days** (`F_clean_path_5d`, an order-free stricter proxy: MFE ≥ +4% and MAE > −2%). Secondary: beat SPY and return > 0 at 1, 3 and 5 days.

## Lifecycle

`candidate → validated → shadow → champion → retired` (any stage can be retired; a retired former champion can return only by rollback). Every change records who, why and the evidence.

- **validated** needs the evaluation stored on the model (registration with an evaluation report).
- **champion** needs the promotion gates to pass on forward shadow predictions, plus explicit approval in Admin → Models. Promoting retires the previous champion for that target and horizon.
- **rollback** (`POST /v1/admin/ml/rollback`, approve=true) restores the most recent former champion.

## Promotion gates (`hsf-ml-v4-gates-2`)

All must pass: evaluation attached; final holdout scored; currently in shadow; ≥ 3 walk-forward folds with no fold AUC below 0.5; ≥ 200 matured forward shadow predictions over ≥ 20 trading days; on ≥ 100 of the same rows as AI Confidence as served, AUC at least as good and Brier no worse; ECE ≤ 0.10; per-prediction scoring time p95 ≤ 100 ms. Thresholds live in `analytics/ml_v4_shadow.GATES`.

## Step by step

1. **Wait for data.** Admin → Research → ML v4 readiness must show READY (expected around 2026-11-13).
2. **Finalize a dataset** (writes to production; needs Howard's go): `scripts/research_dataset.py --finalize`.
3. **Evaluate (read-only):** run *ML v4 Train & Evaluate* with mode=evaluate. Read the report in the run summary.
4. **Train + register:** mode=train, register=true. Uploads model files to a new prerelease `ml-v4-models-<run>` and registers candidates with their evaluation.
5. **Validate** a candidate in Admin → Models, then **start its shadow test**.
6. **Shadow** runs weekdays 21:40 UTC (*ML v4 Shadow*). Watch Admin → Models → Monitoring and the run summary.
7. **Champion** only when every gate passes, with approval. This changes the registry only.

## Switches

GitHub repository variables (Settings → Secrets and variables → Actions → Variables), unset = on:

- `HSF_ML_V4_SHADOW_ENABLED=0`: stop all shadow scoring.
- `HSF_ML_V4_SHADOW_DISABLED=catboost,<model version>,<id>`: skip those models.

Retiring a model in Admin → Models also stops its shadow scoring.

## Monitoring checks

Shadow job freshness (2 trading days), overdue labels (10 trading days), probability drift (PSI vs the first shadow weeks: 0.10 watch, 0.25 warning), recent calibration (ECE 0.10 watch, 0.15 warning), recent AUC vs AI Confidence (watch below −0.05), feature schema match (the shadow job refuses to score a model whose features changed).

## Safety rules

- Model files are GitHub Release assets with sha256; the checksum is verified before loading. The API never returns their location and never imports the training libraries.
- Shadow scoring writes only `ml_shadow_predictions`; it never changes AI Confidence, alerts or PreBreakout.
- No live trading or order execution.
- No performance claims without a real evaluation run on a finalized dataset.
