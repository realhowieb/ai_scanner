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

## Preparation safeguards (2026-11-04)

Start with **ML v4 Train & Evaluate → mode=preflight**. It is now the default.
Use an explicit finalized dataset version for a reviewed run. The workflow resolves
an empty version to newest once, then pins that same version for training and evaluation.
It never finalizes a dataset automatically.

Every CLI entry point rebuilds and verifies the frozen dataset fingerprint and exact
membership. A mismatch stops the run; it does not silently accept a newer reconstruction.
Finalize a fresh, reviewed dataset if historical inputs can no longer be reproduced.
Preflight audits that selected population at its freeze time, then also checks the
actual `F_clean_path_5d` target: both outcome classes, at least 20% minority class,
a valid 100-row / five-day final test, and at least three chronological validation
periods before it. A green generic return-based readiness report alone is insufficient.
The code commit must be supplied (`GITHUB_SHA`, or `--git-sha` locally).

The target means **reaches +4% and never touches −2% anywhere in five trading days**.
It does not establish which price was reached first. No thresholds were relaxed.

### What the preparation report shows

`ml_v4_preflight.md` and `.json` show READY/BLOCKED, reasons, dataset identity,
target definition, positive/negative examples, training dates, validation-period
count, and reserved final-test dates/counts. Actions publishes it even on blocked
runs. Training stops before fitting anything when preparation fails. Neither the
preflight nor training mode writes to production.

### Development versus final evaluation

- `mode=evaluate`, `final_holdout=false` (default): repeatable development reports.
  The primary final-test observations are reserved and excluded from fitting and
  validation. Secondary targets use only dates before that final-test boundary.
- `final_holdout=true`: an explicit database write. Before predicting any final-test
  outcome, the run commits a claim to the additive `ml_holdout_evaluations` table.
  Claims serialize per target; overlapping final-test observation IDs are refused.
  The table is created only by this explicitly requested operation, never by a read.
- A completed report is reused exactly when the holdout and evaluation binding match.
  A different code commit, settings, dataset, library versions, cost assumption or
  frozen AI inputs cannot silently reuse it as new evidence. Model artifacts can be
  regenerated under the same binding without consuming the final test again.
- An interrupted/failed final evaluation leaves its claim consumed. It cannot retry
  automatically. Do not delete/reset its claim to get another look at the outcomes;
  inspect the failure and arrange a genuinely unseen final test with owner approval.
- `register=true` requires `mode=train` and `final_holdout=true`. Evaluation attachment
  checks the provider, feature set, target, dataset fingerprint/version, training
  slice, settings, code and libraries. Registration verifies the supplied report
  against the ledger's completed report before writing any model rows.

The existing post-shadow promotion gates still apply. This work does not train,
register, promote or replace any production model by itself.

### Offline rehearsal

Install the pinned ML requirements, then run:

```bash
python scripts/ml_v4_rehearsal.py --out /tmp/hsf-ml-rehearsal
```

Use an empty output directory. The rehearsal uses deterministic fictional data,
small versions of all three real estimators, and a local SQLite final-test ledger.
It exercises preparation, training, development evaluation, final evaluation,
exact report reuse, registration dry-run checks, artifact reload and 30 shadow
predictions. It never connects to Neon, GitHub or a market-data provider. Its scores
are not evidence about real trading performance. The `ml-v4-models` CI job now runs
this rehearsal alongside the unit tests.
