import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from db.runs import list_runs, load_run_results

try:
    from db.prebreakout_models import (
        load_latest_prebreakout_model_bundle,
        save_prebreakout_model,
        serialize_model_to_bytes,
    )
except Exception:  # pragma: no cover - keeps ML imports resilient in partial deploys
    load_latest_prebreakout_model_bundle = None  # type: ignore[assignment]
    save_prebreakout_model = None  # type: ignore[assignment]
    serialize_model_to_bytes = None  # type: ignore[assignment]

try:
    import joblib
except Exception:  # pragma: no cover - optional ML dependency
    joblib = None  # type: ignore

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover - optional ML dependency
    train_test_split = None  # type: ignore
    roc_auc_score = None  # type: ignore

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional ML dependency
    XGBClassifier = None  # type: ignore


MODEL_PATH = "prebreakout_model.pkl"
MODEL_VERSION = "prebreakout-xgb-v1"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_datetime(value) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def load_run_history(days_back: int = 90, max_runs: int = 2000) -> pd.DataFrame:
    """
    Load past runs from the runs DB and explode results into a ticker-level DataFrame.

    Uses:
      - db.runs.list_runs(limit=...) for run metadata
      - db.runs.load_run_results(run_id) to fetch the results_json payload

    The returned DataFrame will have one row per (Symbol, run_time) with columns such as:
      Symbol, Timestamp, run_label, run_time, IsBreakout, BreakoutScore, GapPct,
      VolRel20, DollarVol20, Trend10D%, Trend20D%, Last, etc. (where available).
    """
    try:
        runs = list_runs(limit=max_runs)
    except Exception as e:
        print(f"[ml_prebreakout] Failed to load runs from DB: {e}")
        return pd.DataFrame()

    records: list[dict] = []
    cutoff = _utc_now() - timedelta(days=days_back)

    for run in runs:
        run_id = run.get("id")
        created_at = run.get("created_at") or run.get("timestamp")

        created_at = _normalize_datetime(created_at)

        if created_at is None or created_at < cutoff:
            continue

        label = run.get("label", "")

        # Fetch full results_json for this run
        try:
            results_json = load_run_results(run_id) if run_id is not None else None
        except Exception as e:
            print(f"[ml_prebreakout] Failed to load results for run {run_id}: {e}")
            continue

        if not results_json:
            continue

        # Parse JSON string or accept already-parsed list
        try:
            rows = results_json
            if isinstance(results_json, str):
                rows = json.loads(results_json)
        except Exception as e:
            print(f"[ml_prebreakout] Malformed results_json for run {run_id}: {e}")
            continue

        if not isinstance(rows, list):
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue

            # Support both "Symbol" and "Ticker" keys
            sym = str(row.get("Symbol") or row.get("Ticker") or "").strip().upper()
            if not sym:
                continue

            rec = dict(row)
            rec["Symbol"] = sym
            rec["run_label"] = label
            rec["run_time"] = created_at
            # Keep a consistent timestamp column name for labeling logic
            rec["Timestamp"] = created_at
            records.append(rec)

    if not records:
        print("[ml_prebreakout] No records built from run history.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Ensure required columns exist (even if empty), so downstream code doesn't break
    for col in [
        "IsBreakout",
        "BreakoutScore",
        "GapPct",
        "VolRel20",
        "DollarVol20",
        "Trend10D%",
        "Trend20D%",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values(["Symbol", "Timestamp"]).reset_index(drop=True)
    print(f"[ml_prebreakout] load_run_history built {len(df)} rows from {len(runs)} runs.")
    return df


def add_future_breakout_label(df: pd.DataFrame, horizon_scans: int = 3) -> pd.DataFrame:
    """
    Mark each row y=1 if the same symbol has a breakout within the next horizon_scans rows.
    Assumes df sorted by (Symbol, Timestamp).
    """
    if df.empty:
        return df

    df = df.sort_values(["Symbol", "Timestamp"]).reset_index(drop=True)
    df["FutureBreakout"] = 0

    for i in range(len(df)):
        sym = df.at[i, "Symbol"]
        end_i = min(i + horizon_scans, len(df) - 1)
        future_slice = df.iloc[i+1:end_i+1]
        if (future_slice["Symbol"] == sym).any() and (future_slice["IsBreakout"] == 1).any():
            df.at[i, "FutureBreakout"] = 1

    return df


def build_ml_dataset(df: pd.DataFrame):
    """
    Select feature columns and target column.
    """
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    feature_cols = [
        "Trend10D%", "Trend20D%", "VolRel20", "DollarVol20",
        "BreakoutScore", "GapPct",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    X = X.fillna(0.0)

    y = df.get("FutureBreakout", pd.Series([0] * len(df)))

    return X, y


# Process-level model cache: every scan was re-downloading the model BYTEA from
# Neon and re-deserializing it (plus xgboost's old-pickle conversion), adding
# seconds per scan. The model changes at most daily, so cache for 15 minutes.
_MODEL_CACHE: dict = {}
_MODEL_CACHE_TTL_S = 900


def load_prebreakout_model(model_path: str = MODEL_PATH):
    """
    Load model bundle with model, features, trained_at, auc.

    Database is the durable source of truth. Local file loading is retained as
    a best-effort fallback only, so app reboots do not require retraining when
    a saved database model exists. Cached in-process for 15 minutes.
    """
    if joblib is None:
        return None
    import time as _time

    cached = _MODEL_CACHE.get("prebreakout")
    if cached and (_time.time() - cached[0]) < _MODEL_CACHE_TTL_S:
        return cached[1]
    if load_latest_prebreakout_model_bundle is not None:
        try:
            bundle = load_latest_prebreakout_model_bundle(joblib)
            _MODEL_CACHE["prebreakout"] = (_time.time(), bundle or None)
            return bundle or None
        except Exception as e:
            print(f"[ml_prebreakout] DB model load failed: {e}")
    try:
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict):
            bundle.setdefault("source", "local")
        _MODEL_CACHE["prebreakout"] = (_time.time(), bundle)
        return bundle
    except Exception:
        _MODEL_CACHE["prebreakout"] = (_time.time(), None)
        return None


def score_prebreakout(df: pd.DataFrame, model_path: str = MODEL_PATH) -> pd.DataFrame:
    """
    Add PreBreakoutProb and PreBreakoutProb% columns using trained XGBoost model.
    """
    if df is None or df.empty:
        return df

    bundle = load_prebreakout_model(model_path)
    if not bundle:
        df["PreBreakoutProb"] = 0.0
        df["PreBreakoutProb%"] = 0.0
        return df

    model = bundle["model"]
    feature_cols = bundle["features"]

    X = df.copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols].fillna(0.0)

    proba = model.predict_proba(X)[:, 1]
    df["PreBreakoutProb"] = proba
    df["PreBreakoutProb%"] = (proba * 100.0).round(1)
    return df


def train_prebreakout_model(
    days_back: int = 30,
    horizon_scans: int = 3,
    model_path: str = MODEL_PATH,
):
    """
    Train an XGBoost model to predict future breakout likelihood.
    Saves a bundle containing model, features, trained_at, and auc.
    """
    if joblib is None or train_test_split is None or roc_auc_score is None or XGBClassifier is None:
        print(
            "[ml_prebreakout] ML dependencies are not installed. "
            "Install requirements-ml.txt to train the prebreakout model."
        )
        return {}

    df = load_run_history(days_back=days_back)
    if df.empty:
        print("[ml_prebreakout] No history data found.")
        return {}

    if "IsBreakout" not in df.columns:
        print("[ml_prebreakout] History missing 'IsBreakout'; cannot label.")
        return {}

    df_labeled = add_future_breakout_label(df, horizon_scans=horizon_scans)
    X, y = build_ml_dataset(df_labeled)
    if X.empty:
        print("[ml_prebreakout] No features available.")
        return {}

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_proba)
    print(f"[ml_prebreakout] XGBoost Validation AUC: {auc:.3f}")

    bundle = {
        "model": clf,
        "features": list(X.columns),
        "trained_at": _utc_now().isoformat().replace("+00:00", "Z"),
        "auc": float(auc),
        "model_version": MODEL_VERSION,
        "source": "local",
    }

    if save_prebreakout_model is not None and serialize_model_to_bytes is not None:
        try:
            saved = save_prebreakout_model(
                model_bytes=serialize_model_to_bytes(clf, joblib),
                feature_names=list(X.columns),
                auc=float(auc),
                trained_at=str(bundle["trained_at"]),
                model_version=MODEL_VERSION,
            )
            if saved:
                bundle["source"] = "database"
        except Exception as e:
            bundle["db_save_error"] = str(e)
            print(f"[ml_prebreakout] DB model save failed: {e}")

    joblib.dump(bundle, model_path)
    print(f"[ml_prebreakout] Saved XGBoost model to {model_path}")
    return bundle
