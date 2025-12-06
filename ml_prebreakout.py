import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from db.runs import list_runs


MODEL_PATH = "prebreakout_model.pkl"


def load_run_history(days_back: int = 90, max_runs: int = 2000) -> pd.DataFrame:
    """
    Load past runs from the runs DB and explode results into a ticker-level DataFrame.

    Assumes db.runs.list_runs(limit=...) returns an iterable of dicts with at least:
      - created_at (ISO string or datetime)
      - label (optional)
      - results_json or results (JSON array of rows from your scanner)

    The returned DataFrame will have one row per (Symbol, run_time) with columns such as:
      Symbol, run_time, IsBreakout, BreakoutScore, GapPct, VolRel20,
      DollarVol20, Trend10D%, Trend20D%, Last, etc. (where available).
    """
    try:
        runs = list_runs(limit=max_runs)
    except Exception as e:
        print(f"[ml_prebreakout] Failed to load runs from DB: {e}")
        return pd.DataFrame()

    records: list[dict] = []
    cutoff = datetime.utcnow() - timedelta(days=days_back)

    for run in runs:
        # Created_at / timestamp parsing
        created_at = run.get("created_at") or run.get("timestamp")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                created_at = None

        if created_at is None or created_at < cutoff:
            continue

        label = run.get("label", "")
        results_json = run.get("results_json") or run.get("results")
        if not results_json:
            continue

        try:
            rows = results_json
            # If stored as JSON string, parse it
            if isinstance(results_json, str):
                import json

                rows = json.loads(results_json)
        except Exception:
            # Skip malformed results
            continue

        for row in rows:
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


def load_prebreakout_model(model_path: str = MODEL_PATH):
    """
    Load model bundle with model, features, trained_at, auc.
    """
    try:
        return joblib.load(model_path)
    except Exception:
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
    days_back: int = 90,
    horizon_scans: int = 3,
    model_path: str = MODEL_PATH,
):
    """
    Train an XGBoost model to predict future breakout likelihood.
    Saves a bundle containing model, features, trained_at, and auc.
    """
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
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "auc": float(auc),
    }
    joblib.dump(bundle, model_path)
    print(f"[ml_prebreakout] Saved XGBoost model to {model_path}")
    return bundle
