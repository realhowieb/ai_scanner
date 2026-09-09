import json
from datetime import datetime, timedelta, timezone
from statistics import mean

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

# ML libraries are loaded LAZILY: this module sits on app.py's boot import
# chain (via prebreakout_tab / three_step_scanner), and importing xgboost +
# scikit-learn eagerly added hundreds of MB to boot memory — the standing
# suspect for the Streamlit Cloud startup segfaults. Placeholders stay module
# attributes so tests can patch them (patch.object(ml_prebreakout, "joblib"));
# _load_ml_libs() fills any that are still None on first use.
# If imports fail they remain None: Install requirements-ml.txt to enable.
joblib = None  # type: ignore
roc_auc_score = None  # type: ignore
XGBClassifier = None  # type: ignore
_ML_IMPORT_TRIED = False


def _load_ml_libs() -> None:
    """Populate the ML globals on first use (no-op when patched or loaded)."""
    global joblib, roc_auc_score, XGBClassifier, _ML_IMPORT_TRIED
    if _ML_IMPORT_TRIED:
        return
    _ML_IMPORT_TRIED = True
    if joblib is None:
        try:
            import joblib as _joblib

            joblib = _joblib
        except Exception:  # pragma: no cover - optional ML dependency
            pass
    if roc_auc_score is None:
        try:
            from sklearn.metrics import roc_auc_score as _ras

            roc_auc_score = _ras
        except Exception:  # pragma: no cover - optional ML dependency
            pass
    if XGBClassifier is None:
        try:
            from xgboost import XGBClassifier as _xgb

            XGBClassifier = _xgb
        except Exception:  # pragma: no cover - optional ML dependency
            pass


MODEL_PATH = "prebreakout_model.pkl"
MODEL_VERSION = "prebreakout-xgb-v1"
TARGET_COLUMN = "ForwardReturnHit"
RETURN_COLUMN = "Return_5D"
RETURN_HORIZON_DAYS = 5
UPSIDE_HIT_THRESHOLD = 0.04
DOWNSIDE_STOP_THRESHOLD = -0.02


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


def _run_date(value):
    ts = _normalize_datetime(value)
    return ts.date() if ts is not None else None


def _forward_path_hit(
    bars,
    run_date,
    horizon_days: int,
    hit_threshold: float = UPSIDE_HIT_THRESHOLD,
    stop_threshold: float = DOWNSIDE_STOP_THRESHOLD,
) -> bool | None:
    """True when +4% is reached before -2%; None when path data is incomplete."""
    try:
        if "Close" not in getattr(bars, "columns", []):
            return None
        closes = bars["Close"].dropna()
        if closes.empty:
            return None
        entry_pos = None
        for pos, ts in enumerate(closes.index):
            d = ts.date() if hasattr(ts, "date") else ts
            if d >= run_date:
                entry_pos = pos
                break
        if entry_pos is None:
            return None
        end_pos = entry_pos + int(horizon_days)
        if end_pos >= len(closes):
            return None
        entry = float(closes.iloc[entry_pos])
        if entry <= 0:
            return None

        highs = bars["High"].reindex(closes.index) if "High" in bars.columns else closes
        lows = bars["Low"].reindex(closes.index) if "Low" in bars.columns else closes
        for pos in range(entry_pos + 1, end_pos + 1):
            high_ret = (float(highs.iloc[pos]) - entry) / entry
            low_ret = (float(lows.iloc[pos]) - entry) / entry
            if low_ret <= float(stop_threshold):
                return False
            if high_ret >= float(hit_threshold):
                return True
        return False
    except Exception:
        return None


def _download_label_bars(symbols: list[str], lookback_days: int):
    try:
        from data.price_alpaca import download_multi_alpaca

        return download_multi_alpaca(
            symbols,
            period=f"{max(RETURN_HORIZON_DAYS + 15, lookback_days + RETURN_HORIZON_DAYS + 10)}d",
            interval="1d",
            prepost=False,
            timeout_s=25.0,
        )
    except Exception as e:
        print(f"[ml_prebreakout] Failed to download label bars: {e}")
        return {}


def add_forward_return_labels(
    df: pd.DataFrame,
    horizon_days: int = RETURN_HORIZON_DAYS,
    hit_threshold: float = UPSIDE_HIT_THRESHOLD,
    stop_threshold: float = DOWNSIDE_STOP_THRESHOLD,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """
    Add money-based labels:
      - Return_5D: forward close-to-close return over horizon_days.
      - ForwardReturnHit: 1 when +4% is reached before -2% within horizon_days.

    If full path data is unavailable, the hit label falls back to Return_5D >= +4%.
    """
    if df.empty:
        return df

    out = df.sort_values(["Symbol", "Timestamp"]).reset_index(drop=True).copy()
    out[TARGET_COLUMN] = np.nan

    if RETURN_COLUMN in out.columns:
        out[RETURN_COLUMN] = pd.to_numeric(out[RETURN_COLUMN], errors="coerce")
        out[TARGET_COLUMN] = (out[RETURN_COLUMN] >= float(hit_threshold)).astype(int)
        return out
    out[RETURN_COLUMN] = np.nan

    symbols = sorted({str(s).upper() for s in out.get("Symbol", pd.Series(dtype=str)).dropna() if str(s).strip()})
    bars_by_symbol = _download_label_bars(symbols, lookback_days=lookback_days) if symbols else {}
    if not bars_by_symbol:
        print("[ml_prebreakout] No price bars available for forward-return labels.")
        return out.iloc[0:0].copy()

    try:
        from analytics.track_record import _bars_for, _forward_return
    except Exception:
        _bars_for = lambda bars, sym: bars.get(sym)  # type: ignore[assignment]
        _forward_return = None  # type: ignore[assignment]

    keep = []
    for idx, row in out.iterrows():
        run_date = _run_date(row.get("Timestamp") or row.get("run_time"))
        sym = str(row.get("Symbol") or "").upper()
        bars = _bars_for(bars_by_symbol, sym)
        if run_date is None or bars is None or _forward_return is None:
            continue
        ret = _forward_return(bars, run_date, int(horizon_days), entry_mode="close")
        if ret is None:
            continue
        path_hit = _forward_path_hit(
            bars,
            run_date,
            int(horizon_days),
            hit_threshold=float(hit_threshold),
            stop_threshold=float(stop_threshold),
        )
        out.at[idx, RETURN_COLUMN] = float(ret)
        out.at[idx, TARGET_COLUMN] = int(path_hit if path_hit is not None else ret >= float(hit_threshold))
        keep.append(idx)

    labeled = out.loc[keep].reset_index(drop=True)
    if labeled.empty:
        print("[ml_prebreakout] No rows had complete forward-return labels.")
    return labeled


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

    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].astype(int)
    elif RETURN_COLUMN in df.columns:
        y = (pd.to_numeric(df[RETURN_COLUMN], errors="coerce").fillna(0.0) >= UPSIDE_HIT_THRESHOLD).astype(int)
    else:
        y = df.get("FutureBreakout", pd.Series([0] * len(df))).astype(int)

    return X, y


def walk_forward_split(X: pd.DataFrame, y: pd.Series, df_labeled: pd.DataFrame, validation_fraction: float = 0.2):
    """Chronological split that validates on later scans, matching live usage."""
    if X.empty or len(X) < 2:
        return X, X, y, y
    if "Timestamp" in df_labeled.columns:
        order = pd.to_datetime(df_labeled["Timestamp"], errors="coerce", utc=True).sort_values().index
    else:
        order = X.index
    ordered_index = [i for i in order if i in X.index]
    if len(ordered_index) < 2:
        ordered_index = list(X.index)
    split_at = max(1, int(len(ordered_index) * (1.0 - validation_fraction)))
    if split_at >= len(ordered_index):
        split_at = len(ordered_index) - 1
    train_idx = ordered_index[:split_at]
    val_idx = ordered_index[split_at:]
    return X.loc[train_idx], X.loc[val_idx], y.loc[train_idx], y.loc[val_idx]


def confidence_bucket_diagnostics(y_true, y_proba, bucket_size: float = 0.1) -> list[dict]:
    """Calibration-style hit rate by confidence bucket."""
    rows = []
    frame = pd.DataFrame(
        {
            "actual": pd.Series(y_true).reset_index(drop=True).astype(float),
            "predicted": pd.Series(y_proba).reset_index(drop=True).astype(float),
        }
    )
    frame = frame.dropna()
    if frame.empty:
        return rows
    edges = np.arange(0.0, 1.0 + bucket_size, bucket_size)
    for low, high in zip(edges[:-1], edges[1:]):
        if high >= 1.0:
            bucket = frame[(frame["predicted"] >= low) & (frame["predicted"] <= high)]
        else:
            bucket = frame[(frame["predicted"] >= low) & (frame["predicted"] < high)]
        if bucket.empty:
            continue
        rows.append(
            {
                "bucket": f"{int(low * 100)}-{int(high * 100)}%",
                "n": int(len(bucket)),
                "mean_confidence": float(mean(bucket["predicted"].tolist())),
                "hit_rate": float(mean(bucket["actual"].tolist())),
            }
        )
    return rows


# Process-level model cache: every scan was re-downloading the model BYTEA from
# Neon and re-deserializing it (plus xgboost's old-pickle conversion), adding
# seconds per scan. The model changes at most daily, so cache for 15 minutes.
_MODEL_CACHE: dict = {}
_MODEL_CACHE_TTL_S = 900


def clear_model_cache() -> None:
    """Drop the cached model bundle (tests; or force a reload after retrain)."""
    _MODEL_CACHE.clear()


def load_prebreakout_model(model_path: str = MODEL_PATH):
    """
    Load model bundle with model, features, trained_at, auc.

    Database is the durable source of truth. Local file loading is retained as
    a best-effort fallback only, so app reboots do not require retraining when
    a saved database model exists. Cached in-process for 15 minutes.
    """
    _load_ml_libs()
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
    _load_ml_libs()
    if joblib is None or roc_auc_score is None or XGBClassifier is None:
        print(
            "[ml_prebreakout] ML dependencies are not installed. "
            "Install requirements-ml.txt to train the prebreakout model."
        )
        return {}

    df = load_run_history(days_back=days_back)
    if df.empty:
        print("[ml_prebreakout] No history data found.")
        return {}

    df_labeled = add_forward_return_labels(
        df,
        horizon_days=RETURN_HORIZON_DAYS,
        hit_threshold=UPSIDE_HIT_THRESHOLD,
        stop_threshold=DOWNSIDE_STOP_THRESHOLD,
        lookback_days=days_back,
    )
    if df_labeled.empty:
        print("[ml_prebreakout] No complete forward-return labels available.")
        return {}

    X, y = build_ml_dataset(df_labeled)
    if X.empty:
        print("[ml_prebreakout] No features available.")
        return {}
    if y.nunique(dropna=True) < 2:
        print("[ml_prebreakout] Forward-return labels have only one class; cannot train AUC model.")
        return {}

    X_train, X_val, y_train, y_val = walk_forward_split(X, y, df_labeled)
    if y_train.nunique(dropna=True) < 2 or y_val.nunique(dropna=True) < 2:
        print("[ml_prebreakout] Walk-forward split has only one class in train or validation.")
        return {}

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
    calibration = confidence_bucket_diagnostics(y_val, y_proba)
    print(f"[ml_prebreakout] XGBoost Walk-forward AUC: {auc:.3f}")

    bundle = {
        "model": clf,
        "features": list(X.columns),
        "trained_at": _utc_now().isoformat().replace("+00:00", "Z"),
        "auc": float(auc),
        "validation_method": "walk_forward",
        "target": TARGET_COLUMN,
        "target_rule": "+4% before -2% in 5 trading days; fallback Return_5D >= +4%",
        "return_column": RETURN_COLUMN,
        "calibration": calibration,
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
