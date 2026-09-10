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
average_precision_score = None  # type: ignore
brier_score_loss = None  # type: ignore
log_loss = None  # type: ignore
XGBClassifier = None  # type: ignore
_ML_IMPORT_TRIED = False


def _load_ml_libs() -> None:
    """Populate the ML globals on first use (no-op when patched or loaded)."""
    global joblib, roc_auc_score, average_precision_score, brier_score_loss, log_loss
    global XGBClassifier, _ML_IMPORT_TRIED
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
    if average_precision_score is None or brier_score_loss is None or log_loss is None:
        try:
            from sklearn.metrics import average_precision_score as _aps
            from sklearn.metrics import brier_score_loss as _bsl
            from sklearn.metrics import log_loss as _ll

            average_precision_score = _aps
            brier_score_loss = _bsl
            log_loss = _ll
        except Exception:  # pragma: no cover - optional ML dependency
            pass
    if XGBClassifier is None:
        try:
            from xgboost import XGBClassifier as _xgb

            XGBClassifier = _xgb
        except Exception:  # pragma: no cover - optional ML dependency
            pass


MODEL_PATH = "prebreakout_model.pkl"
MODEL_VERSION = "prebreakout-xgb-v5"
TARGET_COLUMN = "ForwardReturnHit"
RETURN_COLUMN = "Return_5D"
RETURN_HORIZON_DAYS = 5
UPSIDE_HIT_THRESHOLD = 0.04
DOWNSIDE_STOP_THRESHOLD = -0.02
PREBREAKOUT_TARGET_COLUMN = "FutureQualitySetupHit"
PREBREAKOUT_SETUP_SCORE_THRESHOLD = 8.0
PREBREAKOUT_LEAD_DAYS = 3
BASE_FEATURE_COLS = [
    "Trend10D%",
    "Trend20D%",
    "VolRel20",
    "DollarVol20",
    "BreakoutScore",
    "GapPct",
]
ENGINEERED_HISTORY_COLS = [
    "BreakoutScore",
    "Trend10D%",
    "Trend20D%",
    "VolRel20",
    "GapPct",
    "BreakoutPos20D",
]
ENGINEERED_WINDOWS = (1, 3, 5)
REGIME_FEATURE_COLS = [
    "SPYTrend10D",
    "SPYTrend20D",
    "SPYEMA9EMA21Spread",
    "QQQTrend10D",
    "QQQTrend20D",
    "QQQEMA9EMA21Spread",
    "SPYAboveEMA21",
    "QQQAboveEMA21",
    "SPYVolatility20D",
    "QQQVolatility20D",
    "RSvsSPY10D",
    "RSvsSPY20D",
    "RSvsQQQ10D",
    "RSvsQQQ20D",
    "MarketBreadthAboveEMA21",
]
SPY_REGIME_FEATURE_COLS = [
    "SPYTrend10D",
    "SPYTrend20D",
    "SPYEMA9EMA21Spread",
    "SPYAboveEMA21",
    "SPYVolatility20D",
]
QQQ_REGIME_FEATURE_COLS = [
    "QQQTrend10D",
    "QQQTrend20D",
    "QQQEMA9EMA21Spread",
    "QQQAboveEMA21",
    "QQQVolatility20D",
]
RELATIVE_STRENGTH_FEATURE_COLS = [
    "RSvsSPY10D",
    "RSvsSPY20D",
    "RSvsQQQ10D",
    "RSvsQQQ20D",
]
OPTIONAL_MARKET_CONTEXT_COLS = ["MarketBreadthAboveEMA21"]


def _feature_safe_name(name: str) -> str:
    return "".join(ch for ch in str(name) if ch.isalnum())


def _engineered_feature_names() -> list[str]:
    names = []
    for window in ENGINEERED_WINDOWS:
        names.extend([f"PriceReturn{window}D", f"PriceSlope{window}D"])
    for col in ENGINEERED_HISTORY_COLS:
        safe = _feature_safe_name(col)
        for window in ENGINEERED_WINDOWS:
            names.extend([f"{safe}Delta{window}D", f"{safe}Slope{window}D"])
    return names


ENGINEERED_FEATURE_COLS = _engineered_feature_names()
FEATURE_COLS = BASE_FEATURE_COLS + ENGINEERED_FEATURE_COLS + REGIME_FEATURE_COLS


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


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    try:
        return bool(value) and value == value
    except Exception:
        return False


def _numeric_series(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series([default] * len(frame), index=frame.index)


def _price_below_20d_high(frame: pd.DataFrame) -> pd.Series:
    if "BreakoutPos20D" in frame.columns:
        pos = pd.to_numeric(frame["BreakoutPos20D"], errors="coerce")
        return pos < 0.999
    high = _numeric_series(frame, "High20")
    last_col = "Last" if "Last" in frame.columns else "Close"
    last = _numeric_series(frame, last_col)
    return (high > 0) & (last < 0.999 * high)


def _future_setup_window_days(start, candidate) -> bool:
    start_date = _run_date(start)
    candidate_date = _run_date(candidate)
    if start_date is None or candidate_date is None or candidate_date <= start_date:
        return False
    trading_days = int(np.busday_count(start_date, candidate_date))
    return 1 <= trading_days <= PREBREAKOUT_LEAD_DAYS


def _high_quality_setup_mask(frame: pd.DataFrame) -> pd.Series:
    score = _numeric_series(frame, "BreakoutScore", default=0.0).fillna(0.0)
    breakout = frame.get("IsBreakout", pd.Series([False] * len(frame), index=frame.index)).apply(_as_bool)
    return breakout | (score >= PREBREAKOUT_SETUP_SCORE_THRESHOLD)


def prebreakout_candidate_mask(frame: pd.DataFrame) -> pd.Series:
    """Rows eligible for PreBreakout training: not yet a strong HSF setup."""
    if frame.empty:
        return pd.Series(dtype=bool)
    score = _numeric_series(frame, "BreakoutScore", default=0.0).fillna(0.0)
    breakout = frame.get("IsBreakout", pd.Series([False] * len(frame), index=frame.index)).apply(_as_bool)
    below_high = _price_below_20d_high(frame)
    return (~breakout) & (score < PREBREAKOUT_SETUP_SCORE_THRESHOLD) & below_high.fillna(False)


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

    out = sort_symbol_history(df).reset_index(drop=True).copy()
    out[TARGET_COLUMN] = np.nan

    if RETURN_COLUMN in out.columns:
        out[RETURN_COLUMN] = pd.to_numeric(out[RETURN_COLUMN], errors="coerce")
        out = out[out[RETURN_COLUMN].notna()].reset_index(drop=True)
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


def add_prebreakout_target_label(
    df: pd.DataFrame,
    *,
    lookback_days: int = 90,
    benchmark_context: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Build the PreBreakout objective:
      eligible rows are not broken out, below the alert score threshold, and
      still below the 20-day high.
      y=1 only if the same symbol produces a high-quality setup within the next
      1-3 trading days and that future setup subsequently hits the economic
      ForwardReturnHit target.
    """
    if df.empty:
        return df

    labeled = add_forward_return_labels(df, lookback_days=lookback_days)
    if labeled.empty:
        return labeled

    labeled = sort_symbol_history(labeled).reset_index(drop=True)
    setup_mask = _high_quality_setup_mask(labeled) & (labeled[TARGET_COLUMN].astype(int) == 1)
    candidate_mask = prebreakout_candidate_mask(labeled)
    labeled[PREBREAKOUT_TARGET_COLUMN] = 0

    timestamps = pd.to_datetime(labeled["Timestamp"], errors="coerce", utc=True)
    symbols = labeled["Symbol"].astype(str).str.upper()
    for _, group_idx in labeled.groupby(symbols, sort=False).groups.items():
        ordered_idx = list(group_idx)
        setup_idx = [idx for idx in ordered_idx if bool(setup_mask.loc[idx])]
        if not setup_idx:
            continue
        setup_pos = 0
        for idx in ordered_idx:
            if not bool(candidate_mask.loc[idx]):
                continue
            start_ts = timestamps.loc[idx]
            if pd.isna(start_ts):
                continue
            while setup_pos < len(setup_idx) and timestamps.loc[setup_idx[setup_pos]] <= start_ts:
                setup_pos += 1
            probe = setup_pos
            while probe < len(setup_idx):
                future_ts = timestamps.loc[setup_idx[probe]]
                if pd.isna(future_ts):
                    probe += 1
                    continue
                lead_days = int(np.busday_count(start_ts.date(), future_ts.date()))
                if lead_days > PREBREAKOUT_LEAD_DAYS:
                    break
                if lead_days >= 1:
                    labeled.at[idx, PREBREAKOUT_TARGET_COLUMN] = 1
                    break
                probe += 1

    labeled = add_market_regime_features(labeled, benchmark_context=benchmark_context)
    return labeled.loc[candidate_mask].reset_index(drop=True)


def sort_symbol_history(df: pd.DataFrame) -> pd.DataFrame:
    """Sort each ticker by Symbol + Timestamp for repeatable history features."""
    if df is None or df.empty:
        return df
    sort_cols = [col for col in ["Symbol", "Timestamp"] if col in df.columns]
    if not sort_cols:
        return df
    out = df.copy()
    if "Timestamp" in out.columns:
        out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce", utc=True)
    return out.sort_values(sort_cols, kind="mergesort")


def _above_ema21(row: pd.Series) -> float:
    if row is None:
        return np.nan
    for col in ("AboveEMA21", "AboveEma21", "PriceAboveEMA21", "CloseAboveEMA21"):
        if col in row.index and pd.notna(row.get(col)):
            return 1.0 if bool(row.get(col)) else 0.0
    for col in ("EMA21", "Ema21", "ema21"):
        if col in row.index and pd.notna(row.get(col)):
            price = row.get("Last", row.get("Close"))
            try:
                return 1.0 if float(price) > float(row.get(col)) else 0.0
            except (TypeError, ValueError):
                return np.nan
    return np.nan


def _ema9_ema21_spread(row: pd.Series | None) -> float:
    if row is None:
        return np.nan
    ema9 = None
    ema21 = None
    for col in ("EMA9", "Ema9", "ema9"):
        if col in row.index and pd.notna(row.get(col)):
            ema9 = row.get(col)
            break
    for col in ("EMA21", "Ema21", "ema21"):
        if col in row.index and pd.notna(row.get(col)):
            ema21 = row.get(col)
            break
    if ema9 is None or ema21 is None:
        return np.nan
    price = row.get("Last", row.get("Close", np.nan))
    try:
        price = float(price)
        if price <= 0:
            return np.nan
        return (float(ema9) - float(ema21)) / price * 100.0
    except (TypeError, ValueError):
        return np.nan


def _numeric_group_column(group: pd.DataFrame, name: str) -> pd.Series:
    if name not in group.columns:
        return pd.Series(np.nan, index=group.index, dtype=float)
    return pd.to_numeric(group[name], errors="coerce")


def _all_benchmark_context(df: pd.DataFrame, symbol: str, prefix: str) -> pd.DataFrame:
    if "Symbol" not in df.columns or "Timestamp" not in df.columns:
        return pd.DataFrame()
    symbols = df["Symbol"].astype(str).str.upper()
    rows = df.loc[symbols == symbol].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["Timestamp"] = pd.to_datetime(rows["Timestamp"], errors="coerce", utc=True)
    rows = rows.dropna(subset=["Timestamp"]).sort_values("Timestamp", kind="mergesort")
    if rows.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Timestamp": rows["Timestamp"],
            f"{prefix}Trend10D": _numeric_group_column(rows, "Trend10D%"),
            f"{prefix}Trend20D": _numeric_group_column(rows, "Trend20D%"),
            f"{prefix}EMA9EMA21Spread": rows.apply(_ema9_ema21_spread, axis=1),
            f"{prefix}AboveEMA21": rows.apply(_above_ema21, axis=1),
            f"{prefix}Volatility20D": _numeric_group_column(rows, "Volatility20D%"),
        }
    ).drop_duplicates(subset=["Timestamp"], keep="last")


def _benchmark_context_from_bars(bars: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if bars is None or bars.empty or "Close" not in bars.columns:
        return pd.DataFrame()
    frame = bars.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce", utc=True)
    frame = frame.loc[frame.index.notna()].sort_index()
    closes = pd.to_numeric(frame["Close"], errors="coerce")
    if closes.dropna().empty:
        return pd.DataFrame()
    ema9 = closes.ewm(span=9, adjust=False).mean()
    ema21 = closes.ewm(span=21, adjust=False).mean()
    returns = closes.pct_change()
    context = pd.DataFrame(
        {
            "Timestamp": frame.index.normalize() + pd.Timedelta(1, unit="D"),
            f"{prefix}Trend10D": (closes / closes.shift(10) - 1.0) * 100.0,
            f"{prefix}Trend20D": (closes / closes.shift(20) - 1.0) * 100.0,
            f"{prefix}EMA9EMA21Spread": (ema9 - ema21) / closes * 100.0,
            f"{prefix}AboveEMA21": (closes > ema21).astype(float),
            f"{prefix}Volatility20D": returns.rolling(20).std() * 100.0,
        }
    )
    return context.dropna(subset=["Timestamp"]).drop_duplicates(subset=["Timestamp"], keep="last")


def load_benchmark_regime_context(days_back: int) -> dict[str, pd.DataFrame]:
    """Fetch SPY/QQQ daily bars through the existing provider for training-time context."""
    try:
        from data.price_alpaca import download_multi_alpaca

        bars = download_multi_alpaca(
            ["SPY", "QQQ"],
            period=f"{int(days_back) + 70}d",
            interval="1d",
            prepost=False,
            timeout_s=20,
        )
    except Exception as e:
        print(f"[ml_prebreakout] Benchmark regime fallback unavailable: {e}")
        return {}
    return {
        "SPY": _benchmark_context_from_bars(bars.get("SPY"), "SPY") if bars.get("SPY") is not None else pd.DataFrame(),
        "QQQ": _benchmark_context_from_bars(bars.get("QQQ"), "QQQ") if bars.get("QQQ") is not None else pd.DataFrame(),
    }


def _merge_benchmark_asof(out: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    if benchmark.empty or "Timestamp" not in out.columns:
        return out
    order_col = "__regime_original_order"
    left = out.copy()
    left[order_col] = np.arange(len(left))
    left["Timestamp"] = pd.to_datetime(left["Timestamp"], errors="coerce", utc=True)
    left_sorted = left.sort_values("Timestamp", kind="mergesort")
    right_sorted = benchmark.copy()
    right_sorted["Timestamp"] = pd.to_datetime(right_sorted["Timestamp"], errors="coerce", utc=True)
    right_sorted = right_sorted.dropna(subset=["Timestamp"]).sort_values("Timestamp", kind="mergesort")
    merged = pd.merge_asof(left_sorted, right_sorted, on="Timestamp", direction="backward")
    return merged.sort_values(order_col, kind="mergesort").drop(columns=[order_col])


def _market_breadth_above_ema21(group: pd.DataFrame) -> float:
    if "Symbol" not in group.columns:
        return np.nan
    symbols = group["Symbol"].astype(str).str.upper()
    stock_group = group.loc[~symbols.isin(["SPY", "QQQ"])]
    if stock_group.empty:
        return np.nan
    values = stock_group.apply(_above_ema21, axis=1).dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def add_market_regime_features(
    df: pd.DataFrame,
    benchmark_context: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Stamp each row with benchmark context available at or before its timestamp."""
    if df is None or df.empty:
        return df

    out = df.copy()
    if "Timestamp" not in out.columns:
        for col in REGIME_FEATURE_COLS:
            if col not in out.columns:
                out[col] = 0.0
        return out

    out = out.drop(columns=[col for col in REGIME_FEATURE_COLS if col in out.columns])
    timestamps = pd.to_datetime(out["Timestamp"], errors="coerce", utc=True)
    out["Timestamp"] = timestamps
    for benchmark, prefix in (("SPY", "SPY"), ("QQQ", "QQQ")):
        context = _all_benchmark_context(out, benchmark, prefix)
        if context.empty and benchmark_context:
            context = benchmark_context.get(benchmark, pd.DataFrame())
        out = _merge_benchmark_asof(out, context)

    for benchmark, prefix in (("SPY", "SPY"), ("QQQ", "QQQ")):
        trend10 = f"{prefix}Trend10D"
        trend20 = f"{prefix}Trend20D"
        if trend10 in out.columns and "Trend10D%" in out.columns:
            out[f"RSvs{benchmark}10D"] = pd.to_numeric(out["Trend10D%"], errors="coerce") - pd.to_numeric(
                out[trend10], errors="coerce"
            )
        if trend20 in out.columns and "Trend20D%" in out.columns:
            out[f"RSvs{benchmark}20D"] = pd.to_numeric(out["Trend20D%"], errors="coerce") - pd.to_numeric(
                out[trend20], errors="coerce"
            )

    out["__regime_ts"] = timestamps
    for _, group in out.groupby(out["__regime_ts"], sort=False, dropna=False):
        breadth = _market_breadth_above_ema21(group)
        if pd.notna(breadth):
            out.loc[group.index, "MarketBreadthAboveEMA21"] = breadth

    for col in REGIME_FEATURE_COLS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out.drop(columns=["__regime_ts"])


def _spark_price_return(value, window: int):
    if not isinstance(value, (list, tuple)) or len(value) <= window:
        return np.nan
    try:
        start = float(value[-(window + 1)])
        end = float(value[-1])
        if start <= 0:
            return np.nan
        return (end - start) / start
    except (TypeError, ValueError):
        return np.nan


def add_prebreakout_features(
    df: pd.DataFrame,
    benchmark_context: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Add 1/3/5-day slopes and deltas without changing the target."""
    if df is None or df.empty:
        return df

    out = df.copy()
    order_col = "__prebreakout_original_order"
    out[order_col] = np.arange(len(out))
    out = sort_symbol_history(out)
    if any(col not in out.columns for col in REGIME_FEATURE_COLS):
        out = add_market_regime_features(out, benchmark_context=benchmark_context)
    if "Spark10D" in out.columns:
        for window in ENGINEERED_WINDOWS:
            ret_col = f"PriceReturn{window}D"
            slope_col = f"PriceSlope{window}D"
            out[ret_col] = out["Spark10D"].apply(lambda value, w=window: _spark_price_return(value, w))
            out[slope_col] = out[ret_col] / float(window)
    else:
        price_col = "Last" if "Last" in out.columns else "Close"
        if price_col in out.columns and "Symbol" in out.columns:
            prices = pd.to_numeric(out[price_col], errors="coerce")
            symbols = out["Symbol"].astype(str).str.upper()
            for window in ENGINEERED_WINDOWS:
                prior = prices.groupby(symbols, sort=False).shift(window)
                ret_col = f"PriceReturn{window}D"
                slope_col = f"PriceSlope{window}D"
                out[ret_col] = np.where(prior > 0, (prices - prior) / prior, np.nan)
                out[slope_col] = out[ret_col] / float(window)

    if "Symbol" not in out.columns:
        for name in ENGINEERED_FEATURE_COLS:
            if name not in out.columns:
                out[name] = 0.0
        return out.sort_values(order_col, kind="mergesort").drop(columns=[order_col])

    symbols = out["Symbol"].astype(str).str.upper()
    for col in ENGINEERED_HISTORY_COLS:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        safe = _feature_safe_name(col)
        for window in ENGINEERED_WINDOWS:
            prior = values.groupby(symbols, sort=False).shift(window)
            delta_col = f"{safe}Delta{window}D"
            slope_col = f"{safe}Slope{window}D"
            out[delta_col] = values - prior
            out[slope_col] = out[delta_col] / float(window)

    for name in ENGINEERED_FEATURE_COLS:
        if name not in out.columns:
            out[name] = 0.0
    return out.sort_values(order_col, kind="mergesort").drop(columns=[order_col])


def build_ml_dataset(df: pd.DataFrame, benchmark_context: dict[str, pd.DataFrame] | None = None):
    """
    Select feature columns and target column.
    """
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    df = add_prebreakout_features(df, benchmark_context=benchmark_context)
    feature_cols = FEATURE_COLS
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    X = X.fillna(0.0)

    if PREBREAKOUT_TARGET_COLUMN in df.columns:
        y = df[PREBREAKOUT_TARGET_COLUMN].astype(int)
    elif TARGET_COLUMN in df.columns:
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


def _chronological_order(df_labeled: pd.DataFrame, index) -> list:
    if "Timestamp" not in df_labeled.columns:
        return list(index)
    timestamps = pd.to_datetime(df_labeled["Timestamp"], errors="coerce", utc=True)
    symbols = (
        df_labeled["Symbol"].astype(str).str.upper()
        if "Symbol" in df_labeled.columns
        else pd.Series([""] * len(df_labeled), index=df_labeled.index)
    )
    ordered = pd.DataFrame({"Timestamp": timestamps, "Symbol": symbols}, index=df_labeled.index)
    ordered = ordered.loc[[idx for idx in ordered.index if idx in index]]
    ordered = ordered.sort_values(["Timestamp", "Symbol"], kind="mergesort")
    return list(ordered.index)


def expanding_window_folds(
    X: pd.DataFrame,
    y: pd.Series,
    df_labeled: pd.DataFrame,
    *,
    n_splits: int = 5,
    purge_days: int = 5,
) -> list[dict]:
    """Build chronological expanding-window folds with a trading-day purge before validation."""
    if X.empty or len(X) < n_splits + 1:
        return []
    ordered_index = _chronological_order(df_labeled, X.index)
    if len(ordered_index) < n_splits + 1 or "Timestamp" not in df_labeled.columns:
        return []

    timestamps = pd.to_datetime(df_labeled["Timestamp"], errors="coerce", utc=True)
    ordered_index = [idx for idx in ordered_index if idx in timestamps.index and pd.notna(timestamps.loc[idx])]
    if len(ordered_index) < n_splits + 1:
        return []

    boundaries = np.linspace(0, len(ordered_index), n_splits + 2, dtype=int)
    folds = []
    for fold_num in range(1, n_splits + 1):
        test_start_pos = int(boundaries[fold_num])
        test_end_pos = int(boundaries[fold_num + 1])
        if test_start_pos <= 0 or test_end_pos <= test_start_pos:
            continue

        test_idx = ordered_index[test_start_pos:test_end_pos]
        test_start_ts = timestamps.loc[test_idx[0]]
        train_candidates = ordered_index[:test_start_pos]
        train_idx = [
            idx
            for idx in train_candidates
            if int(np.busday_count(timestamps.loc[idx].date(), test_start_ts.date())) > int(purge_days)
        ]
        if not train_idx or not test_idx:
            continue

        folds.append(
            {
                "fold": fold_num,
                "train_idx": train_idx,
                "val_idx": test_idx,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(test_idx)),
                "validation_start": test_start_ts.isoformat().replace("+00:00", "Z"),
                "validation_end": timestamps.loc[test_idx[-1]].isoformat().replace("+00:00", "Z"),
                "purge_days": int(purge_days),
            }
        )
    return folds


def _top_decile_hit_rate(y_true, y_proba) -> tuple[float, int]:
    frame = pd.DataFrame(
        {
            "actual": pd.Series(y_true).reset_index(drop=True).astype(float),
            "predicted": pd.Series(y_proba).reset_index(drop=True).astype(float),
        }
    ).dropna()
    if frame.empty:
        return 0.0, 0
    top_n = max(1, int(np.ceil(len(frame) * 0.10)))
    top = frame.sort_values("predicted", ascending=False).head(top_n)
    return float(top["actual"].mean()), int(len(top))


def classification_diagnostics(y_true, y_proba) -> dict:
    """Validation metrics for rare-event model credibility."""
    actual = pd.Series(y_true).reset_index(drop=True).astype(int)
    predicted = pd.Series(y_proba).reset_index(drop=True).astype(float).clip(1e-9, 1.0 - 1e-9)
    baseline = float(actual.mean()) if len(actual) else 0.0
    top_hit_rate, top_n = _top_decile_hit_rate(actual, predicted)
    metrics = {
        "baseline_hit_rate": baseline,
        "top_10pct_hit_rate": top_hit_rate,
        "top_10pct_n": top_n,
        "lift_over_baseline": float(top_hit_rate / baseline) if baseline > 0 else 0.0,
    }
    if actual.nunique(dropna=True) >= 2:
        metrics["auc"] = float(roc_auc_score(actual, predicted))
        metrics["pr_auc"] = float(average_precision_score(actual, predicted))
        metrics["log_loss"] = float(log_loss(actual, predicted, labels=[0, 1]))
    else:
        metrics["auc"] = None
        metrics["pr_auc"] = None
        metrics["log_loss"] = None
    metrics["brier_score"] = float(brier_score_loss(actual, predicted)) if len(actual) else None
    return metrics


def summarize_fold_metrics(fold_metrics: list[dict]) -> dict:
    summary = {}
    metric_names = ["auc", "pr_auc", "brier_score", "log_loss", "top_10pct_hit_rate", "lift_over_baseline"]
    for name in metric_names:
        values = [float(row[name]) for row in fold_metrics if row.get(name) is not None]
        if values:
            summary[f"{name}_mean"] = float(np.mean(values))
            summary[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return summary


def _new_prebreakout_classifier():
    return XGBClassifier(
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


RUN6_BASELINE_MEAN_AUC = 0.5819816609896494
RUN6_BASELINE_STD_AUC = 0.043


def prebreakout_feature_sets(all_features: list[str]) -> list[dict]:
    base = [col for col in all_features if col not in REGIME_FEATURE_COLS]
    spy = base + [col for col in SPY_REGIME_FEATURE_COLS if col in all_features]
    spy_qqq = spy + [col for col in QQQ_REGIME_FEATURE_COLS if col in all_features]
    spy_qqq_rs = spy_qqq + [col for col in RELATIVE_STRENGTH_FEATURE_COLS if col in all_features]
    return [
        {"name": "Baseline", "features": base, "market_features": []},
        {"name": "+SPY", "features": spy, "market_features": [col for col in SPY_REGIME_FEATURE_COLS if col in spy]},
        {
            "name": "+SPY+QQQ",
            "features": spy_qqq,
            "market_features": [col for col in SPY_REGIME_FEATURE_COLS + QQQ_REGIME_FEATURE_COLS if col in spy_qqq],
        },
        {
            "name": "+SPY+QQQ+RelativeStr",
            "features": spy_qqq_rs,
            "market_features": [
                col
                for col in SPY_REGIME_FEATURE_COLS + QQQ_REGIME_FEATURE_COLS + RELATIVE_STRENGTH_FEATURE_COLS
                if col in spy_qqq_rs
            ],
        },
    ]


def _date_range_for_indices(df_labeled: pd.DataFrame, indices: list) -> tuple[str | None, str | None]:
    if "Timestamp" not in df_labeled.columns or not indices:
        return None, None
    timestamps = pd.to_datetime(df_labeled.loc[indices, "Timestamp"], errors="coerce", utc=True).dropna()
    if timestamps.empty:
        return None, None
    return (
        timestamps.min().isoformat().replace("+00:00", "Z"),
        timestamps.max().isoformat().replace("+00:00", "Z"),
    )


def evaluate_prebreakout_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    df_labeled: pd.DataFrame,
    folds: list[dict],
    feature_cols: list[str],
) -> dict:
    fold_metrics = []
    validation_proba = []
    validation_actual = []
    for fold in folds:
        X_train = X.loc[fold["train_idx"], feature_cols]
        X_val = X.loc[fold["val_idx"], feature_cols]
        y_train = y.loc[fold["train_idx"]]
        y_val = y.loc[fold["val_idx"]]
        if y_train.nunique(dropna=True) < 2 or y_val.nunique(dropna=True) < 2:
            continue

        fold_clf = _new_prebreakout_classifier()
        fold_clf.fit(X_train, y_train)
        y_proba = fold_clf.predict_proba(X_val)[:, 1]
        metrics = classification_diagnostics(y_val, y_proba)
        train_start, train_end = _date_range_for_indices(df_labeled, fold["train_idx"])
        val_start, val_end = _date_range_for_indices(df_labeled, fold["val_idx"])
        metrics.update(
            {
                "fold": int(fold["fold"]),
                "train_rows": int(len(X_train)),
                "validation_rows": int(len(X_val)),
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": val_start or fold["validation_start"],
                "validation_end": val_end or fold["validation_end"],
                "purge_days": int(fold["purge_days"]),
                "positive_validation_rows": int(y_val.sum()),
                "positive_rate": float(y_val.mean()),
            }
        )
        fold_metrics.append(metrics)
        validation_proba.extend([float(value) for value in y_proba])
        validation_actual.extend([int(value) for value in y_val])

    return {
        "fold_metrics": fold_metrics,
        "validation_summary": summarize_fold_metrics(fold_metrics),
        "validation_proba": validation_proba,
        "validation_actual": validation_actual,
    }


def _fmt_metric(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _print_validation_report(name: str, fold_metrics: list[dict], summary: dict, validation_rows: int) -> None:
    print(f"[ml_prebreakout] {name}")
    for metrics in fold_metrics:
        print(
            "[ml_prebreakout] Fold "
            f"{metrics['fold']}: train_rows={metrics['train_rows']}, "
            f"validation_rows={metrics['validation_rows']}, "
            f"train_range={metrics.get('train_start')}..{metrics.get('train_end')}, "
            f"validation_range={metrics.get('validation_start')}..{metrics.get('validation_end')}, "
            f"positive_rate={_fmt_metric(metrics.get('positive_rate'))}, "
            f"ROC-AUC={_fmt_metric(metrics.get('auc'))}, "
            f"PR-AUC={_fmt_metric(metrics.get('pr_auc'))}, "
            f"Brier={_fmt_metric(metrics.get('brier_score'))}, "
            f"LogLoss={_fmt_metric(metrics.get('log_loss'))}, "
            f"Top10Hit={_fmt_metric(metrics.get('top_10pct_hit_rate'))}, "
            f"Top10Lift={_fmt_metric(metrics.get('lift_over_baseline'), 2)}x"
        )
    print("[ml_prebreakout] PREBREAKOUT MARKET-REGIME CHALLENGER")
    print(f"[ml_prebreakout] Mean ROC-AUC: {_fmt_metric(summary.get('auc_mean'), 6)}")
    print(f"[ml_prebreakout] Std ROC-AUC: {_fmt_metric(summary.get('auc_std'), 6)}")
    print(f"[ml_prebreakout] Mean PR-AUC: {_fmt_metric(summary.get('pr_auc_mean'), 6)}")
    print(f"[ml_prebreakout] Mean Brier: {_fmt_metric(summary.get('brier_score_mean'), 6)}")
    print(f"[ml_prebreakout] Mean LogLoss: {_fmt_metric(summary.get('log_loss_mean'), 6)}")
    print(f"[ml_prebreakout] Mean Top10Hit: {_fmt_metric(summary.get('top_10pct_hit_rate_mean'), 6)}")
    print(f"[ml_prebreakout] Mean Top10Lift: {_fmt_metric(summary.get('lift_over_baseline_mean'), 6)}")
    print(f"[ml_prebreakout] OOF rows: {validation_rows}")


def _print_baseline_comparison(summary: dict, fold_metrics: list[dict], same_run_baseline_auc: float | None = None) -> str:
    challenger_auc = summary.get("auc_mean")
    challenger_std = summary.get("auc_std")
    weakest = min(fold_metrics, key=lambda row: row.get("auc") if row.get("auc") is not None else float("inf"))
    best = max(fold_metrics, key=lambda row: row.get("auc") if row.get("auc") is not None else float("-inf"))
    promote = bool(
        challenger_auc is not None
        and float(challenger_auc) > RUN6_BASELINE_MEAN_AUC
        and (same_run_baseline_auc is None or float(challenger_auc) >= float(same_run_baseline_auc))
    )
    result = "PROMOTE" if promote else "REJECT"
    print("[ml_prebreakout] BASELINE COMPARISON")
    print(f"[ml_prebreakout] Run #6 Mean AUC: {RUN6_BASELINE_MEAN_AUC}")
    print(f"[ml_prebreakout] Challenger Mean AUC: {_fmt_metric(challenger_auc, 6)}")
    delta_auc = float(challenger_auc) - RUN6_BASELINE_MEAN_AUC if challenger_auc is not None else None
    print(f"[ml_prebreakout] Delta AUC: {_fmt_metric(delta_auc, 6)}")
    print(f"[ml_prebreakout] Run #6 Std AUC: ~{RUN6_BASELINE_STD_AUC}")
    print(f"[ml_prebreakout] Challenger Std AUC: {_fmt_metric(challenger_std, 6)}")
    delta_std = float(challenger_std) - RUN6_BASELINE_STD_AUC if challenger_std is not None else None
    print(f"[ml_prebreakout] Delta Std: {_fmt_metric(delta_std, 6)}")
    print(f"[ml_prebreakout] Weakest challenger fold: Fold {weakest['fold']} AUC={_fmt_metric(weakest.get('auc'))}")
    print(f"[ml_prebreakout] Best challenger fold: Fold {best['fold']} AUC={_fmt_metric(best.get('auc'))}")
    print(f"[ml_prebreakout] MARKET REGIME RESULT: {result}")
    return result


def _print_feature_ablation(ablation: list[dict]) -> None:
    print("[ml_prebreakout] FEATURE ABLATION")
    for row in ablation:
        summary = row.get("validation_summary") or {}
        print(
            f"[ml_prebreakout] {row['name']:<24} "
            f"AUC={_fmt_metric(summary.get('auc_mean'), 6)} "
            f"Top10Lift={_fmt_metric(summary.get('lift_over_baseline_mean'), 6)}"
        )


def _feature_importance_rows(model, feature_cols: list[str], top_n: int = 20) -> list[dict]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    rows = []
    for name, importance in zip(feature_cols, list(importances)):
        rows.append(
            {
                "feature": str(name),
                "importance": float(importance),
                "is_market_regime_feature": str(name) in REGIME_FEATURE_COLS,
            }
        )
    return sorted(rows, key=lambda row: row["importance"], reverse=True)[:top_n]


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

    X = add_prebreakout_features(df.copy())
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
    if (
        joblib is None
        or roc_auc_score is None
        or average_precision_score is None
        or brier_score_loss is None
        or log_loss is None
        or XGBClassifier is None
    ):
        print(
            "[ml_prebreakout] ML dependencies are not installed. "
            "Install requirements-ml.txt to train the prebreakout model."
        )
        return {}

    df = load_run_history(days_back=days_back)
    if df.empty:
        print("[ml_prebreakout] No history data found.")
        return {}

    benchmark_context = load_benchmark_regime_context(days_back)
    benchmark_context_source = {
        symbol: "alpaca_daily_prior_completed_bar" if not context.empty else "missing"
        for symbol, context in benchmark_context.items()
    }
    df_labeled = add_prebreakout_target_label(
        df,
        lookback_days=days_back,
        benchmark_context=benchmark_context,
    )
    if df_labeled.empty:
        print("[ml_prebreakout] No eligible prebreakout rows with complete labels available.")
        return {}

    X, y = build_ml_dataset(df_labeled, benchmark_context=benchmark_context)
    if X.empty:
        print("[ml_prebreakout] No features available.")
        return {}
    if y.nunique(dropna=True) < 2:
        print("[ml_prebreakout] PreBreakout labels have only one class; cannot train AUC model.")
        return {}

    folds = expanding_window_folds(X, y, df_labeled, n_splits=5, purge_days=RETURN_HORIZON_DAYS)
    if not folds:
        print("[ml_prebreakout] No valid expanding-window validation folds available.")
        return {}

    ablation_results = []
    for feature_set in prebreakout_feature_sets(list(X.columns)):
        features = [col for col in feature_set["features"] if col in X.columns]
        if not features:
            continue
        evaluation = evaluate_prebreakout_feature_set(X, y, df_labeled, folds, features)
        if not evaluation["fold_metrics"]:
            continue
        evaluation.update(
            {
                "name": feature_set["name"],
                "features": features,
                "market_features": feature_set["market_features"],
            }
        )
        ablation_results.append(evaluation)

    if not ablation_results:
        print("[ml_prebreakout] Expanding-window folds have only one class in train or validation.")
        return {}

    _print_feature_ablation(ablation_results)
    baseline_eval = next((row for row in ablation_results if row["name"] == "Baseline"), None)
    market_candidates = [row for row in ablation_results if row["name"] != "Baseline"]
    if not baseline_eval or not market_candidates:
        print("[ml_prebreakout] Missing baseline or market ablation results.")
        return {}

    best_market_eval = max(
        market_candidates,
        key=lambda row: row["validation_summary"].get("auc_mean", float("-inf")),
    )
    baseline_auc = baseline_eval["validation_summary"].get("auc_mean")
    best_market_auc = best_market_eval["validation_summary"].get("auc_mean")
    selected_eval = (
        best_market_eval
        if best_market_auc is not None and baseline_auc is not None and float(best_market_auc) >= float(baseline_auc)
        else baseline_eval
    )

    fold_metrics = best_market_eval["fold_metrics"]
    validation_summary = best_market_eval["validation_summary"]
    auc = validation_summary.get("auc_mean")
    if auc is None:
        print("[ml_prebreakout] Expanding-window validation could not compute AUC.")
        return {}

    selected_features = selected_eval["features"]
    clf = _new_prebreakout_classifier()
    clf.fit(X[selected_features], y)

    calibration = confidence_bucket_diagnostics(best_market_eval["validation_actual"], best_market_eval["validation_proba"])
    validation_rows = int(sum(row["validation_rows"] for row in fold_metrics))
    _print_validation_report("Best market-regime feature set: " + best_market_eval["name"], fold_metrics, validation_summary, validation_rows)
    market_regime_result = _print_baseline_comparison(validation_summary, fold_metrics, baseline_auc)

    feature_importances = _feature_importance_rows(clf, selected_features)
    if feature_importances:
        print("[ml_prebreakout] TOP 20 FEATURE IMPORTANCES")
        for row in feature_importances:
            tag = " market-regime" if row["is_market_regime_feature"] else ""
            print(f"[ml_prebreakout] {row['feature']}: {row['importance']:.6f}{tag}")

    bundle = {
        "model": clf,
        "features": list(selected_features),
        "feature_names": list(selected_features),
        "market_feature_names": list(selected_eval["market_features"]),
        "trained_at": _utc_now().isoformat().replace("+00:00", "Z"),
        "auc": float(auc),
        "validation_method": "expanding_window_5fold_purged",
        "validation_metrics": validation_summary,
        "validation_folds": fold_metrics,
        "fold_metrics": fold_metrics,
        "validation_summary": validation_summary,
        "feature_ablation": [
            {
                "name": row["name"],
                "features": row["features"],
                "market_features": row["market_features"],
                "validation_summary": row["validation_summary"],
            }
            for row in ablation_results
        ],
        "selected_feature_set": selected_eval["name"],
        "best_market_feature_set": best_market_eval["name"],
        "market_regime_result": market_regime_result,
        "baseline_comparison": {
            "run6_mean_auc": RUN6_BASELINE_MEAN_AUC,
            "run6_std_auc": RUN6_BASELINE_STD_AUC,
            "challenger_mean_auc": validation_summary.get("auc_mean"),
            "challenger_std_auc": validation_summary.get("auc_std"),
            "same_run_baseline_mean_auc": baseline_auc,
        },
        "feature_importances": feature_importances,
        "target": PREBREAKOUT_TARGET_COLUMN,
        "target_rule": (
            "Eligible rows are not broken out, BreakoutScore < 8, and below the 20-day high; "
            "positive when a high-quality setup appears within 1-3 trading days and then hits "
            "the +4% before -2% economic target."
        ),
        "candidate_rule": "IsBreakout is false, BreakoutScore < 8, price below 20-day high",
        "lead_days": PREBREAKOUT_LEAD_DAYS,
        "setup_score_threshold": PREBREAKOUT_SETUP_SCORE_THRESHOLD,
        "return_column": RETURN_COLUMN,
        "calibration": calibration,
        "rows": int(len(X)),
        "training_rows": int(len(X)),
        "positive_rows": int(y.sum()),
        "validation_rows": validation_rows,
        "benchmark_context_source": benchmark_context_source,
        "model_version": MODEL_VERSION,
        "source": "local",
    }

    if save_prebreakout_model is not None and serialize_model_to_bytes is not None:
        try:
            saved = save_prebreakout_model(
                model_bytes=serialize_model_to_bytes(clf, joblib),
                feature_names=list(selected_features),
                auc=float(auc),
                trained_at=str(bundle["trained_at"]),
                model_version=MODEL_VERSION,
                metadata={
                    key: value
                    for key, value in bundle.items()
                    if key not in {"model"}
                },
            )
            if saved:
                bundle["source"] = "database"
        except Exception as e:
            bundle["db_save_error"] = str(e)
            print(f"[ml_prebreakout] DB model save failed: {e}")

    joblib.dump(bundle, model_path)
    print(f"[ml_prebreakout] Saved XGBoost model to {model_path}")
    return bundle
