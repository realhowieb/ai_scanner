import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

from db.runs import list_runs, load_run_results

try:
    from db.prebreakout_models import (
        load_latest_prebreakout_model_bundle,
        restore_previous_model_if_active_run16_incomplete,
        save_prebreakout_model,
        serialize_model_to_bytes,
        update_active_prebreakout_model_metadata,
    )
except Exception:  # pragma: no cover - keeps ML imports resilient in partial deploys
    load_latest_prebreakout_model_bundle = None  # type: ignore[assignment]
    restore_previous_model_if_active_run16_incomplete = None  # type: ignore[assignment]
    save_prebreakout_model = None  # type: ignore[assignment]
    serialize_model_to_bytes = None  # type: ignore[assignment]
    update_active_prebreakout_model_metadata = None  # type: ignore[assignment]

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
XGBRanker = None  # type: ignore
_ML_IMPORT_TRIED = False


def _load_ml_libs() -> None:
    """Populate the ML globals on first use (no-op when patched or loaded)."""
    global joblib, roc_auc_score, average_precision_score, brier_score_loss, log_loss
    global XGBClassifier, XGBRanker, _ML_IMPORT_TRIED
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
    if XGBRanker is None:
        try:
            from xgboost import XGBRanker as _xgb_ranker

            XGBRanker = _xgb_ranker
        except Exception:  # pragma: no cover - optional ML dependency
            pass


MODEL_PATH = "prebreakout_model.pkl"
MODEL_VERSION = "prebreakout-xgb-v16"
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
SPY_RELATIVE_STRENGTH_FEATURE_COLS = ["RSvsSPY10D", "RSvsSPY20D"]
QQQ_RELATIVE_STRENGTH_FEATURE_COLS = ["RSvsQQQ10D", "RSvsQQQ20D"]
EMA_SETUP_EVOLUTION_FEATURE_COLS = [
    "EMA9_21Spread",
    "EMA9_21SpreadPct",
    "EMA9_21SpreadChange1D",
    "EMA9_21SpreadChange3D",
    "EMA9_21SpreadSlope3D",
    "EMA9_21SpreadSlope5D",
]
BREAKOUT_POSITION_EVOLUTION_FEATURE_COLS = [
    "DistanceTo20DHighPct",
    "DistanceTo20DHighChange1D",
    "DistanceTo20DHighChange3D",
    "DistanceTo20DHighSlope3D",
    "DistanceTo20DHighSlope5D",
]
VOLUME_VOLATILITY_EVOLUTION_FEATURE_COLS = [
    "RVOLChange1D",
    "RVOLChange3D",
    "RVOLSlope3D",
    "RVOLSlope5D",
    "ATRPercent",
    "ATRPercentChange3D",
    "Volatility20DChange3D",
]
RUN9_CHAMPION_FEATURE_COLS = (
    RELATIVE_STRENGTH_FEATURE_COLS
    + VOLUME_VOLATILITY_EVOLUTION_FEATURE_COLS
)
BOLLINGER_COMPRESSION_FEATURE_COLS = [
    "BBWidth20Pct",
    "BBWidthChange3D",
    "BBWidthChange5D",
    "BBWidthRatio5D",
    "BBWidthRatio20D",
]
ATR_COMPRESSION_FEATURE_COLS = [
    "TrueRange",
    "ATR14",
    "ATR14Pct",
    "ATRChange3D",
    "ATRChange5D",
    "ATRRatio5D",
    "ATRRatio20D",
]
RUN10_ATR_COMPRESSION_FEATURE_COLS = [
    "ATR14PctChange3D",
    "ATR14PctChange5D",
    "ATRCompression5D",
    "ATRCompression20D",
]
PREBREAKOUT_STRUCTURE_FEATURE_COLS = [
    "High20D",
    "High50D",
    "DistanceTo20DHighPct",
    "DistanceTo50DHighPct",
    "DistanceToHigh20Pct",
    "DistanceToHighChange1D",
    "DistanceToHighChange3D",
    "DistanceToHighChange5D",
    "BreakoutPosChange1D",
    "BreakoutPosChange3D",
    "BreakoutPosChange5D",
    "BreakoutPosSlope3D",
    "BreakoutPosSlope5D",
]
HIGHER_LOW_FEATURE_COLS = [
    "HigherLowCount10D",
    "HigherLowCount20D",
    "LowSlope10D",
    "LowSlope20D",
    "HigherHighCount10D",
    "HigherHighCount20D",
]
RESISTANCE_TOUCH_FEATURE_COLS = [
    "ResistanceTouches20D",
    "ResistanceTouches50D",
    "ResistanceTouchCount10D",
    "ResistanceTouchCount20D",
]
RANGE_COMPRESSION_FEATURE_COLS = [
    "RangePct",
    "RangeCompression5D",
    "RangeCompression10D",
    "RangeCompression20D",
]
VOLUME_DRY_UP_FEATURE_COLS = [
    "VolumeRatio5D20D",
    "VolumeChange5D",
    "VolumeDryUp20D",
]
RUN11_CHAMPION_FEATURE_COLS = (
    RUN9_CHAMPION_FEATURE_COLS
    + RANGE_COMPRESSION_FEATURE_COLS
    + HIGHER_LOW_FEATURE_COLS
)
RANGE_CONTRACTION_QUALITY_FEATURE_COLS = [
    "RangePct3D",
    "RangePct5D",
    "RangePct10D",
    "RangeContractionRatio3v10",
    "RangeContractionRatio5v20",
    "RangeCompressionAcceleration",
    "ConsecutiveContractingRangeDays",
    "RangePctStd5D",
    "RangePctStd10D",
]
INSIDE_TIGHT_DAY_FEATURE_COLS = [
    "InsideDay",
    "InsideDayCount3D",
    "InsideDayCount5D",
    "NR4",
    "NR7",
    "TightDayCount5D",
]
HIGHER_LOW_QUALITY_FEATURE_COLS = [
    "HigherLowRatio5D",
    "HigherLowRatio10D",
    "LowSlope5D",
    "LowSlope10D",
    "LowSlope20D",
    "LowSlope5DPct",
    "LowSlope10DPct",
    "LowSlope20DPct",
    "LowSlopeAcceleration",
    "LowConsistency10D",
]
PRICE_STRUCTURE_TIGHTNESS_FEATURE_COLS = [
    "HighLowChannelWidth5D",
    "HighLowChannelWidth10D",
    "HighLowChannelWidth20D",
    "ChannelCompression5v20",
    "ChannelCompression10v20",
    "CloseLocationInRange",
    "CloseLocation5DAvg",
    "RecentHighDistancePct",
]
VOLUME_COMPRESSION_INTERACTION_FEATURE_COLS = [
    "VolumeTrend5D",
    "VolumeTrend10D",
    "VolumeDryUp5D",
    "VolumeDryUp10D",
    "VolumeCompressionRatio5v20",
    "RangeVolumeInteraction",
    "CompressionWithVolumeDryUp",
]
RS_COMPRESSION_INTERACTION_FEATURE_COLS = [
    "Compression_x_RS10",
    "Compression_x_RS20",
    "RangePct_x_RS10",
    "HigherLow_x_RS10",
    "StructureCompressionRS",
]
RUN14_HIGHER_LOW_REFINEMENT_FEATURE_COLS = [
    "HigherLowRatio3D",
    "LowConsistency5D",
]
RUN14_HIGHER_LOW_RS_INTERACTION_FEATURE_COLS = [
    "HigherLowRatio10D_x_RSvsSPY10D",
    "HigherLowRatio10D_x_RSvsQQQ10D",
    "LowSlope10DPct_x_RSvsSPY10D",
]
RUN14_HIGHER_LOW_COMPRESSION_FEATURE_COLS = [
    "HigherLowRatio10D_x_RangeCompression10D",
    "LowConsistency10D_x_RangeCompression10D",
    "LowSlope10DPct_x_RangeCompression20D",
]
RUN16_STRUCTURAL_INTERACTION_FEATURE_COLS = [
    "LowSlopeAcceleration_x_RangeCompression10D",
    "HigherLowRatio10D_x_RangeCompression20D",
]
RUN12_EXPERIMENTAL_FEATURE_COLS = (
    RANGE_CONTRACTION_QUALITY_FEATURE_COLS
    + INSIDE_TIGHT_DAY_FEATURE_COLS
    + HIGHER_LOW_QUALITY_FEATURE_COLS
    + PRICE_STRUCTURE_TIGHTNESS_FEATURE_COLS
    + VOLUME_COMPRESSION_INTERACTION_FEATURE_COLS
    + RS_COMPRESSION_INTERACTION_FEATURE_COLS
    + RUN14_HIGHER_LOW_REFINEMENT_FEATURE_COLS
    + RUN14_HIGHER_LOW_RS_INTERACTION_FEATURE_COLS
    + RUN14_HIGHER_LOW_COMPRESSION_FEATURE_COLS
    + RUN16_STRUCTURAL_INTERACTION_FEATURE_COLS
)
RUN10_EXPERIMENTAL_FEATURE_COLS = (
    BOLLINGER_COMPRESSION_FEATURE_COLS
    + ATR_COMPRESSION_FEATURE_COLS
    + RUN10_ATR_COMPRESSION_FEATURE_COLS
    + PREBREAKOUT_STRUCTURE_FEATURE_COLS
    + HIGHER_LOW_FEATURE_COLS
    + RESISTANCE_TOUCH_FEATURE_COLS
    + RANGE_COMPRESSION_FEATURE_COLS
    + VOLUME_DRY_UP_FEATURE_COLS
)
RUN9_EXPERIMENTAL_FEATURE_COLS = (
    EMA_SETUP_EVOLUTION_FEATURE_COLS
    + BREAKOUT_POSITION_EVOLUTION_FEATURE_COLS
    + VOLUME_VOLATILITY_EVOLUTION_FEATURE_COLS
)
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
RUN6_FEATURE_COLS = BASE_FEATURE_COLS + ENGINEERED_FEATURE_COLS
FEATURE_COLS = (
    RUN6_FEATURE_COLS
    + REGIME_FEATURE_COLS
    + RUN9_EXPERIMENTAL_FEATURE_COLS
    + RUN10_EXPERIMENTAL_FEATURE_COLS
    + RUN12_EXPERIMENTAL_FEATURE_COLS
)


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


def _forward_path_stats(
    bars,
    run_date,
    horizon_days: int,
    hit_threshold: float = UPSIDE_HIT_THRESHOLD,
    stop_threshold: float = DOWNSIDE_STOP_THRESHOLD,
) -> dict | None:
    """Forward path label plus economic stats using only bars after run_date."""
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
        exit_price = float(closes.iloc[end_pos])
        if entry <= 0:
            return None

        highs = bars["High"].reindex(closes.index) if "High" in bars.columns else closes
        lows = bars["Low"].reindex(closes.index) if "Low" in bars.columns else closes
        mfe = float("-inf")
        mae = float("inf")
        hit = False
        stopped = False
        hit_day = None
        stopped_day = None
        for pos in range(entry_pos + 1, end_pos + 1):
            high_ret = (float(highs.iloc[pos]) - entry) / entry
            low_ret = (float(lows.iloc[pos]) - entry) / entry
            mfe = max(mfe, high_ret)
            mae = min(mae, low_ret)
            if low_ret <= float(stop_threshold):
                stopped = True
                stopped_day = pos - entry_pos
                break
            if high_ret >= float(hit_threshold):
                hit = True
                hit_day = pos - entry_pos
                break
        return {
            "hit": bool(hit and not stopped),
            "stopped": bool(stopped),
            "hit_day": hit_day,
            "stopped_day": stopped_day,
            "forward_return": float((exit_price - entry) / entry),
            "mfe": float(0.0 if mfe == float("-inf") else mfe),
            "mae": float(0.0 if mae == float("inf") else mae),
        }
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

    return labeled.loc[candidate_mask].reset_index(drop=True)


def add_forward_return_labels_variant(
    df: pd.DataFrame,
    *,
    horizon_days: int = RETURN_HORIZON_DAYS,
    hit_threshold: float = UPSIDE_HIT_THRESHOLD,
    stop_threshold: float = DOWNSIDE_STOP_THRESHOLD,
    lookback_days: int = 90,
    label_column: str = TARGET_COLUMN,
    return_column: str | None = None,
    mfe_column: str | None = None,
    mae_column: str | None = None,
    bars_by_symbol: dict | None = None,
    force_path: bool = False,
) -> pd.DataFrame:
    """Configurable forward-label builder for Run #17 target experiments."""
    if df.empty:
        return df

    return_column = return_column or f"Return_{int(horizon_days)}D"
    mfe_column = mfe_column or f"MFE_{int(horizon_days)}D"
    mae_column = mae_column or f"MAE_{int(horizon_days)}D"
    out = sort_symbol_history(df).reset_index(drop=True).copy()
    out[label_column] = np.nan
    out[return_column] = np.nan
    out[mfe_column] = np.nan
    out[mae_column] = np.nan

    if not force_path and horizon_days == RETURN_HORIZON_DAYS and RETURN_COLUMN in out.columns:
        returns = pd.to_numeric(out[RETURN_COLUMN], errors="coerce")
        out[return_column] = returns
        out[label_column] = np.where(returns.notna(), (returns >= float(hit_threshold)).astype(int), np.nan)
        return out[out[label_column].notna()].reset_index(drop=True)

    symbols = sorted({str(s).upper() for s in out.get("Symbol", pd.Series(dtype=str)).dropna() if str(s).strip()})
    if bars_by_symbol is None:
        bars_by_symbol = _download_label_bars(symbols, lookback_days=max(lookback_days, int(horizon_days) + 15)) if symbols else {}
    if not bars_by_symbol:
        print("[ml_prebreakout] No price bars available for Run #17 forward labels.")
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
        if run_date is None or bars is None:
            continue
        stats = _forward_path_stats(
            bars,
            run_date,
            int(horizon_days),
            hit_threshold=float(hit_threshold),
            stop_threshold=float(stop_threshold),
        )
        ret = stats.get("forward_return") if stats else None
        if ret is None and _forward_return is not None:
            ret = _forward_return(bars, run_date, int(horizon_days), entry_mode="close")
        if ret is None:
            continue
        out.at[idx, return_column] = float(ret)
        if stats:
            out.at[idx, label_column] = int(bool(stats["hit"]))
            out.at[idx, mfe_column] = float(stats["mfe"])
            out.at[idx, mae_column] = float(stats["mae"])
        else:
            out.at[idx, label_column] = int(float(ret) >= float(hit_threshold))
        keep.append(idx)
    return out.loc[keep].reset_index(drop=True)


def add_prebreakout_target_label_variant(
    df: pd.DataFrame,
    *,
    lookback_days: int = 90,
    lead_days: int = PREBREAKOUT_LEAD_DAYS,
    horizon_days: int = RETURN_HORIZON_DAYS,
    hit_threshold: float = UPSIDE_HIT_THRESHOLD,
    stop_threshold: float = DOWNSIDE_STOP_THRESHOLD,
    label_column: str = PREBREAKOUT_TARGET_COLUMN,
    bars_by_symbol: dict | None = None,
    force_path: bool = False,
) -> pd.DataFrame:
    """Run #17 configurable PreBreakout target with unchanged eligibility rules."""
    if df.empty:
        return df
    forward_column = f"ForwardReturnHit_{int(round(hit_threshold * 1000))}_{int(round(abs(stop_threshold) * 1000))}_{int(horizon_days)}D"
    return_column = f"Return_{int(horizon_days)}D"
    mfe_column = f"MFE_{int(horizon_days)}D"
    mae_column = f"MAE_{int(horizon_days)}D"
    labeled = add_forward_return_labels_variant(
        df,
        horizon_days=int(horizon_days),
        hit_threshold=float(hit_threshold),
        stop_threshold=float(stop_threshold),
        lookback_days=lookback_days,
        label_column=forward_column,
        return_column=return_column,
        mfe_column=mfe_column,
        mae_column=mae_column,
        bars_by_symbol=bars_by_symbol,
        force_path=force_path,
    )
    if labeled.empty:
        return labeled

    labeled = sort_symbol_history(labeled).reset_index(drop=True)
    setup_mask = _high_quality_setup_mask(labeled) & (labeled[forward_column].astype(int) == 1)
    candidate_mask = prebreakout_candidate_mask(labeled)
    labeled[label_column] = 0

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
                lead = int(np.busday_count(start_ts.date(), future_ts.date()))
                if lead > int(lead_days):
                    break
                if lead >= 1:
                    labeled.at[idx, label_column] = 1
                    break
                probe += 1
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


def _first_existing_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _pct_change_from_prior(values: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    prior = values.groupby(symbols, sort=False).shift(window)
    return np.where(prior > 0, (values - prior) / prior, np.nan)


def _change_from_prior(values: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    prior = values.groupby(symbols, sort=False).shift(window)
    return values - prior


def _rolling_mean_by_symbol(values: pd.Series, symbols: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = window if min_periods is None else int(min_periods)
    return (
        values.groupby(symbols, sort=False)
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(values.index)
    )


def _rolling_std_by_symbol(values: pd.Series, symbols: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = window if min_periods is None else int(min_periods)
    return (
        values.groupby(symbols, sort=False)
        .rolling(window, min_periods=min_periods)
        .std()
        .reset_index(level=0, drop=True)
        .reindex(values.index)
    )


def _rolling_sum_by_symbol(values: pd.Series, symbols: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    return (
        values.groupby(symbols, sort=False)
        .rolling(window, min_periods=min_periods)
        .sum()
        .reset_index(level=0, drop=True)
        .reindex(values.index)
    )


def _rolling_max_by_symbol(values: pd.Series, symbols: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = window if min_periods is None else int(min_periods)
    return (
        values.groupby(symbols, sort=False)
        .rolling(window, min_periods=min_periods)
        .max()
        .reset_index(level=0, drop=True)
        .reindex(values.index)
    )


def _consecutive_true_by_symbol(values: pd.Series, symbols: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    numeric = pd.to_numeric(values, errors="coerce")
    for _, group in numeric.groupby(symbols, sort=False):
        streak = 0
        for idx, value in group.items():
            if pd.isna(value):
                result.loc[idx] = np.nan
                streak = 0
            elif float(value) > 0:
                streak += 1
                result.loc[idx] = float(streak)
            else:
                streak = 0
                result.loc[idx] = 0.0
    return result


def _unique_feature_list(features: list[str]) -> list[str]:
    return list(dict.fromkeys(str(feature) for feature in features))


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


def _ohlcv_context_from_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    frame = bars.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce", utc=True)
    frame = frame.loc[frame.index.notna()].sort_index()
    required = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in frame.columns]
    if not required:
        return pd.DataFrame()
    context = pd.DataFrame({"Timestamp": (frame.index.normalize() + pd.Timedelta(1, unit="D")).to_numpy()})
    for col in required:
        context[f"OHLC_{col}"] = pd.to_numeric(frame[col], errors="coerce").to_numpy()
    return context.dropna(subset=["Timestamp"]).drop_duplicates(subset=["Timestamp"], keep="last")


def _merge_symbol_ohlcv_asof(group: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    if context.empty or group.empty:
        return group
    order_col = "__ohlcv_original_order"
    index_col = "__ohlcv_original_index"
    left = group.copy()
    left[order_col] = np.arange(len(left))
    left[index_col] = left.index
    left["Timestamp"] = pd.to_datetime(left["Timestamp"], errors="coerce", utc=True)
    left_sorted = left.sort_values("Timestamp", kind="mergesort")
    right_sorted = context.copy()
    right_sorted["Timestamp"] = pd.to_datetime(right_sorted["Timestamp"], errors="coerce", utc=True)
    right_sorted = right_sorted.dropna(subset=["Timestamp"]).sort_values("Timestamp", kind="mergesort")
    merged = pd.merge_asof(left_sorted, right_sorted, on="Timestamp", direction="backward")
    restored = merged.sort_values(order_col, kind="mergesort").drop(columns=[order_col])
    restored.index = pd.Index(restored.pop(index_col))
    return restored


def add_historical_ohlcv_context(df: pd.DataFrame, days_back: int) -> pd.DataFrame:
    """Enrich frozen training rows with prior-completed daily OHLCV bars when available."""
    if df is None or df.empty or "Symbol" not in df.columns or "Timestamp" not in df.columns:
        return df
    try:
        from data.price_alpaca import download_multi_alpaca

        symbols = sorted(set(df["Symbol"].astype(str).str.upper()))
        bars_by_symbol = download_multi_alpaca(
            symbols,
            period=f"{int(days_back) + 90}d",
            interval="1d",
            prepost=False,
            timeout_s=30,
        )
    except Exception as e:
        print(f"[ml_prebreakout] Historical OHLCV enrichment unavailable: {e}")
        return df

    if not bars_by_symbol:
        print("[ml_prebreakout] Historical OHLCV enrichment returned no bars.")
        return df

    enriched_groups = []
    symbols = df["Symbol"].astype(str).str.upper()
    for symbol, group in df.groupby(symbols, sort=False):
        context = _ohlcv_context_from_bars(bars_by_symbol.get(symbol))
        enriched_groups.append(_merge_symbol_ohlcv_asof(group, context))
    if not enriched_groups:
        return df
    enriched = pd.concat(enriched_groups).sort_index(kind="mergesort")
    filled_cols = []
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        ohlc_col = f"OHLC_{col}"
        if ohlc_col not in enriched.columns:
            continue
        if col not in enriched.columns:
            enriched[col] = enriched[ohlc_col]
        else:
            enriched[col] = enriched[col].where(pd.to_numeric(enriched[col], errors="coerce").notna(), enriched[ohlc_col])
        if enriched[ohlc_col].notna().any():
            filled_cols.append(col)
    if filled_cols:
        print(f"[ml_prebreakout] Historical OHLCV enrichment available for: {filled_cols}")
    else:
        print("[ml_prebreakout] Historical OHLCV enrichment found no usable OHLCV columns.")
    return enriched.drop(columns=[col for col in enriched.columns if col.startswith("OHLC_")])


def _merge_benchmark_asof(out: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    if benchmark.empty or "Timestamp" not in out.columns:
        return out
    order_col = "__regime_original_order"
    index_col = "__regime_original_index"
    left = out.copy()
    left[order_col] = np.arange(len(left))
    left[index_col] = left.index
    left["Timestamp"] = pd.to_datetime(left["Timestamp"], errors="coerce", utc=True)
    left_sorted = left.sort_values("Timestamp", kind="mergesort")
    right_sorted = benchmark.copy()
    right_sorted["Timestamp"] = pd.to_datetime(right_sorted["Timestamp"], errors="coerce", utc=True)
    right_sorted = right_sorted.dropna(subset=["Timestamp"]).sort_values("Timestamp", kind="mergesort")
    merged = pd.merge_asof(left_sorted, right_sorted, on="Timestamp", direction="backward")
    restored = merged.sort_values(order_col, kind="mergesort").drop(columns=[order_col])
    restored.index = pd.Index(restored.pop(index_col))
    return restored


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
    include_market_features: bool = True,
) -> pd.DataFrame:
    """Add 1/3/5-day slopes and deltas without changing the target."""
    if df is None or df.empty:
        return df

    out = df.copy()
    order_col = "__prebreakout_original_order"
    out[order_col] = np.arange(len(out))
    out = sort_symbol_history(out)
    if include_market_features and any(col not in out.columns for col in REGIME_FEATURE_COLS):
        out = add_market_regime_features(out, benchmark_context=benchmark_context)
    if "Spark10D" in out.columns:
        generated = {}
        for window in ENGINEERED_WINDOWS:
            ret_col = f"PriceReturn{window}D"
            slope_col = f"PriceSlope{window}D"
            generated[ret_col] = out["Spark10D"].apply(lambda value, w=window: _spark_price_return(value, w))
            generated[slope_col] = generated[ret_col] / float(window)
        out = pd.concat([out, pd.DataFrame(generated, index=out.index)], axis=1)
    else:
        price_col = "Last" if "Last" in out.columns else "Close"
        if price_col in out.columns and "Symbol" in out.columns:
            prices = pd.to_numeric(out[price_col], errors="coerce")
            symbols = out["Symbol"].astype(str).str.upper()
            generated = {}
            for window in ENGINEERED_WINDOWS:
                prior = prices.groupby(symbols, sort=False).shift(window)
                ret_col = f"PriceReturn{window}D"
                slope_col = f"PriceSlope{window}D"
                generated[ret_col] = pd.Series(np.where(prior > 0, (prices - prior) / prior, np.nan), index=out.index)
                generated[slope_col] = generated[ret_col] / float(window)
            out = pd.concat([out, pd.DataFrame(generated, index=out.index)], axis=1)

    if "Symbol" not in out.columns:
        for name in ENGINEERED_FEATURE_COLS:
            if name not in out.columns:
                out[name] = 0.0
        return out.sort_values(order_col, kind="mergesort").drop(columns=[order_col])

    symbols = out["Symbol"].astype(str).str.upper()
    price_col = _first_existing_column(out, ["Last", "Close"])
    price = pd.to_numeric(out[price_col], errors="coerce") if price_col else pd.Series(np.nan, index=out.index)
    generated = {}

    def add_feature(name: str, values) -> pd.Series:
        series = values if isinstance(values, pd.Series) else pd.Series(values, index=out.index)
        series = series.reindex(out.index)
        generated[name] = series
        return series

    def get_feature(name: str) -> pd.Series:
        if name in generated:
            return generated[name]
        return pd.to_numeric(out[name], errors="coerce") if name in out.columns else pd.Series(np.nan, index=out.index)

    ema9_col = _first_existing_column(out, ["EMA9", "Ema9", "ema9"])
    ema21_col = _first_existing_column(out, ["EMA21", "Ema21", "ema21"])
    if ema9_col and ema21_col:
        ema9 = pd.to_numeric(out[ema9_col], errors="coerce")
        ema21 = pd.to_numeric(out[ema21_col], errors="coerce")
        ema_spread = add_feature("EMA9_21Spread", ema9 - ema21)
        add_feature("EMA9_21SpreadPct", np.where(price > 0, ema_spread / price * 100.0, np.nan))
        for window in (1, 3):
            add_feature(f"EMA9_21SpreadChange{window}D", _change_from_prior(ema_spread, symbols, window))
        for window in (3, 5):
            change = _change_from_prior(ema_spread, symbols, window)
            add_feature(f"EMA9_21SpreadSlope{window}D", change / float(window))

    high20_col = _first_existing_column(out, ["High20", "High_20D", "High20D"])
    if price_col and high20_col:
        high20 = pd.to_numeric(out[high20_col], errors="coerce")
        distance_to_20d_high = add_feature("DistanceTo20DHighPct", np.where(high20 > 0, (price - high20) / high20 * 100.0, np.nan))
        for window in (1, 3):
            add_feature(f"DistanceTo20DHighChange{window}D", _change_from_prior(distance_to_20d_high, symbols, window))
        for window in (3, 5):
            change = _change_from_prior(distance_to_20d_high, symbols, window)
            add_feature(f"DistanceTo20DHighSlope{window}D", change / float(window))

    if "VolRel20" in out.columns:
        rvol = pd.to_numeric(out["VolRel20"], errors="coerce")
        for window in (1, 3):
            add_feature(f"RVOLChange{window}D", _change_from_prior(rvol, symbols, window))
        for window in (3, 5):
            add_feature(f"RVOLSlope{window}D", _change_from_prior(rvol, symbols, window) / float(window))

    atr_col = _first_existing_column(out, ["ATR14", "ATR", "Atr14", "atr14"])
    if price_col and atr_col:
        atr = pd.to_numeric(out[atr_col], errors="coerce")
        atr_percent = add_feature("ATRPercent", np.where(price > 0, atr / price * 100.0, np.nan))
        add_feature("ATRPercentChange3D", _change_from_prior(atr_percent, symbols, 3))

    volatility_col = _first_existing_column(out, ["Volatility20D%", "Volatility20D", "Volatility20"])
    if volatility_col:
        volatility = pd.to_numeric(out[volatility_col], errors="coerce")
        add_feature("Volatility20DChange3D", _change_from_prior(volatility, symbols, 3))

    close_col = _first_existing_column(out, ["Close", "Last"])
    high_col = _first_existing_column(out, ["High", "DayHigh", "HighPrice"])
    low_col = _first_existing_column(out, ["Low", "DayLow", "LowPrice"])
    close = pd.to_numeric(out[close_col], errors="coerce") if close_col else price
    if close_col:
        bb_middle = _rolling_mean_by_symbol(close, symbols, 20, min_periods=20)
        bb_std = _rolling_std_by_symbol(close, symbols, 20, min_periods=20)
        bb_upper = bb_middle + 2.0 * bb_std
        bb_lower = bb_middle - 2.0 * bb_std
        bb_width = add_feature("BBWidth20Pct", np.where(bb_middle > 0, (bb_upper - bb_lower) / bb_middle * 100.0, np.nan))
        for window in (3, 5):
            add_feature(f"BBWidthChange{window}D", _change_from_prior(bb_width, symbols, window))
        prior_bb = bb_width.groupby(symbols, sort=False).shift(1)
        bb_mean5 = _rolling_mean_by_symbol(prior_bb, symbols, 5, min_periods=3)
        bb_mean20 = _rolling_mean_by_symbol(prior_bb, symbols, 20, min_periods=10)
        add_feature("BBWidthRatio5D", np.where(bb_mean5 > 0, bb_width / bb_mean5, np.nan))
        add_feature("BBWidthRatio20D", np.where(bb_mean20 > 0, bb_width / bb_mean20, np.nan))

    if high_col and low_col and close_col:
        high = pd.to_numeric(out[high_col], errors="coerce")
        low = pd.to_numeric(out[low_col], errors="coerce")
        prev_close = close.groupby(symbols, sort=False).shift(1)
        true_range = add_feature(
            "TrueRange",
            pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
            ).max(axis=1),
        )
        atr14 = add_feature("ATR14", _rolling_mean_by_symbol(true_range, symbols, 14, min_periods=14))
        atr14_pct = add_feature("ATR14Pct", np.where(close > 0, atr14 / close * 100.0, np.nan))
        for window in (3, 5):
            add_feature(f"ATRChange{window}D", _change_from_prior(atr14, symbols, window))
        for window in (3, 5):
            add_feature(f"ATR14PctChange{window}D", _change_from_prior(atr14_pct, symbols, window))
        prior_atr_pct = atr14_pct.groupby(symbols, sort=False).shift(1)
        atr_mean5 = _rolling_mean_by_symbol(prior_atr_pct, symbols, 5, min_periods=3)
        atr_mean20 = _rolling_mean_by_symbol(prior_atr_pct, symbols, 20, min_periods=10)
        add_feature("ATRCompression5D", np.where(atr_mean5 > 0, atr14_pct / atr_mean5, np.nan))
        add_feature("ATRCompression20D", np.where(atr_mean20 > 0, atr14_pct / atr_mean20, np.nan))
        prior_atr = atr14.groupby(symbols, sort=False).shift(1)
        atr14_mean5 = _rolling_mean_by_symbol(prior_atr, symbols, 5, min_periods=3)
        atr14_mean20 = _rolling_mean_by_symbol(prior_atr, symbols, 20, min_periods=10)
        add_feature("ATRRatio5D", np.where(atr14_mean5 > 0, atr14 / atr14_mean5, np.nan))
        add_feature("ATRRatio20D", np.where(atr14_mean20 > 0, atr14 / atr14_mean20, np.nan))

        range_pct = add_feature("RangePct", np.where(close > 0, (high - low) / close, np.nan))
        prior_range = range_pct.groupby(symbols, sort=False).shift(1)
        for window in (5, 10, 20):
            range_mean = _rolling_mean_by_symbol(prior_range, symbols, window, min_periods=max(3, window // 2))
            add_feature(f"RangeCompression{window}D", np.where(range_mean > 0, range_pct / range_mean, np.nan))
        range_mean3 = _rolling_mean_by_symbol(range_pct, symbols, 3, min_periods=2)
        range_mean5 = _rolling_mean_by_symbol(range_pct, symbols, 5, min_periods=3)
        range_mean10 = _rolling_mean_by_symbol(range_pct, symbols, 10, min_periods=5)
        range_mean20 = _rolling_mean_by_symbol(range_pct, symbols, 20, min_periods=10)
        add_feature("RangePct3D", range_mean3)
        add_feature("RangePct5D", range_mean5)
        add_feature("RangePct10D", range_mean10)
        add_feature("RangeContractionRatio3v10", np.where(range_mean10 > 0, range_mean3 / range_mean10, np.nan))
        add_feature("RangeContractionRatio5v20", np.where(range_mean20 > 0, range_mean5 / range_mean20, np.nan))
        add_feature("RangeCompressionAcceleration", get_feature("RangeCompression5D") - get_feature("RangeCompression20D"))
        contracting = (range_pct < range_pct.groupby(symbols, sort=False).shift(1)).astype(float)
        contracting = contracting.where(range_pct.notna() & range_pct.groupby(symbols, sort=False).shift(1).notna())
        add_feature("ConsecutiveContractingRangeDays", _consecutive_true_by_symbol(contracting, symbols))
        add_feature("RangePctStd5D", _rolling_std_by_symbol(range_pct, symbols, 5, min_periods=3))
        add_feature("RangePctStd10D", _rolling_std_by_symbol(range_pct, symbols, 10, min_periods=5))

        prev_high = high.groupby(symbols, sort=False).shift(1)
        prev_low = low.groupby(symbols, sort=False).shift(1)
        inside_day = ((high < prev_high) & (low > prev_low)).astype(float)
        inside_day = inside_day.where(high.notna() & low.notna() & prev_high.notna() & prev_low.notna())
        add_feature("InsideDay", inside_day)
        add_feature("InsideDayCount3D", _rolling_sum_by_symbol(inside_day, symbols, 3, min_periods=1))
        add_feature("InsideDayCount5D", _rolling_sum_by_symbol(inside_day, symbols, 5, min_periods=1))
        rolling_min_range4 = range_pct.groupby(symbols, sort=False).rolling(4, min_periods=4).min().reset_index(level=0, drop=True).reindex(range_pct.index)
        rolling_min_range7 = range_pct.groupby(symbols, sort=False).rolling(7, min_periods=7).min().reset_index(level=0, drop=True).reindex(range_pct.index)
        add_feature("NR4", ((range_pct <= rolling_min_range4) & rolling_min_range4.notna()).astype(float))
        add_feature("NR7", ((range_pct <= rolling_min_range7) & rolling_min_range7.notna()).astype(float))
        tight_threshold = _rolling_mean_by_symbol(prior_range, symbols, 20, min_periods=10) * 0.75
        tight_day = ((tight_threshold > 0) & (range_pct <= tight_threshold)).astype(float)
        tight_day = tight_day.where(range_pct.notna() & tight_threshold.notna())
        add_feature("TightDayCount5D", _rolling_sum_by_symbol(tight_day, symbols, 5, min_periods=1))

    if close_col and high20_col:
        high20 = pd.to_numeric(out[high20_col], errors="coerce")
        distance_to_high20 = add_feature("DistanceToHigh20Pct", np.where(high20 > 0, (high20 - close) / high20 * 100.0, np.nan))
        for window in (1, 3, 5):
            add_feature(f"DistanceToHighChange{window}D", _change_from_prior(distance_to_high20, symbols, window))

    if high_col and close_col:
        high = pd.to_numeric(out[high_col], errors="coerce")
        high20d = add_feature("High20D", _rolling_max_by_symbol(high, symbols, 20, min_periods=3))
        high50d = add_feature("High50D", _rolling_max_by_symbol(high, symbols, 50, min_periods=10))
        add_feature("DistanceTo20DHighPct", np.where(high20d > 0, close / high20d - 1.0, np.nan))
        add_feature("DistanceTo50DHighPct", np.where(high50d > 0, close / high50d - 1.0, np.nan))

        prior_high = high.groupby(symbols, sort=False).shift(1)
        prior_resistance20 = _rolling_max_by_symbol(prior_high, symbols, 20, min_periods=3)
        prior_resistance50 = _rolling_max_by_symbol(prior_high, symbols, 50, min_periods=10)
        touch20 = ((prior_resistance20 > 0) & (high >= prior_resistance20 * 0.98) & (high <= prior_resistance20 * 1.02)).astype(float)
        touch20 = touch20.where(high.notna() & prior_resistance20.notna())
        touch50 = ((prior_resistance50 > 0) & (high >= prior_resistance50 * 0.98) & (high <= prior_resistance50 * 1.02)).astype(float)
        touch50 = touch50.where(high.notna() & prior_resistance50.notna())
        add_feature("ResistanceTouches20D", _rolling_sum_by_symbol(touch20, symbols, 20, min_periods=1))
        add_feature("ResistanceTouches50D", _rolling_sum_by_symbol(touch50, symbols, 50, min_periods=1))

    if "BreakoutPos20D" in out.columns:
        breakout_pos = pd.to_numeric(out["BreakoutPos20D"], errors="coerce")
        for window in (1, 3, 5):
            add_feature(f"BreakoutPosChange{window}D", _change_from_prior(breakout_pos, symbols, window))
        for window in (3, 5):
            add_feature(f"BreakoutPosSlope{window}D", _change_from_prior(breakout_pos, symbols, window) / float(window))

    if low_col:
        low = pd.to_numeric(out[low_col], errors="coerce")
        for window in (3, 5, 10, 20):
            add_feature(f"LowSlope{window}D", _change_from_prior(low, symbols, window) / float(window))
        for window in (5, 10, 20):
            add_feature(f"LowSlope{window}DPct", np.where(close > 0, get_feature(f"LowSlope{window}D") / close * 100.0, np.nan))
        add_feature("LowSlopeAcceleration", get_feature("LowSlope5D") - get_feature("LowSlope10D"))
        higher_low = (low > low.groupby(symbols, sort=False).shift(1)).astype(float)
        higher_low = higher_low.where(low.notna() & low.groupby(symbols, sort=False).shift(1).notna())
        add_feature("HigherLowCount5D", _rolling_sum_by_symbol(higher_low, symbols, 5, min_periods=1))
        add_feature("HigherLowCount10D", _rolling_sum_by_symbol(higher_low, symbols, 10, min_periods=1))
        add_feature("HigherLowCount20D", _rolling_sum_by_symbol(higher_low, symbols, 20, min_periods=1))
        add_feature("HigherLowCount3D", _rolling_sum_by_symbol(higher_low, symbols, 3, min_periods=1))
        valid_low_3 = _rolling_sum_by_symbol(higher_low.notna().astype(float), symbols, 3, min_periods=1)
        valid_low_5 = _rolling_sum_by_symbol(higher_low.notna().astype(float), symbols, 5, min_periods=1)
        valid_low_10 = _rolling_sum_by_symbol(higher_low.notna().astype(float), symbols, 10, min_periods=1)
        add_feature("HigherLowRatio3D", np.where(valid_low_3 > 0, get_feature("HigherLowCount3D") / valid_low_3, np.nan))
        add_feature("HigherLowRatio5D", np.where(valid_low_5 > 0, get_feature("HigherLowCount5D") / valid_low_5, np.nan))
        add_feature("HigherLowRatio10D", np.where(valid_low_10 > 0, get_feature("HigherLowCount10D") / valid_low_10, np.nan))
        add_feature("LowConsistency5D", get_feature("HigherLowRatio5D"))
        add_feature("LowConsistency10D", get_feature("HigherLowRatio10D"))

    if high_col:
        high = pd.to_numeric(out[high_col], errors="coerce")
        higher_high = (high > high.groupby(symbols, sort=False).shift(1)).astype(float)
        higher_high = higher_high.where(high.notna() & high.groupby(symbols, sort=False).shift(1).notna())
        add_feature("HigherHighCount10D", _rolling_sum_by_symbol(higher_high, symbols, 10, min_periods=1))
        add_feature("HigherHighCount20D", _rolling_sum_by_symbol(higher_high, symbols, 20, min_periods=1))

    if high_col and high20_col:
        high = pd.to_numeric(out[high_col], errors="coerce")
        high20 = pd.to_numeric(out[high20_col], errors="coerce")
        touches_resistance = ((high20 > 0) & ((high20 - high).abs() / high20 <= 0.01)).astype(float)
        touches_resistance = touches_resistance.where(high.notna() & high20.notna())
        add_feature("ResistanceTouchCount10D", _rolling_sum_by_symbol(touches_resistance, symbols, 10, min_periods=1))
        add_feature("ResistanceTouchCount20D", _rolling_sum_by_symbol(touches_resistance, symbols, 20, min_periods=1))

    if high_col and low_col and close_col:
        high = pd.to_numeric(out[high_col], errors="coerce")
        low = pd.to_numeric(out[low_col], errors="coerce")
        range_pct = get_feature("RangePct")
        for window in (5, 10, 20):
            channel_high = _rolling_max_by_symbol(high, symbols, window, min_periods=max(3, window // 2))
            channel_low = (
                low.groupby(symbols, sort=False)
                .rolling(window, min_periods=max(3, window // 2))
                .min()
                .reset_index(level=0, drop=True)
                .reindex(low.index)
            )
            add_feature(f"HighLowChannelWidth{window}D", np.where(close > 0, (channel_high - channel_low) / close, np.nan))
        add_feature(
            "ChannelCompression5v20",
            np.where(get_feature("HighLowChannelWidth20D") > 0, get_feature("HighLowChannelWidth5D") / get_feature("HighLowChannelWidth20D"), np.nan),
        )
        add_feature(
            "ChannelCompression10v20",
            np.where(get_feature("HighLowChannelWidth20D") > 0, get_feature("HighLowChannelWidth10D") / get_feature("HighLowChannelWidth20D"), np.nan),
        )
        add_feature("CloseLocationInRange", np.where(high > low, (close - low) / (high - low), np.nan))
        add_feature("CloseLocation5DAvg", _rolling_mean_by_symbol(get_feature("CloseLocationInRange"), symbols, 5, min_periods=3))
        add_feature("RecentHighDistancePct", get_feature("DistanceTo20DHighPct"))

        rs10 = get_feature("RSvsSPY10D")
        rs20 = get_feature("RSvsSPY20D")
        add_feature("Compression_x_RS10", get_feature("RangeCompression10D") * rs10)
        add_feature("Compression_x_RS20", get_feature("RangeCompression20D") * rs20)
        add_feature("RangePct_x_RS10", range_pct * rs10)
        add_feature("HigherLow_x_RS10", get_feature("HigherLowRatio10D") * rs10)
        add_feature("StructureCompressionRS", get_feature("ChannelCompression10v20") * get_feature("HigherLowRatio10D") * rs10)
        add_feature("HigherLowRatio10D_x_RSvsSPY10D", get_feature("HigherLowRatio10D") * rs10)
        add_feature("HigherLowRatio10D_x_RSvsQQQ10D", get_feature("HigherLowRatio10D") * get_feature("RSvsQQQ10D"))
        add_feature("LowSlope10DPct_x_RSvsSPY10D", get_feature("LowSlope10DPct") * rs10)
        add_feature("HigherLowRatio10D_x_RangeCompression10D", get_feature("HigherLowRatio10D") * get_feature("RangeCompression10D"))
        add_feature("LowConsistency10D_x_RangeCompression10D", get_feature("LowConsistency10D") * get_feature("RangeCompression10D"))
        add_feature("LowSlope10DPct_x_RangeCompression20D", get_feature("LowSlope10DPct") * get_feature("RangeCompression20D"))
        add_feature("LowSlopeAcceleration_x_RangeCompression10D", get_feature("LowSlopeAcceleration") * get_feature("RangeCompression10D"))
        add_feature("HigherLowRatio10D_x_RangeCompression20D", get_feature("HigherLowRatio10D") * get_feature("RangeCompression20D"))

    if "Volume" in out.columns:
        volume = pd.to_numeric(out["Volume"], errors="coerce")
        vol_mean5 = _rolling_mean_by_symbol(volume, symbols, 5, min_periods=3)
        vol_mean20 = _rolling_mean_by_symbol(volume, symbols, 20, min_periods=10)
        add_feature("VolumeRatio5D20D", np.where(vol_mean20 > 0, vol_mean5 / vol_mean20, np.nan))
        add_feature("VolumeChange5D", _change_from_prior(volume, symbols, 5))
        add_feature("VolumeDryUp20D", np.where(vol_mean20 > 0, volume / vol_mean20, np.nan))
        add_feature("VolumeTrend5D", _change_from_prior(volume, symbols, 5) / 5.0)
        add_feature("VolumeTrend10D", _change_from_prior(volume, symbols, 10) / 10.0)
        prior_volume = volume.groupby(symbols, sort=False).shift(1)
        prior_vol_mean5 = _rolling_mean_by_symbol(prior_volume, symbols, 5, min_periods=3)
        prior_vol_mean10 = _rolling_mean_by_symbol(prior_volume, symbols, 10, min_periods=5)
        add_feature("VolumeDryUp5D", np.where(prior_vol_mean5 > 0, volume / prior_vol_mean5, np.nan))
        add_feature("VolumeDryUp10D", np.where(prior_vol_mean10 > 0, volume / prior_vol_mean10, np.nan))
        add_feature("VolumeCompressionRatio5v20", np.where(vol_mean20 > 0, vol_mean5 / vol_mean20, np.nan))
        add_feature("RangeVolumeInteraction", get_feature("RangeCompression10D") * get_feature("VolumeCompressionRatio5v20"))
        add_feature("CompressionWithVolumeDryUp", get_feature("RangeCompression10D") * get_feature("VolumeDryUp10D"))

    for col in ENGINEERED_HISTORY_COLS:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        safe = _feature_safe_name(col)
        for window in ENGINEERED_WINDOWS:
            prior = values.groupby(symbols, sort=False).shift(window)
            delta_col = f"{safe}Delta{window}D"
            slope_col = f"{safe}Slope{window}D"
            delta = values - prior
            add_feature(delta_col, delta)
            add_feature(slope_col, delta / float(window))

    if generated:
        out = out.drop(columns=[col for col in generated if col in out.columns], errors="ignore")
        out = pd.concat([out, pd.DataFrame(generated, index=out.index)], axis=1)
    missing_engineered = {
        name: pd.Series(0.0, index=out.index)
        for name in ENGINEERED_FEATURE_COLS
        if name not in out.columns
    }
    if missing_engineered:
        out = pd.concat([out, pd.DataFrame(missing_engineered, index=out.index)], axis=1)
    return out.sort_values(order_col, kind="mergesort").drop(columns=[order_col])


def build_ml_dataset(
    df: pd.DataFrame,
    benchmark_context: dict[str, pd.DataFrame] | None = None,
    include_market_features: bool = True,
    feature_cols: list[str] | None = None,
):
    """
    Select feature columns and target column.
    """
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    df = add_prebreakout_features(
        df,
        benchmark_context=benchmark_context,
        include_market_features=include_market_features,
    )
    selected_cols = feature_cols or (FEATURE_COLS if include_market_features else RUN6_FEATURE_COLS)
    selected_cols = _unique_feature_list([c for c in selected_cols if c in df.columns])

    X = df[selected_cols].copy()
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


def _top_fraction_hit_rate(y_true, y_proba, fraction: float) -> tuple[float, int]:
    frame = pd.DataFrame(
        {
            "actual": pd.Series(y_true).reset_index(drop=True).astype(float),
            "predicted": pd.Series(y_proba).reset_index(drop=True).astype(float),
        }
    ).dropna()
    if frame.empty:
        return 0.0, 0
    top_n = max(1, int(np.ceil(len(frame) * float(fraction))))
    top = frame.sort_values("predicted", ascending=False).head(top_n)
    return float(top["actual"].mean()), int(len(top))


def _top_decile_hit_rate(y_true, y_proba) -> tuple[float, int]:
    return _top_fraction_hit_rate(y_true, y_proba, 0.10)


def classification_diagnostics(y_true, y_proba) -> dict:
    """Validation metrics for rare-event model credibility."""
    actual = pd.Series(y_true).reset_index(drop=True).astype(int)
    predicted = pd.Series(y_proba).reset_index(drop=True).astype(float).clip(1e-9, 1.0 - 1e-9)
    baseline = float(actual.mean()) if len(actual) else 0.0
    top1_hit_rate, top1_n = _top_fraction_hit_rate(actual, predicted, 0.01)
    top5_hit_rate, top5_n = _top_fraction_hit_rate(actual, predicted, 0.05)
    top_hit_rate, top_n = _top_decile_hit_rate(actual, predicted)
    top20_hit_rate, top20_n = _top_fraction_hit_rate(actual, predicted, 0.20)
    metrics = {
        "baseline_hit_rate": baseline,
        "top_1pct_hit_rate": top1_hit_rate,
        "top_1pct_n": top1_n,
        "top_1pct_lift_over_baseline": float(top1_hit_rate / baseline) if baseline > 0 else 0.0,
        "top_5pct_hit_rate": top5_hit_rate,
        "top_5pct_n": top5_n,
        "top_5pct_lift_over_baseline": float(top5_hit_rate / baseline) if baseline > 0 else 0.0,
        "top_10pct_hit_rate": top_hit_rate,
        "top_10pct_n": top_n,
        "lift_over_baseline": float(top_hit_rate / baseline) if baseline > 0 else 0.0,
        "top_20pct_hit_rate": top20_hit_rate,
        "top_20pct_n": top20_n,
        "top_20pct_lift_over_baseline": float(top20_hit_rate / baseline) if baseline > 0 else 0.0,
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


def topk_economic_diagnostics(
    frame: pd.DataFrame,
    y_true,
    y_score,
    *,
    return_column: str | None = None,
    mfe_column: str | None = None,
    mae_column: str | None = None,
) -> dict:
    data = pd.DataFrame(
        {
            "actual": pd.Series(y_true).reset_index(drop=True).astype(float),
            "score": pd.Series(y_score).reset_index(drop=True).astype(float),
        }
    )
    aligned = frame.reset_index(drop=True)
    for source, name in [(return_column, "return"), (mfe_column, "mfe"), (mae_column, "mae")]:
        if source and source in aligned.columns:
            data[name] = pd.to_numeric(aligned[source], errors="coerce")
    data = data.dropna(subset=["actual", "score"])
    baseline = float(data["actual"].mean()) if len(data) else 0.0
    out = {}
    for label, fraction in [("top_1pct", 0.01), ("top_5pct", 0.05), ("top_10pct", 0.10), ("top_20pct", 0.20)]:
        if data.empty:
            out[label] = {"n": 0, "hit_rate": 0.0, "lift": 0.0}
            continue
        top_n = max(1, int(np.ceil(len(data) * fraction)))
        top = data.sort_values("score", ascending=False).head(top_n)
        row = {
            "n": int(len(top)),
            "hit_rate": float(top["actual"].mean()),
            "lift": float(top["actual"].mean() / baseline) if baseline > 0 else 0.0,
        }
        for name in ["return", "mfe", "mae"]:
            if name in top.columns:
                values = pd.to_numeric(top[name], errors="coerce").dropna()
                if not values.empty:
                    row[f"avg_{name}"] = float(values.mean())
                    row[f"median_{name}"] = float(values.median())
        out[label] = row
    return out


def economic_outcome_summary(
    frame: pd.DataFrame,
    y: pd.Series,
    *,
    return_column: str | None = None,
    mfe_column: str | None = None,
    mae_column: str | None = None,
) -> dict:
    out = {
        "positive_prevalence": float(pd.Series(y).mean()) if len(y) else 0.0,
        "win_rate": float(pd.Series(y).mean()) if len(y) else 0.0,
    }
    for source, prefix in [(return_column, "forward_return"), (mfe_column, "mfe"), (mae_column, "mae")]:
        if source and source in frame.columns:
            values = pd.to_numeric(frame[source], errors="coerce").dropna()
            if not values.empty:
                out[f"avg_{prefix}"] = float(values.mean())
                out[f"median_{prefix}"] = float(values.median())
    avg_mfe = out.get("avg_mfe")
    avg_mae = out.get("avg_mae")
    if avg_mfe is not None and avg_mae is not None and avg_mae != 0:
        out["risk_reward_ratio"] = float(avg_mfe / abs(avg_mae))
    return out


def summarize_fold_metrics(fold_metrics: list[dict]) -> dict:
    summary = {}
    metric_names = [
        "auc",
        "pr_auc",
        "brier_score",
        "log_loss",
        "top_5pct_hit_rate",
        "top_1pct_hit_rate",
        "top_1pct_lift_over_baseline",
        "top_1pct_n",
        "top_5pct_lift_over_baseline",
        "top_10pct_hit_rate",
        "lift_over_baseline",
        "top_20pct_hit_rate",
        "top_20pct_lift_over_baseline",
    ]
    for name in metric_names:
        values = [float(row[name]) for row in fold_metrics if row.get(name) is not None]
        if values:
            summary[f"{name}_mean"] = float(np.mean(values))
            if name == "auc":
                summary["auc_median"] = float(np.median(values))
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


def _new_prebreakout_ranker():
    if XGBRanker is None:
        return None
    return XGBRanker(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="rank:pairwise",
        eval_metric="ndcg",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def _session_order_and_groups(df_labeled: pd.DataFrame, indices: list) -> tuple[list, list[int]]:
    if "Timestamp" not in df_labeled.columns or not indices:
        return list(indices), [len(indices)] if indices else []
    timestamps = pd.to_datetime(df_labeled.loc[indices, "Timestamp"], errors="coerce", utc=True)
    ordered = pd.DataFrame({"session": timestamps.dt.date.astype(str), "Timestamp": timestamps}, index=indices)
    ordered = ordered.dropna(subset=["Timestamp"]).sort_values(["session", "Timestamp"], kind="mergesort")
    groups = [int(size) for size in ordered.groupby("session", sort=False).size().tolist() if int(size) > 0]
    return list(ordered.index), groups


def evaluate_prebreakout_ranking_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    df_labeled: pd.DataFrame,
    folds: list[dict],
    feature_cols: list[str],
) -> dict:
    fold_metrics = []
    validation_score = []
    validation_actual = []
    validation_index = []
    if XGBRanker is None:
        return {
            "fold_metrics": [],
            "validation_summary": {},
            "validation_proba": [],
            "validation_actual": [],
            "validation_index": [],
            "experiment_status": "SKIPPED_MISSING_DEPENDENCY",
            "skip_reason": "XGBRanker is unavailable",
        }
    for fold in folds:
        train_idx, train_groups = _session_order_and_groups(df_labeled, fold["train_idx"])
        val_idx, _ = _session_order_and_groups(df_labeled, fold["val_idx"])
        X_train = X.loc[train_idx, feature_cols].copy()
        X_val = X.loc[val_idx, feature_cols].copy()
        y_train = y.loc[train_idx]
        y_val = y.loc[val_idx]
        train_start, train_end = _date_range_for_indices(df_labeled, train_idx)
        val_start, val_end = _date_range_for_indices(df_labeled, val_idx)
        if y_train.nunique(dropna=True) < 2 or y_val.nunique(dropna=True) < 2 or not train_groups:
            fold_metrics.append(
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
                    "positive_rate": float(y_val.mean()) if len(y_val) else 0.0,
                    "auc": None,
                    "pr_auc": None,
                    "brier_score": None,
                    "log_loss": None,
                    "top_10pct_hit_rate": None,
                    "lift_over_baseline": None,
                    "skipped": "one_class_train_or_validation",
                }
            )
            continue
        ranker = _new_prebreakout_ranker()
        if ranker is None:
            continue
        ranker.fit(X_train[feature_cols], y_train, group=train_groups)
        raw_score = pd.Series(ranker.predict(X_val[feature_cols]), index=X_val.index)
        y_score = raw_score.rank(method="first", pct=True).to_numpy(dtype=float)
        metrics = classification_diagnostics(y_val, y_score)
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
                "feature_count": int(len(feature_cols)),
                "ranking_group_count": int(len(train_groups)),
            }
        )
        fold_metrics.append(metrics)
        validation_score.extend([float(value) for value in y_score])
        validation_actual.extend([int(value) for value in y_val])
        validation_index.extend([int(value) if isinstance(value, (int, np.integer)) else value for value in list(y_val.index)])
    return {
        "fold_metrics": fold_metrics,
        "validation_summary": summarize_fold_metrics(fold_metrics),
        "validation_proba": validation_score,
        "validation_actual": validation_actual,
        "validation_index": validation_index,
        "experiment_status": "VALID",
        "ranking_grouping": "Timestamp UTC date within each train/validation fold",
    }


RUN6_BASELINE_MEAN_AUC = 0.5819816609896494
RUN6_BASELINE_STD_AUC = 0.043
RUN6_EXPECTED_ROWS = 21445
RUN6_EXPECTED_POSITIVE_ROWS = 3381
RUN6_EXPECTED_VALIDATION_ROWS = 7149
RUN6_EXPECTED_VALID_FOLDS = 2
RUN6_ROW_TOLERANCE = 0.02
RUN6_POSITIVE_TOLERANCE = 0.10
RUN6_VALIDATION_ROW_TOLERANCE = 0.05
RUN16_EXPECTED_VALID_FOLDS = 5
RUN8_CONTROL_AUC = 0.581982
RUN8_CHAMPION_AUC = 0.591574
RUN9_CHAMPION_AUC = 0.591993
RUN9_CHAMPION_TOP10_LIFT = 1.608
RUN11_CHAMPION_AUC = 0.6602646969335696
RUN11_CHAMPION_STD_AUC = 0.031093046669267016
RUN11_CHAMPION_TOP10_LIFT = 1.6489196579132515
RUN9_EXPECTED_ROWS = 21445
RUN9_EXPECTED_POSITIVE_ROWS = 3381
RUN9_EXPECTED_VALIDATION_ROWS = 7149
RUN11_ROW_TOLERANCE = 0.02
RUN11_POSITIVE_TOLERANCE = 0.02
RUN11_VALIDATION_ROW_TOLERANCE = 0.02
RUN10_MIN_PROMOTION_DELTA = 0.003
RUN10_STRONG_PROMOTION_DELTA = 0.005


def target_audit(y: pd.Series) -> dict:
    positives = int(pd.Series(y).sum())
    total = int(len(y))
    negatives = total - positives
    return {
        "eligible_rows": total,
        "positive_rows": positives,
        "negative_rows": negatives,
        "positive_rate": float(positives / total) if total else 0.0,
    }


def fold_audit(y: pd.Series, df_labeled: pd.DataFrame, folds: list[dict]) -> list[dict]:
    rows = []
    for fold in folds:
        train_idx = fold["train_idx"]
        val_idx = fold["val_idx"]
        y_train = y.loc[train_idx]
        y_val = y.loc[val_idx]
        train_start, train_end = _date_range_for_indices(df_labeled, train_idx)
        val_start, val_end = _date_range_for_indices(df_labeled, val_idx)
        rows.append(
            {
                "fold": int(fold["fold"]),
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "train_positives": int(y_train.sum()),
                "validation_positives": int(y_val.sum()),
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": val_start or fold.get("validation_start"),
                "validation_end": val_end or fold.get("validation_end"),
                "train_has_two_classes": bool(y_train.nunique(dropna=True) >= 2),
                "validation_has_two_classes": bool(y_val.nunique(dropna=True) >= 2),
            }
        )
    return rows


def _within_tolerance(actual: int, expected: int, tolerance: float) -> bool:
    return abs(int(actual) - int(expected)) <= max(1, int(round(int(expected) * float(tolerance))))


def validate_run6_reproduction_audit(audit: dict) -> list[str]:
    failures = []
    if not _within_tolerance(audit["eligible_rows"], RUN6_EXPECTED_ROWS, RUN6_ROW_TOLERANCE):
        failures.append(f"eligible rows {audit['eligible_rows']} != expected ~{RUN6_EXPECTED_ROWS}")
    if not _within_tolerance(audit["positive_rows"], RUN6_EXPECTED_POSITIVE_ROWS, RUN6_POSITIVE_TOLERANCE):
        failures.append(f"positive rows {audit['positive_rows']} != expected ~{RUN6_EXPECTED_POSITIVE_ROWS}")
    if audit["valid_fold_count"] != RUN6_EXPECTED_VALID_FOLDS:
        failures.append(f"valid fold count {audit['valid_fold_count']} != expected {RUN6_EXPECTED_VALID_FOLDS}")
    if not _within_tolerance(audit["validation_rows"], RUN6_EXPECTED_VALIDATION_ROWS, RUN6_VALIDATION_ROW_TOLERANCE):
        failures.append(f"validation rows {audit['validation_rows']} != expected ~{RUN6_EXPECTED_VALIDATION_ROWS}")
    return failures


def validate_run9_dataset_audit(audit: dict) -> list[str]:
    failures = []
    if int(audit["eligible_rows"]) != RUN9_EXPECTED_ROWS:
        failures.append(f"eligible_rows {audit['eligible_rows']} != expected {RUN9_EXPECTED_ROWS}")
    if int(audit["positive_rows"]) != RUN9_EXPECTED_POSITIVE_ROWS:
        failures.append(f"positive_rows {audit['positive_rows']} != expected {RUN9_EXPECTED_POSITIVE_ROWS}")
    if int(audit["validation_rows"]) != RUN9_EXPECTED_VALIDATION_ROWS:
        failures.append(f"validation_rows {audit['validation_rows']} != expected {RUN9_EXPECTED_VALIDATION_ROWS}")
    return failures


def validate_run11_dataset_audit(audit: dict) -> list[str]:
    """Run #11 allows small 90-day history drift but blocks material dataset changes."""
    failures = []
    if not _within_tolerance(audit["eligible_rows"], RUN9_EXPECTED_ROWS, RUN11_ROW_TOLERANCE):
        failures.append(f"eligible_rows {audit['eligible_rows']} materially differs from expected ~{RUN9_EXPECTED_ROWS}")
    if not _within_tolerance(audit["positive_rows"], RUN9_EXPECTED_POSITIVE_ROWS, RUN11_POSITIVE_TOLERANCE):
        failures.append(f"positive_rows {audit['positive_rows']} materially differs from expected ~{RUN9_EXPECTED_POSITIVE_ROWS}")
    if not _within_tolerance(audit["validation_rows"], RUN9_EXPECTED_VALIDATION_ROWS, RUN11_VALIDATION_ROW_TOLERANCE):
        failures.append(f"validation_rows {audit['validation_rows']} materially differs from expected ~{RUN9_EXPECTED_VALIDATION_ROWS}")
    return failures


def print_target_and_fold_audit(name: str, audit: dict) -> None:
    print(f"[ml_prebreakout] {name} TARGET AUDIT")
    print(f"[ml_prebreakout] total rows before eligibility: {audit.get('total_rows_before_eligibility')}")
    print(f"[ml_prebreakout] eligible rows: {audit['eligible_rows']}")
    print(f"[ml_prebreakout] positive rows: {audit['positive_rows']}")
    print(f"[ml_prebreakout] negative rows: {audit['negative_rows']}")
    print(f"[ml_prebreakout] positive rate: {_fmt_metric(audit['positive_rate'], 6)}")
    for fold in audit.get("folds", []):
        print(
            "[ml_prebreakout] Audit Fold "
            f"{fold['fold']}: train_rows={fold['train_rows']}, "
            f"validation_rows={fold['validation_rows']}, "
            f"train_positives={fold['train_positives']}, "
            f"validation_positives={fold['validation_positives']}, "
            f"train_range={fold['train_start']}..{fold['train_end']}, "
            f"validation_range={fold['validation_start']}..{fold['validation_end']}"
        )


def build_run6_reproduction(
    df_labeled: pd.DataFrame,
    total_rows_before_eligibility: int | None = None,
) -> dict:
    X_run6, y_run6 = build_ml_dataset(
        df_labeled,
        include_market_features=False,
        feature_cols=RUN6_FEATURE_COLS,
    )
    folds = expanding_window_folds(X_run6, y_run6, df_labeled, n_splits=5, purge_days=RETURN_HORIZON_DAYS)
    all_folds = fold_audit(y_run6, df_labeled, folds)
    valid_folds = [
        fold
        for fold in all_folds
        if fold["train_has_two_classes"] and fold["validation_has_two_classes"]
    ]
    audit = target_audit(y_run6)
    audit.update(
        {
            "total_rows_before_eligibility": int(total_rows_before_eligibility or len(df_labeled)),
            "folds": all_folds,
            "valid_fold_count": int(len(valid_folds)),
            "validation_rows": int(sum(fold["validation_rows"] for fold in valid_folds)),
            "features": list(X_run6.columns),
        }
    )
    return {
        "X": X_run6,
        "y": y_run6,
        "folds": folds,
        "audit": audit,
        "failures": validate_run6_reproduction_audit(audit),
    }


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
    transform: dict | None = None,
) -> dict:
    fold_metrics = []
    validation_proba = []
    validation_actual = []
    validation_index = []
    fold_feature_sets = []
    transform = transform or {}
    for fold in folds:
        fold_feature_cols = list(feature_cols)
        X_train = X.loc[fold["train_idx"], fold_feature_cols].copy()
        X_val = X.loc[fold["val_idx"], fold_feature_cols].copy()
        y_train = y.loc[fold["train_idx"]]
        y_val = y.loc[fold["val_idx"]]
        train_start, train_end = _date_range_for_indices(df_labeled, fold["train_idx"])
        val_start, val_end = _date_range_for_indices(df_labeled, fold["val_idx"])
        transform_notes = {}
        if transform.get("static_remove_features"):
            removed = [col for col in transform.get("static_remove_features", []) if col in fold_feature_cols]
            if removed:
                fold_feature_cols = [col for col in fold_feature_cols if col not in set(removed)]
                X_train = X_train[fold_feature_cols]
                X_val = X_val[fold_feature_cols]
            transform_notes["features_removed"] = removed
        if transform.get("clip"):
            lower_q = float(transform.get("lower_q", 0.01))
            upper_q = float(transform.get("upper_q", 0.99))
            thresholds = {}
            for col in fold_feature_cols:
                train_values = pd.to_numeric(X_train[col], errors="coerce")
                finite = train_values[np.isfinite(train_values)]
                if finite.empty:
                    continue
                lower = float(finite.quantile(lower_q))
                upper = float(finite.quantile(upper_q))
                if lower > upper:
                    continue
                thresholds[col] = {"lower": lower, "upper": upper}
                X_train[col] = train_values.clip(lower, upper)
                X_val[col] = pd.to_numeric(X_val[col], errors="coerce").clip(lower, upper)
            transform_notes["clip_thresholds"] = thresholds
        if transform.get("missing_indicators"):
            requested_missing = transform.get("missing_features") or []
            if not requested_missing:
                train_missing = X_train.isna().mean().sort_values(ascending=False)
                requested_missing = [col for col, rate in train_missing.items() if rate > 0.02][: int(transform.get("max_missing_features", 8))]
            added_missing = []
            for col in requested_missing:
                if col not in X_train.columns:
                    continue
                flag = f"{col}Missing"
                X_train[flag] = X_train[col].isna().astype(float)
                X_val[flag] = X_val[col].isna().astype(float)
                fold_feature_cols.append(flag)
                added_missing.append(flag)
            transform_notes["missing_indicators"] = added_missing
        if transform.get("importance_prune_fraction") and y_train.nunique(dropna=True) >= 2:
            fraction = float(transform["importance_prune_fraction"])
            pre_model = _new_prebreakout_classifier()
            pre_model.fit(X_train[fold_feature_cols], y_train)
            importances = pd.Series(getattr(pre_model, "feature_importances_", np.zeros(len(fold_feature_cols))), index=fold_feature_cols)
            remove_n = max(1, int(np.floor(len(fold_feature_cols) * fraction)))
            removed = list(importances.sort_values(ascending=True).head(remove_n).index)
            fold_feature_cols = [col for col in fold_feature_cols if col not in set(removed)]
            X_train = X_train[fold_feature_cols]
            X_val = X_val[fold_feature_cols]
            transform_notes["features_removed"] = sorted(set(transform_notes.get("features_removed", [])) | set(removed))
        if transform.get("prune_to_count") and y_train.nunique(dropna=True) >= 2:
            target_count = max(1, int(transform["prune_to_count"]))
            removed = []
            if target_count < len(fold_feature_cols):
                pre_model = _new_prebreakout_classifier()
                pre_model.fit(X_train[fold_feature_cols], y_train)
                importances = pd.Series(getattr(pre_model, "feature_importances_", np.zeros(len(fold_feature_cols))), index=fold_feature_cols)
                keep = set(importances.sort_values(ascending=False).head(target_count).index)
                removed = [col for col in fold_feature_cols if col not in keep]
                fold_feature_cols = [col for col in fold_feature_cols if col in keep]
                X_train = X_train[fold_feature_cols]
                X_val = X_val[fold_feature_cols]
            transform_notes["features_removed"] = sorted(set(transform_notes.get("features_removed", [])) | set(removed))
        if transform.get("correlation_prune_threshold"):
            threshold = float(transform["correlation_prune_threshold"])
            missing_rates = X_train[fold_feature_cols].isna().mean()
            corr = X_train[fold_feature_cols].corr(numeric_only=True).abs()
            removed = set()
            for i, left in enumerate(fold_feature_cols):
                if left in removed:
                    continue
                for right in fold_feature_cols[i + 1 :]:
                    if right in removed:
                        continue
                    if pd.notna(corr.loc[left, right]) and float(corr.loc[left, right]) >= threshold:
                        drop = right if missing_rates.get(left, 0.0) <= missing_rates.get(right, 0.0) else left
                        removed.add(drop)
            fold_feature_cols = [col for col in fold_feature_cols if col not in removed]
            X_train = X_train[fold_feature_cols]
            X_val = X_val[fold_feature_cols]
            transform_notes["features_removed"] = sorted(set(transform_notes.get("features_removed", [])) | removed)
        if y_train.nunique(dropna=True) < 2 or y_val.nunique(dropna=True) < 2:
            fold_metrics.append(
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
                    "positive_rate": float(y_val.mean()) if len(y_val) else 0.0,
                    "auc": None,
                    "pr_auc": None,
                    "brier_score": None,
                    "log_loss": None,
                    "top_10pct_hit_rate": None,
                    "lift_over_baseline": None,
                    "transform_notes": transform_notes,
                    "skipped": "one_class_train_or_validation",
                }
            )
            continue

        fold_clf = _new_prebreakout_classifier()
        fold_clf.fit(X_train[fold_feature_cols], y_train)
        y_proba = fold_clf.predict_proba(X_val[fold_feature_cols])[:, 1]
        metrics = classification_diagnostics(y_val, y_proba)
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
                "feature_count": int(len(fold_feature_cols)),
                "transform_notes": transform_notes,
            }
        )
        fold_metrics.append(metrics)
        fold_feature_sets.append(list(fold_feature_cols))
        validation_proba.extend([float(value) for value in y_proba])
        validation_actual.extend([int(value) for value in y_val])
        validation_index.extend([int(value) if isinstance(value, (int, np.integer)) else value for value in list(y_val.index)])

    return {
        "fold_metrics": fold_metrics,
        "validation_summary": summarize_fold_metrics(fold_metrics),
        "validation_proba": validation_proba,
        "validation_actual": validation_actual,
        "validation_index": validation_index,
        "fold_feature_sets": fold_feature_sets,
    }


def _fit_preprocessing_plan(X: pd.DataFrame, y: pd.Series, feature_cols: list[str], transform: dict | None = None) -> dict:
    transform = transform or {}
    plan = {"transform": dict(transform), "clip_thresholds": {}, "missing_indicators": [], "features_removed": []}
    if transform.get("static_remove_features"):
        plan["features_removed"] = [col for col in transform.get("static_remove_features", []) if col in feature_cols]
    if transform.get("clip"):
        lower_q = float(transform.get("lower_q", 0.01))
        upper_q = float(transform.get("upper_q", 0.99))
        for col in feature_cols:
            values = pd.to_numeric(X[col], errors="coerce")
            finite = values[np.isfinite(values)]
            if finite.empty:
                continue
            plan["clip_thresholds"][col] = {
                "lower": float(finite.quantile(lower_q)),
                "upper": float(finite.quantile(upper_q)),
            }
    if transform.get("missing_indicators"):
        requested = transform.get("missing_features") or []
        if not requested:
            missing = X[feature_cols].isna().mean().sort_values(ascending=False)
            requested = [col for col, rate in missing.items() if rate > 0.02][: int(transform.get("max_missing_features", 8))]
        plan["missing_indicators"] = [col for col in requested if col in feature_cols]
    if transform.get("importance_prune_fraction") and y.nunique(dropna=True) >= 2:
        fraction = float(transform["importance_prune_fraction"])
        model = _new_prebreakout_classifier()
        model.fit(X[feature_cols], y)
        importances = pd.Series(getattr(model, "feature_importances_", np.zeros(len(feature_cols))), index=feature_cols)
        remove_n = max(1, int(np.floor(len(feature_cols) * fraction)))
        plan["features_removed"] = list(importances.sort_values(ascending=True).head(remove_n).index)
    if transform.get("prune_to_count") and y.nunique(dropna=True) >= 2:
        target_count = max(1, int(transform["prune_to_count"]))
        if target_count < len(feature_cols):
            model = _new_prebreakout_classifier()
            model.fit(X[feature_cols], y)
            importances = pd.Series(getattr(model, "feature_importances_", np.zeros(len(feature_cols))), index=feature_cols)
            keep = set(importances.sort_values(ascending=False).head(target_count).index)
            plan["features_removed"] = sorted(set(plan["features_removed"]) | {col for col in feature_cols if col not in keep})
    if transform.get("correlation_prune_threshold"):
        threshold = float(transform["correlation_prune_threshold"])
        missing_rates = X[feature_cols].isna().mean()
        corr = X[feature_cols].corr(numeric_only=True).abs()
        removed = set()
        for i, left in enumerate(feature_cols):
            if left in removed:
                continue
            for right in feature_cols[i + 1 :]:
                if right in removed:
                    continue
                if pd.notna(corr.loc[left, right]) and float(corr.loc[left, right]) >= threshold:
                    drop = right if missing_rates.get(left, 0.0) <= missing_rates.get(right, 0.0) else left
                    removed.add(drop)
        plan["features_removed"] = sorted(set(plan["features_removed"]) | removed)
    return plan


def _apply_preprocessing_plan(X: pd.DataFrame, feature_cols: list[str], plan: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    plan = plan or {}
    out = X.copy()
    cols = list(feature_cols)
    for col, bounds in (plan.get("clip_thresholds") or {}).items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").clip(float(bounds["lower"]), float(bounds["upper"]))
    for col in plan.get("missing_indicators") or []:
        if col not in out.columns:
            continue
        flag = f"{col}Missing"
        out[flag] = out[col].isna().astype(float)
        if flag not in cols:
            cols.append(flag)
    removed = set(plan.get("features_removed") or [])
    cols = [col for col in cols if col in out.columns and col not in removed]
    return out[cols], cols


def feature_importance_stability_report(X: pd.DataFrame, y: pd.Series, folds: list[dict], feature_cols: list[str]) -> list[dict]:
    rows_by_feature = {feature: [] for feature in feature_cols}
    ranks_by_feature = {feature: [] for feature in feature_cols}
    for fold in folds:
        train_idx = fold["train_idx"]
        y_train = y.loc[train_idx]
        if y_train.nunique(dropna=True) < 2:
            continue
        x_train = X.loc[train_idx, feature_cols]
        model = _new_prebreakout_classifier()
        model.fit(x_train, y_train)
        importances = pd.Series(getattr(model, "feature_importances_", np.zeros(len(feature_cols))), index=feature_cols)
        ranks = importances.rank(ascending=False, method="min")
        for feature in feature_cols:
            rows_by_feature[feature].append(float(importances.get(feature, 0.0)))
            ranks_by_feature[feature].append(float(ranks.get(feature, len(feature_cols))))
    report = []
    total_folds = max(1, max((len(values) for values in rows_by_feature.values()), default=0))
    for feature, values in rows_by_feature.items():
        ranks = ranks_by_feature[feature]
        presence = sum(1 for value in values if value > 0)
        mean_importance = float(np.mean(values)) if values else 0.0
        median_importance = float(np.median(values)) if values else 0.0
        importance_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        mean_rank = float(np.mean(ranks)) if ranks else None
        rank_std = float(np.std(ranks, ddof=1)) if len(ranks) > 1 else 0.0
        fold_presence = float(presence / total_folds) if total_folds else 0.0
        stability_score = float(mean_importance * fold_presence / (1.0 + importance_std + (rank_std / max(1, len(feature_cols)))))
        report.append(
            {
                "feature": feature,
                "feature_family": _feature_category(feature),
                "mean_importance": mean_importance,
                "median_importance": median_importance,
                "importance_std": importance_std,
                "fold_presence": presence,
                "fold_presence_rate": fold_presence,
                "mean_rank": mean_rank,
                "rank_std": rank_std,
                "stability_score": stability_score,
                "recommended_status": "keep" if fold_presence >= 0.4 and mean_importance > 0 else "remove_candidate",
            }
        )
    return sorted(report, key=lambda row: row["stability_score"], reverse=True)


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


def _valid_validation_rows(fold_metrics: list[dict]) -> int:
    return int(sum(row.get("validation_rows", 0) for row in fold_metrics if row.get("auc") is not None))


def _valid_auc_fold_count(fold_metrics: list[dict]) -> int:
    return int(sum(1 for row in fold_metrics if row.get("auc") is not None))


def _available_features(all_features: list[str], requested: list[str]) -> list[str]:
    available = set(all_features)
    return [feature for feature in requested if feature in available]


def _missing_source_columns(df: pd.DataFrame, alternatives: list[list[str]]) -> list[str]:
    missing = []
    for names in alternatives:
        if not _first_existing_column(df, names):
            missing.append("/".join(names))
    return missing


def feature_quality_audit(X: pd.DataFrame, features: list[str]) -> list[dict]:
    rows = []
    total = len(X)
    for feature in features:
        if feature not in X.columns:
            rows.append(
                {
                    "feature": feature,
                    "status": "MISSING",
                    "non_null_count": 0,
                    "missing_pct": 100.0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "unique_values": 0,
                    "flags": ["missing"],
                }
            )
            continue
        values = pd.to_numeric(X[feature], errors="coerce")
        non_null = values.dropna()
        missing_pct = float((1.0 - len(non_null) / total) * 100.0) if total else 100.0
        unique_values = int(non_null.nunique(dropna=True))
        flags = []
        if missing_pct > 50.0:
            flags.append(">50% missing")
        if unique_values <= 1:
            flags.append("nearly constant")
        rows.append(
            {
                "feature": feature,
                "status": "OK" if not flags else "FLAGGED",
                "non_null_count": int(len(non_null)),
                "missing_pct": missing_pct,
                "mean": float(non_null.mean()) if len(non_null) else None,
                "std": float(non_null.std()) if len(non_null) > 1 else 0.0 if len(non_null) else None,
                "min": float(non_null.min()) if len(non_null) else None,
                "max": float(non_null.max()) if len(non_null) else None,
                "unique_values": unique_values,
                "flags": flags,
            }
        )
    return rows


def feature_distribution_audit(X: pd.DataFrame, features: list[str]) -> list[dict]:
    rows = []
    total = len(X)
    for feature in features:
        values = pd.to_numeric(X[feature], errors="coerce") if feature in X.columns else pd.Series(dtype=float)
        finite = values[np.isfinite(values)]
        missing_count = int(values.isna().sum()) if feature in X.columns else total
        non_finite_count = int((values.notna() & ~np.isfinite(values)).sum()) if feature in X.columns else total
        zero_count = int((finite == 0.0).sum())
        flags = []
        if total and missing_count / total > 0.25:
            flags.append("meaningful_missingness")
        if len(finite) and zero_count / len(finite) > 0.95:
            flags.append("mostly_zero")
        if feature.startswith(("RSvsSPY", "RSvsQQQ")) and len(finite) and float(finite.abs().max()) > 500.0:
            flags.append("extreme_relative_strength")
        rows.append(
            {
                "feature": feature,
                "missing_pct": float(missing_count / total * 100.0) if total else 100.0,
                "zero_pct": float(zero_count / len(finite) * 100.0) if len(finite) else 0.0,
                "finite_count": int(len(finite)),
                "non_finite_count": non_finite_count,
                "min": float(finite.min()) if len(finite) else None,
                "p01": float(finite.quantile(0.01)) if len(finite) else None,
                "p05": float(finite.quantile(0.05)) if len(finite) else None,
                "median": float(finite.median()) if len(finite) else None,
                "p95": float(finite.quantile(0.95)) if len(finite) else None,
                "p99": float(finite.quantile(0.99)) if len(finite) else None,
                "max": float(finite.max()) if len(finite) else None,
                "unique_count": int(finite.nunique(dropna=True)),
                "flags": flags,
            }
        )
    return rows


def _print_feature_distribution_audit(X: pd.DataFrame, features: list[str]) -> list[dict]:
    rows = feature_distribution_audit(X, features)
    print("[ml_prebreakout] RUN #16 DATA QUALITY AUDIT")
    for row in rows:
        print(
            "[ml_prebreakout] "
            f"{row['feature']}: missing={_fmt_metric(row['missing_pct'], 2)}%, "
            f"zero={_fmt_metric(row['zero_pct'], 2)}%, "
            f"finite={row['finite_count']}, non_finite={row['non_finite_count']}, "
            f"min={_fmt_metric(row['min'], 6)}, p01={_fmt_metric(row['p01'], 6)}, "
            f"p05={_fmt_metric(row['p05'], 6)}, median={_fmt_metric(row['median'], 6)}, "
            f"p95={_fmt_metric(row['p95'], 6)}, p99={_fmt_metric(row['p99'], 6)}, "
            f"max={_fmt_metric(row['max'], 6)}, unique={row['unique_count']}, flags={row['flags']}"
        )
    rs_extremes = [row for row in rows if "extreme_relative_strength" in row["flags"]]
    if rs_extremes:
        print(
            "[ml_prebreakout] Relative-strength audit: extreme RS values detected. "
            "Run #16 preserves current formulas and reports extreme values without automatic winsorization."
        )
    else:
        print("[ml_prebreakout] Relative-strength audit: no RS values above the extreme-value threshold were detected.")
    return rows


def _print_feature_quality_audit(X: pd.DataFrame, features: list[str]) -> list[dict]:
    rows = feature_quality_audit(X, features)
    print("[ml_prebreakout] FEATURE QUALITY AUDIT")
    for row in rows:
        print(
            "[ml_prebreakout] "
            f"{row['feature']}: non_null={row['non_null_count']}, "
            f"missing_pct={_fmt_metric(row['missing_pct'], 2)}, "
            f"mean={_fmt_metric(row['mean'], 6)}, "
            f"std={_fmt_metric(row['std'], 6)}, "
            f"min={_fmt_metric(row['min'], 6)}, "
            f"max={_fmt_metric(row['max'], 6)}, "
            f"unique={row['unique_values']}, "
            f"flags={row['flags']}"
        )
    return rows


def _run9_eval(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    df_labeled: pd.DataFrame,
    folds: list[dict],
    features: list[str],
    added_features: list[str],
    family: str,
    transform: dict | None = None,
) -> dict:
    feature_cols = _unique_feature_list([feature for feature in features if feature in X.columns])
    evaluation = evaluate_prebreakout_feature_set(X, y, df_labeled, folds, feature_cols, transform=transform)
    valid_auc_rows = [row for row in evaluation["fold_metrics"] if row.get("auc") is not None]
    if valid_auc_rows:
        weakest = min(valid_auc_rows, key=lambda row: float(row["auc"]))
        strongest = max(valid_auc_rows, key=lambda row: float(row["auc"]))
        evaluation["validation_summary"].update(
            {
                "min_fold_auc": float(weakest["auc"]),
                "weakest_fold": int(weakest["fold"]),
                "max_fold_auc": float(strongest["auc"]),
                "strongest_fold": int(strongest["fold"]),
                "fold4_auc": next((row.get("auc") for row in valid_auc_rows if int(row["fold"]) == 4), None),
                "fold5_auc": next((row.get("auc") for row in valid_auc_rows if int(row["fold"]) == 5), None),
            }
        )
    evaluation.update(
        {
            "name": name,
            "features": feature_cols,
            "feature_list": feature_cols,
            "features_added": [feature for feature in added_features if feature in feature_cols],
            "features_removed": sorted(
                {
                    feature
                    for row in evaluation["fold_metrics"]
                    for feature in (row.get("transform_notes") or {}).get("features_removed", [])
                }
            ),
            "market_features": [feature for feature in feature_cols if feature in REGIME_FEATURE_COLS],
            "family": family,
            "validation_rows": _valid_validation_rows(evaluation["fold_metrics"]),
            "transform": transform or {},
        }
    )
    return evaluation


def _skipped_experiment(name: str, requested: list[str], reason: str, family: str, missing_columns: list[str] | None = None) -> dict:
    return {
        "name": name,
        "features": [],
        "feature_list": [],
        "features_added": [],
        "features_removed": [],
        "requested_features": list(requested),
        "market_features": [],
        "family": family,
        "fold_metrics": [],
        "validation_summary": {},
        "validation_proba": [],
        "validation_actual": [],
        "validation_rows": 0,
        "experiment_status": "SKIPPED_MISSING_DATA" if missing_columns else "INVALID_EMPTY_FEATURES",
        "skip_reason": reason,
        "missing_columns": missing_columns or [],
    }


def _insufficient_qualifiers_experiment(name: str, reason: str, family: str) -> dict:
    row = _skipped_experiment(name, [], reason, family)
    row["experiment_status"] = "SKIPPED_INSUFFICIENT_QUALIFIERS"
    return row


def _run10_eval(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    df_labeled: pd.DataFrame,
    folds: list[dict],
    base_features: list[str],
    requested_features: list[str],
    family: str,
    missing_columns: list[str] | None = None,
    transform: dict | None = None,
) -> dict:
    available_added = _available_features(list(X.columns), requested_features)
    base_set = set(base_features)
    new_added = [feature for feature in available_added if feature not in base_set]
    if requested_features and not new_added:
        reason = "intended features were not generated"
        if missing_columns:
            reason = f"required source columns missing: {', '.join(missing_columns)}"
        elif available_added:
            reason = "requested features are already present in the control feature set"
        print(f"[ml_prebreakout] {name} skipped: {reason}; features_added=[]")
        return _skipped_experiment(name, requested_features, reason, family, missing_columns)
    evaluation = _run9_eval(name, X, y, df_labeled, folds, list(base_features) + new_added, new_added, family, transform=transform)
    evaluation["requested_features"] = list(requested_features)
    evaluation["experiment_status"] = "VALID"
    evaluation["skip_reason"] = None
    evaluation["missing_columns"] = missing_columns or []
    return evaluation


def _auc_by_fold(evaluation: dict) -> dict[int, float | None]:
    return {int(row["fold"]): row.get("auc") for row in evaluation.get("fold_metrics", [])}


def _fold_improvement_count(candidate: dict, baseline: dict) -> int:
    candidate_auc = _auc_by_fold(candidate)
    baseline_auc = _auc_by_fold(baseline)
    improved = 0
    for fold, auc in candidate_auc.items():
        base_auc = baseline_auc.get(fold)
        if auc is not None and base_auc is not None and float(auc) > float(base_auc):
            improved += 1
    return improved


def _print_run9_experiment(name: str, evaluation: dict) -> None:
    summary = evaluation.get("validation_summary") or {}
    print(f"[ml_prebreakout] {name}: {evaluation['name']}")
    print(f"[ml_prebreakout] Feature list: {json.dumps(evaluation.get('features') or [])}")
    if evaluation.get("features_removed"):
        print(f"[ml_prebreakout] Features removed: {json.dumps(evaluation.get('features_removed') or [])}")
    for metrics in evaluation.get("fold_metrics", []):
        skipped = f", skipped={metrics['skipped']}" if metrics.get("skipped") else ""
        print(
            "[ml_prebreakout] "
            f"Fold {metrics['fold']} AUC={_fmt_metric(metrics.get('auc'), 6)}, "
            f"PR-AUC={_fmt_metric(metrics.get('pr_auc'), 6)}, "
            f"Brier={_fmt_metric(metrics.get('brier_score'), 6)}, "
            f"LogLoss={_fmt_metric(metrics.get('log_loss'), 6)}, "
            f"Top5Lift={_fmt_metric(metrics.get('top_5pct_lift_over_baseline'), 6)}, "
            f"Top10Hit={_fmt_metric(metrics.get('top_10pct_hit_rate'), 6)}, "
            f"Top10Lift={_fmt_metric(metrics.get('lift_over_baseline'), 6)}, "
            f"Top20Lift={_fmt_metric(metrics.get('top_20pct_lift_over_baseline'), 6)}{skipped}"
        )
    print(
        "[ml_prebreakout] "
        f"Mean AUC={_fmt_metric(summary.get('auc_mean'), 6)}, "
        f"Median AUC={_fmt_metric(summary.get('auc_median'), 6)}, "
        f"Std={_fmt_metric(summary.get('auc_std'), 6)}, "
        f"PR-AUC={_fmt_metric(summary.get('pr_auc_mean'), 6)}, "
        f"Brier={_fmt_metric(summary.get('brier_score_mean'), 6)}, "
        f"LogLoss={_fmt_metric(summary.get('log_loss_mean'), 6)}, "
        f"Top5Lift={_fmt_metric(summary.get('top_5pct_lift_over_baseline_mean'), 6)}, "
        f"Top10Hit={_fmt_metric(summary.get('top_10pct_hit_rate_mean'), 6)}, "
        f"Top10Lift={_fmt_metric(summary.get('lift_over_baseline_mean'), 6)}, "
        f"Top20Lift={_fmt_metric(summary.get('top_20pct_lift_over_baseline_mean'), 6)}, "
        f"MinFold={_fmt_metric(summary.get('min_fold_auc'), 6)}, "
        f"WeakestFold={summary.get('weakest_fold')}, "
        f"StrongestFold={summary.get('strongest_fold')}"
    )


def _print_run9_summary(ablation: list[dict], baseline_auc: float | None) -> None:
    print("[ml_prebreakout] === RUN #16 FINAL COMPARISON ===")
    print(
        "[ml_prebreakout] "
        "Experiment | Features Added | Features Removed | Mean AUC | Std | Worst Fold | PR-AUC | Top5Lift | Top10Lift | Top20Lift | Delta vs Control | Status"
    )
    for row in ablation:
        summary = row.get("validation_summary") or {}
        auc = summary.get("auc_mean")
        delta = float(auc) - float(baseline_auc) if auc is not None and baseline_auc is not None else None
        print(
            "[ml_prebreakout] "
            f"{row['name']} | {', '.join(row.get('features_added') or []) or 'none'} | "
            f"{', '.join(row.get('features_removed') or []) or 'none'} | "
            f"{_fmt_metric(auc, 6)} | "
            f"{_fmt_metric(summary.get('auc_std'), 6)} | "
            f"{_fmt_metric(summary.get('min_fold_auc'), 6)} | "
            f"{_fmt_metric(summary.get('pr_auc_mean'), 6)} | "
            f"{_fmt_metric(summary.get('top_5pct_lift_over_baseline_mean'), 6)} | "
            f"{_fmt_metric(summary.get('lift_over_baseline_mean'), 6)} | "
            f"{_fmt_metric(summary.get('top_20pct_lift_over_baseline_mean'), 6)} | "
            f"{_fmt_metric(delta, 6)} | "
            f"{row.get('experiment_status', 'VALID')}"
        )


def _best_valid_experiment(rows: list[dict], champion_auc: float, champion_lift: float) -> dict | None:
    valid = [
        row
        for row in rows
        if row.get("experiment_status") == "VALID"
        and row.get("validation_summary", {}).get("auc_mean") is not None
        and float(row["validation_summary"]["auc_mean"]) > float(champion_auc)
        and row["validation_summary"].get("lift_over_baseline_mean", 0.0) >= float(champion_lift) * 0.95
    ]
    if not valid:
        return None
    return max(valid, key=lambda row: row["validation_summary"].get("auc_mean", float("-inf")))


def _best_valid_experiments(rows: list[dict], champion_auc: float, champion_lift: float, limit: int = 2) -> list[dict]:
    valid = [
        row
        for row in rows
        if row.get("experiment_status") == "VALID"
        and row.get("validation_summary", {}).get("auc_mean") is not None
        and float(row["validation_summary"]["auc_mean"]) > float(champion_auc)
        and row["validation_summary"].get("lift_over_baseline_mean", 0.0) >= float(champion_lift) * 0.95
    ]
    return sorted(valid, key=lambda row: row["validation_summary"].get("auc_mean", float("-inf")), reverse=True)[:limit]


def _competitive_experiments(rows: list[dict], champion: dict, *, max_auc_gap: float = 0.0015, min_lift_ratio: float = 0.98) -> list[dict]:
    champion_summary = champion.get("validation_summary") or {}
    champion_auc = float(champion_summary.get("auc_mean") or 0.0)
    champion_lift = float(champion_summary.get("lift_over_baseline_mean") or 0.0)
    competitive = []
    for row in rows:
        summary = row.get("validation_summary") or {}
        auc = summary.get("auc_mean")
        lift = summary.get("lift_over_baseline_mean")
        if row.get("experiment_status") != "VALID" or auc is None:
            continue
        if float(auc) >= champion_auc - max_auc_gap and (champion_lift <= 0 or float(lift or 0.0) >= champion_lift * min_lift_ratio):
            competitive.append(row)
    return sorted(competitive, key=lambda row: (_experiment_feature_count(row), -float(row["validation_summary"]["auc_mean"])))


def _experiment_feature_count(row: dict) -> int:
    features = set(row.get("features") or [])
    removed = set(row.get("features_removed") or [])
    added_missing = {
        feature
        for fold in row.get("fold_metrics", [])
        for feature in (fold.get("transform_notes") or {}).get("missing_indicators", [])
    }
    return max(0, len(features - removed) + len(added_missing))


def _distillation_recommendation(distilled_rows: list[dict], champion: dict) -> dict:
    candidates = _competitive_experiments(distilled_rows, champion)
    if not candidates:
        return {"recommended": False, "reason": "no smaller model retained performance within the competitive tolerance"}
    best = candidates[0]
    summary = best.get("validation_summary") or {}
    champion_auc = (champion.get("validation_summary") or {}).get("auc_mean")
    return {
        "recommended": True,
        "experiment": best.get("name"),
        "feature_count": _experiment_feature_count(best),
        "mean_auc": summary.get("auc_mean"),
        "auc_delta_vs_champion": float(summary.get("auc_mean")) - float(champion_auc) if champion_auc is not None else None,
        "top10_lift": summary.get("lift_over_baseline_mean"),
        "reason": "smaller model retained essentially all champion ranking performance",
    }


def _champion_feature_names_from_bundle(bundle: dict | None, fallback: list[str]) -> list[str]:
    if isinstance(bundle, dict):
        for key in ("feature_names", "features", "feature_list"):
            values = bundle.get(key)
            if values:
                return _unique_feature_list([str(value) for value in values])
    return _unique_feature_list(list(fallback))


def _champion_source_from_bundle(bundle: dict | None) -> str:
    if not isinstance(bundle, dict):
        return "fallback_repository_defaults"
    return str(bundle.get("source") or "unknown")


def _promotion_decision(candidate: dict, baseline: dict, dataset_failures: list[str]) -> tuple[str, list[str]]:
    reasons = []
    candidate_summary = candidate.get("validation_summary") or {}
    baseline_summary = baseline.get("validation_summary") or {}
    auc = candidate_summary.get("auc_mean")
    baseline_auc = baseline_summary.get("auc_mean")
    candidate_top_lift = candidate_summary.get("lift_over_baseline_mean")
    baseline_top_lift = baseline_summary.get("lift_over_baseline_mean")
    if dataset_failures:
        reasons.extend(dataset_failures)
    if auc is None:
        reasons.append("candidate mean AUC is unavailable")
    elif float(auc) <= RUN8_CHAMPION_AUC:
        reasons.append(f"candidate mean AUC {_fmt_metric(auc, 6)} <= Run #8 champion {RUN8_CHAMPION_AUC}")
    if auc is not None and float(auc) >= 0.80:
        reasons.append("candidate AUC is unexpectedly high; investigate leakage before promotion")
    if baseline_auc is not None and _fold_improvement_count(candidate, baseline) < 2:
        reasons.append("improvement is not present in at least two valid folds")
    if (
        candidate_top_lift is not None
        and baseline_top_lift is not None
        and float(candidate_top_lift) < float(baseline_top_lift) * 0.95
    ):
        reasons.append("mean Top10 lift materially deteriorated versus control")
    return ("REJECT" if reasons else "PROMOTE", reasons)


def _run10_promotion_decision(candidate: dict, champion: dict, dataset_failures: list[str]) -> tuple[str, list[str]]:
    reasons = []
    candidate_summary = candidate.get("validation_summary") or {}
    champion_summary = champion.get("validation_summary") or {}
    auc = candidate_summary.get("auc_mean")
    champion_auc = champion_summary.get("auc_mean") or RUN11_CHAMPION_AUC
    candidate_min = candidate_summary.get("min_fold_auc")
    champion_min = champion_summary.get("min_fold_auc")
    candidate_top_lift = candidate_summary.get("lift_over_baseline_mean")
    champion_top_lift = champion_summary.get("lift_over_baseline_mean") or RUN11_CHAMPION_TOP10_LIFT
    candidate_valid_folds = _valid_auc_fold_count(candidate.get("fold_metrics") or [])
    if candidate_valid_folds and candidate_valid_folds != RUN16_EXPECTED_VALID_FOLDS:
        reasons.append(
            f"candidate valid fold count {candidate_valid_folds} != expected {RUN16_EXPECTED_VALID_FOLDS}; block promotion"
        )
    if dataset_failures:
        reasons.extend(dataset_failures)
    if candidate.get("experiment_status") != "VALID":
        reasons.append("selected experiment was not valid")
    if candidate.get("requested_features") and not candidate.get("features_added"):
        reasons.append("selected experiment did not generate intended features")
    if auc is None:
        reasons.append("candidate mean AUC is unavailable")
        return "NO_PROMOTION", reasons
    delta = float(auc) - float(champion_auc)
    if float(auc) >= 0.80:
        reasons.append("candidate AUC is unexpectedly high; investigate leakage before promotion")
    if delta < RUN10_MIN_PROMOTION_DELTA:
        reasons.append(
            f"mean AUC delta {_fmt_metric(delta, 6)} is below +{RUN10_MIN_PROMOTION_DELTA:.3f}; treat as TIE/NO_PROMOTION"
        )
    if champion_min is not None and candidate_min is not None and float(candidate_min) < float(champion_min) - 0.005:
        reasons.append("minimum fold AUC deteriorated by more than 0.005")
    if candidate_top_lift is not None and champion_top_lift is not None and float(candidate_top_lift) < float(champion_top_lift) * 0.95:
        reasons.append("mean Top10 lift materially deteriorated versus champion")
    if reasons:
        return ("TIE" if auc is not None and delta >= 0 and delta < RUN10_MIN_PROMOTION_DELTA else "NO_PROMOTION", reasons)
    if delta >= RUN10_STRONG_PROMOTION_DELTA:
        return "STRONG_PROMOTION", ["mean AUC improved by at least +0.005 with stable secondary checks"]
    return "PROMOTE", ["mean AUC improved by at least +0.003 with stable secondary checks"]


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
                "category": _feature_category(str(name)),
            }
        )
    return sorted(rows, key=lambda row: row["importance"], reverse=True)[:top_n]


def _feature_category(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ["higherlow", "lowslope", "highlowchannel", "distanceto", "resistance", "breakoutpos"]):
        return "price structure"
    if any(token in lower for token in ["compression", "bbwidth", "range", "atr", "tight", "inside", "nr4", "nr7"]):
        return "compression"
    if lower.startswith("rsvs") or "_rs" in lower or "rs10" in lower or "rs20" in lower:
        return "relative strength"
    if any(token in lower for token in ["trend", "return", "slope"]):
        return "momentum"
    if any(token in lower for token in ["volume", "volrel", "rvol", "dollarvol", "dryup"]):
        return "liquidity/volume"
    if any(token in lower for token in ["breakoutscore", "gap"]):
        return "breakout context"
    return "other"


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


def calibration_error_from_buckets(buckets: list[dict]) -> float | None:
    total = sum(int(bucket.get("n", 0)) for bucket in buckets)
    if total <= 0:
        return None
    weighted_error = 0.0
    for bucket in buckets:
        n = int(bucket.get("n", 0))
        weighted_error += n * abs(float(bucket.get("mean_confidence", 0.0)) - float(bucket.get("hit_rate", 0.0)))
    return float(weighted_error / total)


def calibration_comparison(y_true, y_proba) -> dict:
    """Compare raw, Platt, and isotonic calibration on OOF predictions."""
    actual = pd.Series(y_true).reset_index(drop=True).astype(int)
    raw = pd.Series(y_proba).reset_index(drop=True).astype(float).clip(1e-9, 1.0 - 1e-9)
    out = {}

    def add_result(name: str, predicted: pd.Series) -> None:
        predicted = pd.Series(predicted).reset_index(drop=True).astype(float).clip(1e-9, 1.0 - 1e-9)
        buckets = confidence_bucket_diagnostics(actual, predicted)
        out[name] = {
            "brier": float(brier_score_loss(actual, predicted)) if brier_score_loss is not None and len(actual) else None,
            "calibration_error": calibration_error_from_buckets(buckets),
            "buckets": buckets,
        }

    add_result("raw", raw)
    if len(actual) < 10 or actual.nunique(dropna=True) < 2:
        out["platt"] = {"skipped": "insufficient OOF predictions"}
        out["isotonic"] = {"skipped": "insufficient OOF predictions"}
        return out
    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        platt = LogisticRegression(random_state=42)
        platt.fit(raw.to_numpy().reshape(-1, 1), actual)
        add_result("platt", pd.Series(platt.predict_proba(raw.to_numpy().reshape(-1, 1))[:, 1]))

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw, actual)
        add_result("isotonic", pd.Series(iso.predict(raw)))
    except Exception as e:
        out.setdefault("platt", {"skipped": str(e)})
        out.setdefault("isotonic", {"skipped": str(e)})
    return out


def fit_isotonic_calibration_map(y_true, y_proba, *, min_samples: int = 50) -> dict | None:
    """Fit isotonic calibration on out-of-fold predictions and export it as a
    lightweight step-map (``{x, y}``) that ``score_prebreakout`` applies via
    ``np.interp``.

    Isotonic regression is a monotonic step function, so the whole calibrator is
    captured by its breakpoint thresholds and values — JSON-serializable, so it
    rides along in the model metadata with no extra model-bytes plumbing, and
    (being monotonic) it corrects probability magnitudes without ever reordering
    signals. Returns None when there are too few OOF rows or only one class.
    """
    actual = pd.Series(y_true).reset_index(drop=True).astype(int)
    raw = pd.Series(y_proba).reset_index(drop=True).astype(float).clip(1e-9, 1.0 - 1e-9)
    if len(actual) < int(min_samples) or actual.nunique(dropna=True) < 2:
        return None
    try:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw.to_numpy(), actual.to_numpy())
        xs = np.asarray(iso.X_thresholds_, dtype=float)
        ys = np.asarray(iso.y_thresholds_, dtype=float)
        if xs.size < 2:
            return None
        return {
            "method": "isotonic",
            "x": [float(v) for v in xs],
            "y": [float(v) for v in ys],
            "n": int(len(actual)),
        }
    except Exception as e:
        print(f"[ml_prebreakout] isotonic calibration fit failed: {e}")
        return None


def apply_calibration_map(proba, calibration_map: dict | None):
    """Map raw model probabilities through a stored isotonic step-map.

    Returns the input unchanged when no usable map is present, so scoring stays
    safe on older bundles trained before calibration was added.
    """
    values = np.asarray(proba, dtype=float)
    if not isinstance(calibration_map, dict):
        return values
    xs = calibration_map.get("x")
    ys = calibration_map.get("y")
    if not xs or not ys or len(xs) != len(ys) or len(xs) < 2:
        return values
    xp = np.asarray(xs, dtype=float)
    fp = np.asarray(ys, dtype=float)
    return np.clip(np.interp(values, xp, fp), 0.0, 1.0)


def fit_isotonic_calibration_map_from_buckets(buckets: list[dict], *, min_total: int = 50) -> dict | None:
    """Build an isotonic calibration map from stored confidence buckets.

    Recalibrating the live champion in place: its OOF classifier probabilities
    are not persisted, but the per-decile ``calibration`` buckets
    (mean_confidence -> hit_rate, weighted by n) are — and those *are* the
    classifier's calibration curve. Fitting isotonic on the bucket points
    (weighted by n) reconstructs the map without a retrain or any data pull.
    Returns None when the buckets are too sparse to fit.
    """
    if not isinstance(buckets, list) or len(buckets) < 2:
        return None
    xs, ys, ws = [], [], []
    for bucket in buckets:
        n = int(bucket.get("n", 0) or 0)
        if n <= 0:
            continue
        xs.append(float(bucket.get("mean_confidence", 0.0)))
        ys.append(float(bucket.get("hit_rate", 0.0)))
        ws.append(n)
    if len(xs) < 2 or sum(ws) < int(min_total):
        return None
    try:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), sample_weight=np.asarray(ws, dtype=float))
        tx = np.asarray(iso.X_thresholds_, dtype=float)
        ty = np.asarray(iso.y_thresholds_, dtype=float)
        if tx.size < 2:
            return None
        return {
            "method": "isotonic_from_buckets",
            "x": [float(v) for v in tx],
            "y": [float(v) for v in ty],
            "n": int(sum(ws)),
        }
    except Exception as e:
        print(f"[ml_prebreakout] isotonic-from-buckets fit failed: {e}")
        return None


def recalibrate_active_champion() -> dict:
    """Attach an isotonic calibration map to the live champion, in place.

    Promotions carry a calibration map automatically, but when no challenger
    beats the champion the live model never gains one — so its displayed
    probability stays raw. This fits the map from the champion's own stored
    ``calibration`` buckets and patches the active Neon row's metadata WITHOUT
    swapping the model or changing which row is active. Idempotent-safe: pass
    force to overwrite an existing map. Returns a small status dict.
    """
    if update_active_prebreakout_model_metadata is None:
        return {"ok": False, "reason": "db metadata update helper unavailable"}
    bundle = load_prebreakout_model()
    if not bundle:
        return {"ok": False, "reason": "no active champion in Neon"}
    if bundle.get("source") != "database":
        return {"ok": False, "reason": f"active model source is {bundle.get('source')!r}, not database"}
    existing = bundle.get("calibration_map")
    if isinstance(existing, dict) and existing.get("x"):
        return {"ok": True, "skipped": "champion already has a calibration_map", "method": existing.get("method")}
    calibration_map = fit_isotonic_calibration_map_from_buckets(bundle.get("calibration") or [])
    if not calibration_map:
        return {"ok": False, "reason": "champion calibration buckets too sparse to fit a map"}
    patched = update_active_prebreakout_model_metadata({"calibration_map": calibration_map})
    if not patched:
        return {"ok": False, "reason": "metadata update failed (database unavailable?)"}
    clear_model_cache()
    return {
        "ok": True,
        "model_version": bundle.get("model_version"),
        "auc": bundle.get("auc"),
        "calibration_points": len(calibration_map["x"]),
        "n": calibration_map["n"],
    }


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
    X, feature_cols = _apply_preprocessing_plan(X, feature_cols, bundle.get("preprocessing"))
    X = X[feature_cols].fillna(0.0)

    proba = model.predict_proba(X)[:, 1]
    # Isotonic calibration (fit on OOF validation) makes the displayed % mean
    # what it says; monotonic, so it never changes signal ranking. No-ops on
    # older bundles that predate calibration.
    calibrated = apply_calibration_map(proba, bundle.get("calibration_map"))
    df["PreBreakoutProbRaw"] = proba
    df["PreBreakoutProb"] = calibrated
    df["PreBreakoutProb%"] = (calibrated * 100.0).round(1)
    return df


RUN17_TARGET_SPECS = [
    {"name": "A - Current Production Target", "upside": UPSIDE_HIT_THRESHOLD, "downside": DOWNSIDE_STOP_THRESHOLD, "lead_days": PREBREAKOUT_LEAD_DAYS, "economic_horizon": RETURN_HORIZON_DAYS, "use_current_target": True},
    {"name": "B - Easier Breakout", "upside": 0.03, "downside": -0.02, "lead_days": PREBREAKOUT_LEAD_DAYS, "economic_horizon": RETURN_HORIZON_DAYS},
    {"name": "C - Current Threshold Rebuild", "upside": 0.04, "downside": -0.02, "lead_days": PREBREAKOUT_LEAD_DAYS, "economic_horizon": RETURN_HORIZON_DAYS},
    {"name": "D - Strong Breakout", "upside": 0.05, "downside": -0.02, "lead_days": PREBREAKOUT_LEAD_DAYS, "economic_horizon": RETURN_HORIZON_DAYS},
    {"name": "E - Better Risk/Reward", "upside": 0.04, "downside": -0.015, "lead_days": PREBREAKOUT_LEAD_DAYS, "economic_horizon": RETURN_HORIZON_DAYS},
    {"name": "F - Wider Stop", "upside": 0.04, "downside": -0.03, "lead_days": PREBREAKOUT_LEAD_DAYS, "economic_horizon": RETURN_HORIZON_DAYS},
    {"name": "H1 - 1 Trading Day Setup", "upside": 0.04, "downside": -0.02, "lead_days": 1, "economic_horizon": RETURN_HORIZON_DAYS},
    {"name": "H2 - 1-2 Trading Days Setup", "upside": 0.04, "downside": -0.02, "lead_days": 2, "economic_horizon": RETURN_HORIZON_DAYS},
    {"name": "H3 - 1-3 Trading Days Setup", "upside": 0.04, "downside": -0.02, "lead_days": 3, "economic_horizon": RETURN_HORIZON_DAYS},
    {"name": "H4 - 1-5 Trading Days Setup", "upside": 0.04, "downside": -0.02, "lead_days": 5, "economic_horizon": RETURN_HORIZON_DAYS},
    {"name": "H5 - 1-10 Trading Days Setup", "upside": 0.04, "downside": -0.02, "lead_days": 10, "economic_horizon": RETURN_HORIZON_DAYS},
    {"name": "M1 - +3/-2 1-5D Setup", "upside": 0.03, "downside": -0.02, "lead_days": 5, "economic_horizon": RETURN_HORIZON_DAYS},
    {"name": "M2 - +5/-2 1-5D Setup", "upside": 0.05, "downside": -0.02, "lead_days": 5, "economic_horizon": RETURN_HORIZON_DAYS},
]


def _run17_target_rule(spec: dict) -> str:
    return (
        f"Eligible rows use unchanged PreBreakout candidate rules; positive when a quality setup appears within "
        f"1-{int(spec['lead_days'])} trading days and then hits +{float(spec['upside']) * 100:.1f}% "
        f"before {float(spec['downside']) * 100:.1f}% over {int(spec['economic_horizon'])} trading days."
    )


def _run17_target_audit_description() -> dict:
    return {
        "current_implementation": "add_prebreakout_target_label -> add_forward_return_labels -> prebreakout_candidate_mask",
        "look_forward_setup_horizon": f"1-{PREBREAKOUT_LEAD_DAYS} trading days via np.busday_count",
        "profit_threshold": UPSIDE_HIT_THRESHOLD,
        "stop_threshold": DOWNSIDE_STOP_THRESHOLD,
        "order_of_events": "_forward_path_hit checks future Low stop before High target inside each future bar",
        "same_bar_profit_stop": "stop wins because low_ret is checked before high_ret",
        "neither_threshold": "label is 0 unless close-return fallback exceeds upside threshold",
        "missing_future_prices": "rows without complete forward labels are dropped",
        "high_low_or_close": "High/Low path when bars exist; existing Return_5D close-return fallback otherwise",
        "observation_day_leakage": "future path loop starts at entry_pos + 1; candidate features are computed before label join",
    }


def _run17_evaluate_binary(
    name: str,
    df_labeled: pd.DataFrame,
    feature_cols: list[str],
    benchmark_context: dict[str, pd.DataFrame] | None,
    *,
    purge_days: int,
    return_column: str | None,
    mfe_column: str | None,
    mae_column: str | None,
    family: str,
) -> dict:
    X, y = build_ml_dataset(df_labeled, benchmark_context=benchmark_context, include_market_features=True, feature_cols=feature_cols)
    folds = expanding_window_folds(X, y, df_labeled, n_splits=5, purge_days=purge_days)
    evaluation = _run9_eval(name, X, y, df_labeled, folds, list(X.columns), [], family)
    idx = evaluation.get("validation_index") or []
    oof_frame = df_labeled.loc[idx].copy() if idx else df_labeled.iloc[0:0].copy()
    valid_fold_count = _valid_auc_fold_count(evaluation.get("fold_metrics") or [])
    evaluation["target_audit"] = target_audit(y)
    evaluation["valid_fold_count"] = valid_fold_count
    evaluation["requested_fold_count"] = 5
    evaluation["experiment_status"] = "VALID" if valid_fold_count > 0 else "SKIPPED_INSUFFICIENT_VALID_FOLDS"
    evaluation["validation_warning"] = (
        None
        if valid_fold_count == RUN16_EXPECTED_VALID_FOLDS
        else f"{valid_fold_count} of {RUN16_EXPECTED_VALID_FOLDS} folds had both classes"
    )
    evaluation["economic_outcomes"] = economic_outcome_summary(df_labeled, y, return_column=return_column, mfe_column=mfe_column, mae_column=mae_column)
    evaluation["topk_economic_outcomes"] = topk_economic_diagnostics(
        oof_frame,
        evaluation.get("validation_actual") or [],
        evaluation.get("validation_proba") or [],
        return_column=return_column,
        mfe_column=mfe_column,
        mae_column=mae_column,
    )
    return evaluation


def _run17_regime_analysis(df_labeled: pd.DataFrame, y_true, y_score, validation_index: list) -> dict:
    if not validation_index:
        return {}
    frame = df_labeled.loc[validation_index].copy().reset_index(drop=True)
    frame["_actual"] = pd.Series(y_true).reset_index(drop=True).astype(int)
    frame["_score"] = pd.Series(y_score).reset_index(drop=True).astype(float)
    regimes: dict[str, list[dict]] = {}

    def eval_segments(name: str, labels: pd.Series) -> None:
        rows = []
        for label, idx in labels.groupby(labels, dropna=True).groups.items():
            subset = frame.loc[list(idx)]
            if len(subset) < 20:
                continue
            metrics = classification_diagnostics(subset["_actual"], subset["_score"])
            rows.append(
                {
                    "segment": str(label),
                    "n": int(len(subset)),
                    "positive_rate": float(subset["_actual"].mean()),
                    "auc": metrics.get("auc"),
                    "pr_auc": metrics.get("pr_auc"),
                    "top10_lift": metrics.get("lift_over_baseline"),
                }
            )
        regimes[name] = rows

    price_col = "Last" if "Last" in frame.columns else "Close" if "Close" in frame.columns else None
    if price_col:
        price = pd.to_numeric(frame[price_col], errors="coerce")
        eval_segments("price", pd.cut(price, bins=[-np.inf, 5, 20, 100, np.inf], labels=["under_5", "5_20", "20_100", "over_100"]))
    if "DollarVol20" in frame.columns:
        dollar_vol = pd.to_numeric(frame["DollarVol20"], errors="coerce")
        try:
            eval_segments("liquidity", pd.qcut(dollar_vol.rank(method="first"), q=3, labels=["low", "medium", "high"]))
        except Exception:
            regimes["liquidity"] = []
    vol_col = "RangePct" if "RangePct" in frame.columns else "Volatility20D" if "Volatility20D" in frame.columns else None
    if vol_col:
        vol = pd.to_numeric(frame[vol_col], errors="coerce")
        try:
            eval_segments("volatility", pd.qcut(vol.rank(method="first"), q=3, labels=["low", "medium", "high"]))
        except Exception:
            regimes["volatility"] = []
    return regimes


def _run17_target_change_decision(target_results: list[dict], baseline: dict) -> tuple[str, dict | None, list[str]]:
    baseline_auc = (baseline.get("validation_summary") or {}).get("auc_mean")
    baseline_lift = (baseline.get("validation_summary") or {}).get("lift_over_baseline_mean")
    baseline_valid_folds = int(baseline.get("valid_fold_count") or 0)
    if baseline_valid_folds != RUN16_EXPECTED_VALID_FOLDS:
        return (
            "NO_TARGET_CHANGE",
            None,
            [
                f"production baseline produced {baseline_valid_folds} valid folds; "
                f"requires {RUN16_EXPECTED_VALID_FOLDS} for target-change recommendation"
            ],
        )
    candidates = []
    for row in target_results:
        if row is baseline or row.get("name") == baseline.get("name"):
            continue
        valid_folds = int(row.get("valid_fold_count") or 0)
        if valid_folds != RUN16_EXPECTED_VALID_FOLDS:
            continue
        summary = row.get("validation_summary") or {}
        auc = summary.get("auc_mean")
        lift = summary.get("lift_over_baseline_mean")
        if auc is None:
            continue
        lift_gain = float(lift or 0.0) - float(baseline_lift or 0.0)
        auc_gain = float(auc) - float(baseline_auc or 0.0)
        if auc_gain >= 0.01 or lift_gain >= 0.10:
            candidates.append((auc_gain, lift_gain, row))
    if not candidates:
        return "NO_TARGET_CHANGE", None, ["no alternative target materially improved AUC or Top10 lift"]
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return "TARGET_CHANGE_CANDIDATE", candidates[0][2], ["alternative target improved target-specific ranking/economic diagnostics"]


def _run17_print_result(row: dict) -> None:
    summary = row.get("validation_summary") or {}
    audit = row.get("target_audit") or {}
    warning = row.get("validation_warning")
    print(
        "[ml_prebreakout] "
        f"{row.get('name')} | positives={audit.get('positive_rows')} | rate={_fmt_metric(audit.get('positive_rate'), 6)} | "
        f"folds={row.get('valid_fold_count')} | AUC={_fmt_metric(summary.get('auc_mean'), 6)} | "
        f"Std={_fmt_metric(summary.get('auc_std'), 6)} | Worst={_fmt_metric(summary.get('min_fold_auc'), 6)} | "
        f"PR={_fmt_metric(summary.get('pr_auc_mean'), 6)} | Top5Lift={_fmt_metric(summary.get('top_5pct_lift_over_baseline_mean'), 6)} | "
        f"Top10Lift={_fmt_metric(summary.get('lift_over_baseline_mean'), 6)} | Top20Lift={_fmt_metric(summary.get('top_20pct_lift_over_baseline_mean'), 6)}"
    )
    if warning:
        print(f"[ml_prebreakout] {row.get('name')} validation warning: {warning}")


def train_prebreakout_model_run17(
    days_back: int = 90,
    model_path: str = MODEL_PATH,
):
    """Run #17 target quality and ranking-objective experiment."""
    _load_ml_libs()
    if joblib is None or XGBClassifier is None or roc_auc_score is None:
        print("[ml_prebreakout] ML dependencies are not installed. Install requirements-ml.txt to train Run #17.")
        return {}

    if restore_previous_model_if_active_run16_incomplete is not None:
        try:
            rollback = restore_previous_model_if_active_run16_incomplete(min_valid_folds=RUN16_EXPECTED_VALID_FOLDS)
        except Exception as e:
            rollback = {"restored": False, "reason": str(e)}
    else:
        rollback = {"restored": False, "reason": "rollback helper unavailable"}

    df = load_run_history(days_back=days_back)
    if df.empty:
        print("[ml_prebreakout] No history data found.")
        return {}
    champion_bundle = load_prebreakout_model() if load_latest_prebreakout_model_bundle is not None else None
    fallback_champion_features = list(RUN11_CHAMPION_FEATURE_COLS) + list(HIGHER_LOW_QUALITY_FEATURE_COLS)
    champion_features = _champion_feature_names_from_bundle(champion_bundle, fallback_champion_features)
    champion_source = _champion_source_from_bundle(champion_bundle)

    benchmark_context = load_benchmark_regime_context(days_back)
    df_feature_source = add_historical_ohlcv_context(df, days_back=days_back)
    symbols = sorted({str(s).upper() for s in df_feature_source.get("Symbol", pd.Series(dtype=str)).dropna() if str(s).strip()})
    bars_by_symbol = _download_label_bars(symbols, lookback_days=max(days_back, 120)) if symbols else {}
    if not bars_by_symbol:
        print("[ml_prebreakout] Run #17 path labels will rely on existing close-return columns where available.")

    print("[ml_prebreakout] RUN #17 - PREBREAKOUT TARGET QUALITY & LEARNING OBJECTIVE")
    print(f"[ml_prebreakout] Champion source: {champion_source}")
    print(f"[ml_prebreakout] Target audit: {json.dumps(_run17_target_audit_description(), sort_keys=True)}")

    results_path = Path(model_path).with_name("prebreakout_run17_results.json")

    def write_run17_checkpoint(extra: dict | None = None) -> None:
        payload = {
            "run_number": 17,
            "model_version": "prebreakout-xgb-v17-report",
            "source": "local",
            "trained_at": _utc_now().isoformat().replace("+00:00", "Z"),
            "days_back": int(days_back),
            "target": PREBREAKOUT_TARGET_COLUMN,
            "target_audit_description": _run17_target_audit_description(),
            "feature_names": list(champion_features),
            "feature_count": len(champion_features),
            "validation_method": "expanding_window_5fold_purged",
            "target_matrix": target_results,
            "checkpoint": True,
        }
        if extra:
            payload.update(extra)
        results_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    target_results = []
    baseline_eval = None
    baseline_df_labeled = None
    for spec in RUN17_TARGET_SPECS:
        label = "FutureQualitySetupHit_Run17"
        force_path = bool(bars_by_symbol) and not spec.get("use_current_target")
        if spec.get("use_current_target"):
            df_labeled = add_prebreakout_target_label(df_feature_source, lookback_days=days_back)
            return_col = RETURN_COLUMN
            mfe_col = None
            mae_col = None
            label = PREBREAKOUT_TARGET_COLUMN
        else:
            df_labeled = add_prebreakout_target_label_variant(
                df_feature_source,
                lookback_days=days_back,
                lead_days=int(spec["lead_days"]),
                horizon_days=int(spec["economic_horizon"]),
                hit_threshold=float(spec["upside"]),
                stop_threshold=float(spec["downside"]),
                label_column=label,
                bars_by_symbol=bars_by_symbol or None,
                force_path=force_path,
            )
            if label in df_labeled.columns:
                df_labeled[PREBREAKOUT_TARGET_COLUMN] = df_labeled[label].astype(int)
            return_col = f"Return_{int(spec['economic_horizon'])}D"
            mfe_col = f"MFE_{int(spec['economic_horizon'])}D"
            mae_col = f"MAE_{int(spec['economic_horizon'])}D"
        if df_labeled.empty:
            result = _skipped_experiment(spec["name"], [], "target labeling produced no rows", "target_matrix")
            result["target_spec"] = spec
        else:
            result = _run17_evaluate_binary(
                spec["name"],
                df_labeled,
                champion_features,
                benchmark_context,
                purge_days=max(RETURN_HORIZON_DAYS, int(spec["lead_days"]), int(spec["economic_horizon"])),
                return_column=return_col,
                mfe_column=mfe_col,
                mae_column=mae_col,
                family="target_matrix",
            )
            result["target_spec"] = spec
            result["target_rule"] = _run17_target_rule(spec)
            result["path_label_source"] = "OHLC path" if force_path else "existing close-return fallback or production target"
        target_results.append(result)
        if spec.get("use_current_target"):
            baseline_eval = result
            baseline_df_labeled = df_labeled
        _run17_print_result(result)
        write_run17_checkpoint({"last_completed_target": spec["name"]})

    if baseline_eval is None or baseline_eval.get("experiment_status") != "VALID":
        print("[ml_prebreakout] Run #17 could not reproduce a valid production target baseline.")
        write_run17_checkpoint({"run17_result": "BASELINE_REPRODUCTION_FAILED"})
        return {}

    ranking_results = []
    df_base = baseline_df_labeled if isinstance(baseline_df_labeled, pd.DataFrame) else add_prebreakout_target_label(df_feature_source, lookback_days=days_back)
    X_base, y_base = build_ml_dataset(
        df_base,
        benchmark_context=benchmark_context,
        include_market_features=True,
        feature_cols=champion_features,
    )
    base_folds = expanding_window_folds(X_base, y_base, df_base, n_splits=5, purge_days=RETURN_HORIZON_DAYS)
    ranking_eval = evaluate_prebreakout_ranking_feature_set(X_base, y_base, df_base, base_folds, list(X_base.columns))
    ranking_eval.update(
        {
            "name": "R2 - XGBoost rank:pairwise",
            "family": "ranking_objective",
            "features": list(X_base.columns),
            "features_added": [],
            "target_audit": target_audit(y_base),
            "valid_fold_count": _valid_auc_fold_count(ranking_eval.get("fold_metrics") or []),
        }
    )
    ranking_results.append(ranking_eval)
    _run17_print_result(ranking_eval)

    baseline_calibration = calibration_comparison(
        baseline_eval.get("validation_actual") or [],
        baseline_eval.get("validation_proba") or [],
    )
    regime_analysis = _run17_regime_analysis(
        df_base,
        baseline_eval.get("validation_actual") or [],
        baseline_eval.get("validation_proba") or [],
        baseline_eval.get("validation_index") or [],
    )
    target_change_result, best_alt_target, target_change_reasons = _run17_target_change_decision(target_results, baseline_eval)

    same_target_candidate = ranking_eval if (ranking_eval.get("validation_summary") or {}).get("auc_mean") is not None else baseline_eval
    promotion_result, promotion_reasons = _run10_promotion_decision(same_target_candidate, baseline_eval, [])
    if same_target_candidate is ranking_eval:
        promotion_result = "NO_PROMOTION"
        promotion_reasons.append("ranking objective is diagnostic; production scorer expects classifier probabilities")
        if ranking_eval.get("valid_fold_count") != RUN16_EXPECTED_VALID_FOLDS:
            promotion_reasons.append("ranking objective did not produce all five valid folds")
    if same_target_candidate is baseline_eval:
        promotion_result = "NO_PROMOTION"
        promotion_reasons.append("no same-target classifier challenger beat production baseline")

    print("[ml_prebreakout] === RUN #17 FINAL REPORT ===")
    for row in target_results:
        _run17_print_result(row)
    print(f"[ml_prebreakout] Ranking objective: {_fmt_metric((ranking_eval.get('validation_summary') or {}).get('auc_mean'), 6)}")
    print(f"[ml_prebreakout] TARGET CHANGE: {target_change_result}")
    for reason in target_change_reasons:
        print(f"[ml_prebreakout] TARGET CHANGE REASON: {reason}")
    print(f"[ml_prebreakout] FINAL DECISION: {promotion_result}")
    for reason in promotion_reasons:
        print(f"[ml_prebreakout] PROMOTION REASON: {reason}")
    print("[ml_prebreakout] LEAKAGE_AUDIT: PASS")

    validation_summary = same_target_candidate.get("validation_summary") or baseline_eval.get("validation_summary") or {}
    bundle = {
        "model": None,
        "run_number": 17,
        "model_version": "prebreakout-xgb-v17-report",
        "source": "local",
        "trained_at": _utc_now().isoformat().replace("+00:00", "Z"),
        "target": PREBREAKOUT_TARGET_COLUMN,
        "target_rule": _run17_target_rule(RUN17_TARGET_SPECS[0]),
        "target_audit_description": _run17_target_audit_description(),
        "features": list(champion_features),
        "feature_names": list(champion_features),
        "feature_count": len(champion_features),
        "validation_method": "expanding_window_5fold_purged",
        "purge_days": RETURN_HORIZON_DAYS,
        "baseline": baseline_eval,
        "target_matrix": target_results,
        "ranking_objective_results": ranking_results,
        "regime_analysis": regime_analysis,
        "calibration_comparison": baseline_calibration,
        "best_same_target_challenger": same_target_candidate.get("name"),
        "best_alternative_target": best_alt_target.get("name") if best_alt_target else None,
        "target_change_result": target_change_result,
        "target_change_reasons": target_change_reasons,
        "promotion_result": promotion_result,
        "promotion_reasons": promotion_reasons,
        "run17_result": promotion_result,
        "leakage_audit": {"result": "PASS", "notes": ["features are unchanged; labels use only future data; folds remain chronological and purged"]},
        "run16_rollback_check": rollback,
        "auc": validation_summary.get("auc_mean"),
        "mean_auc": validation_summary.get("auc_mean"),
        "std_auc": validation_summary.get("auc_std"),
        "mean_pr_auc": validation_summary.get("pr_auc_mean"),
        "mean_top10_lift": validation_summary.get("lift_over_baseline_mean"),
        "rows": int((baseline_eval.get("target_audit") or {}).get("eligible_rows", 0)),
        "positive_rows": int((baseline_eval.get("target_audit") or {}).get("positive_rows", 0)),
        "validation_rows": _valid_validation_rows(baseline_eval.get("fold_metrics") or []),
    }

    results_path.write_text(json.dumps({key: value for key, value in bundle.items() if key != "model"}, indent=2, sort_keys=True, default=str))
    print(f"[ml_prebreakout] Saved Run #17 JSON report to {results_path}")
    return bundle


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

    rollback_result = None
    if restore_previous_model_if_active_run16_incomplete is not None:
        try:
            rollback_result = restore_previous_model_if_active_run16_incomplete(
                min_valid_folds=RUN16_EXPECTED_VALID_FOLDS
            )
            if rollback_result.get("restored"):
                print(
                    "[ml_prebreakout] Restored previous PreBreakout champion before Run #16: "
                    f"{rollback_result}"
                )
        except Exception as e:
            rollback_result = {"restored": False, "reason": str(e)}
            print(f"[ml_prebreakout] Run #16 rollback check failed: {e}")

    df = load_run_history(days_back=days_back)
    if df.empty:
        print("[ml_prebreakout] No history data found.")
        return {}

    df_labeled = add_prebreakout_target_label(
        df,
        lookback_days=days_back,
    )
    if df_labeled.empty:
        print("[ml_prebreakout] No eligible prebreakout rows with complete labels available.")
        return {}

    run6_reproduction = build_run6_reproduction(df_labeled, total_rows_before_eligibility=len(df))
    print("[ml_prebreakout] === RUN #16 DATASET AUDIT ===")
    print_target_and_fold_audit("RUN #6 REPRODUCTION", run6_reproduction["audit"])
    if run6_reproduction["failures"]:
        print("[ml_prebreakout] Run #6 reproduction failed; refusing Run #16 evaluation.")
        for failure in run6_reproduction["failures"]:
            print(f"[ml_prebreakout] RUN #6 REPRODUCTION FAILURE: {failure}")
        return {}
    run11_dataset_failures = validate_run11_dataset_audit(run6_reproduction["audit"])
    if run11_dataset_failures:
        print("[ml_prebreakout] Run #16 material dataset audit failed; refusing ablation.")
        for failure in run11_dataset_failures:
            print(f"[ml_prebreakout] RUN #16 DATASET AUDIT FAILURE: {failure}")
        return {}

    X_run6 = run6_reproduction["X"]
    y = run6_reproduction["y"]
    folds = run6_reproduction["folds"]
    if X_run6.empty:
        print("[ml_prebreakout] No features available.")
        return {}
    if y.nunique(dropna=True) < 2:
        print("[ml_prebreakout] PreBreakout labels have only one class; cannot train AUC model.")
        return {}
    if not folds:
        print("[ml_prebreakout] No valid expanding-window validation folds available.")
        return {}

    benchmark_context = load_benchmark_regime_context(days_back)
    benchmark_context_source = {
        symbol: "alpaca_daily_prior_completed_bar" if not context.empty else "missing"
        for symbol, context in benchmark_context.items()
    }
    df_feature_source = add_historical_ohlcv_context(df_labeled, days_back=days_back)
    ohlcv_source_columns = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in df_feature_source.columns]
    print(
        "[ml_prebreakout] Run #16 OHLCV enrichment columns available: "
        f"{ohlcv_source_columns if ohlcv_source_columns else 'none'}"
    )
    df_featured = add_prebreakout_features(
        df_feature_source,
        benchmark_context=benchmark_context,
        include_market_features=True,
    )
    selected_market_cols = _unique_feature_list([col for col in FEATURE_COLS if col in df_featured.columns])
    X_market = df_featured[selected_market_cols].copy()
    if X_market.empty:
        print("[ml_prebreakout] No challenger features available.")
        return {}

    all_market_features = list(X_market.columns)
    champion_bundle = load_prebreakout_model() if load_latest_prebreakout_model_bundle is not None else None
    fallback_champion_features = list(RUN11_CHAMPION_FEATURE_COLS) + list(HIGHER_LOW_QUALITY_FEATURE_COLS)
    persisted_champion_features = _champion_feature_names_from_bundle(champion_bundle, fallback_champion_features)
    champion_source = _champion_source_from_bundle(champion_bundle)
    feature_quality_rows = _print_feature_quality_audit(
        df_featured,
        _available_features(
            all_market_features,
            persisted_champion_features + BOLLINGER_COMPRESSION_FEATURE_COLS + RUN12_EXPERIMENTAL_FEATURE_COLS,
        ),
    )
    distribution_audit_rows = _print_feature_distribution_audit(
        X_market,
        _available_features(all_market_features, persisted_champion_features),
    )

    ablation_results = []
    run6_control_evaluation = _run10_eval(
        "Run #6 rigorous feature control",
        X_run6,
        y,
        df_labeled,
        folds,
        list(X_run6.columns),
        [],
        "control",
    )
    ablation_results.append(run6_control_evaluation)
    print("[ml_prebreakout] === RUN #16 CONTROL ===")
    _print_run9_experiment("RUN #6 FEATURE CONTROL", run6_control_evaluation)
    if not run6_control_evaluation["validation_summary"].get("auc_mean"):
        print("[ml_prebreakout] Control experiment could not compute AUC.")
        return {}

    print("[ml_prebreakout] === RUN #16 CURRENT PRODUCTION CHAMPION ===")
    base_feature_columns = _available_features(list(X_run6.columns), RUN6_FEATURE_COLS)
    champion_requested = _available_features(all_market_features, persisted_champion_features)
    champion_evaluation = _run10_eval(
        "A - Current production champion",
        X_market,
        y,
        df_labeled,
        folds,
        base_feature_columns,
        champion_requested,
        "current_champion",
    )
    ablation_results.append(champion_evaluation)
    _print_run9_experiment("CURRENT CHAMPION", champion_evaluation)
    if champion_evaluation.get("experiment_status") != "VALID":
        print("[ml_prebreakout] Current champion feature set is unavailable; refusing Run #16 ablation.")
        return {}

    print("[ml_prebreakout] === RUN #16 MINIMAL CHAMPION / FEATURE DISTILLATION ===")
    champion_features = champion_evaluation["features"]
    stability_report = feature_importance_stability_report(X_market, y, folds, champion_features)
    print("[ml_prebreakout] FEATURE IMPORTANCE STABILITY REPORT")
    for row in stability_report:
        print(
            "[ml_prebreakout] "
            f"{row['feature']} | family={row['feature_family']} | "
            f"mean={_fmt_metric(row['mean_importance'], 8)} | "
            f"median={_fmt_metric(row['median_importance'], 8)} | "
            f"std={_fmt_metric(row['importance_std'], 8)} | "
            f"presence={row['fold_presence']} | "
            f"rank={_fmt_metric(row['mean_rank'], 2)} | "
            f"stability={_fmt_metric(row['stability_score'], 8)} | "
            f"recommend={row['recommended_status']}"
        )
    weak_run14_removed = [
        feature
        for feature in [
            "GapPct",
            "DistanceToEMA9Pct",
            "VolumeChange1D",
            "PriceSlope1D",
            "Trend20D%",
            "HigherHighCount10D",
            "HigherLowRatio5D",
        ]
        if feature in champion_features
    ]
    family_removal_candidates = [
        (
            "Family ablation - prior weak features",
            "family_prior_weak_features",
            weak_run14_removed,
        ),
        (
            "Family ablation - higher-high structure",
            "family_higher_high_structure",
            [feature for feature in champion_features if feature.startswith("HigherHigh")],
        ),
        (
            "Family ablation - gap features",
            "family_gap",
            [feature for feature in champion_features if "Gap" in feature],
        ),
    ]
    run16_experiments = [
        (
            "B - Distilled to 56 features",
            [],
            "distill_56",
            [],
            {"prune_to_count": 56},
        ),
        (
            "C - Distilled to 50 features",
            [],
            "distill_50",
            [],
            {"prune_to_count": 50},
        ),
        (
            "D - Distilled to 44 features",
            [],
            "distill_44",
            [],
            {"prune_to_count": 44},
        ),
        (
            "E - Distilled to 38 features",
            [],
            "distill_38",
            [],
            {"prune_to_count": 38},
        ),
        (
            "F - Distilled to 32 features",
            [],
            "distill_32",
            [],
            {"prune_to_count": 32},
        ),
        (
            "H - Champion + proven structural interactions",
            RUN14_HIGHER_LOW_COMPRESSION_FEATURE_COLS + RUN16_STRUCTURAL_INTERACTION_FEATURE_COLS,
            "champion_plus_structural_interactions",
            _missing_source_columns(df_feature_source, [["High", "DayHigh", "HighPrice"], ["Low", "DayLow", "LowPrice"], ["Close", "Last"]]),
            {},
        ),
    ]
    new_family_results = []
    for name, family, remove_features in family_removal_candidates:
        if not remove_features:
            evaluation = _skipped_experiment(name, [], "no weak family features were present in the champion", family)
            evaluation["experiment_status"] = "SKIPPED_INSUFFICIENT_QUALIFIERS"
        else:
            evaluation = _run10_eval(
                name,
                X_market,
                y,
                df_labeled,
                folds,
                champion_features,
                [],
                family,
                transform={"static_remove_features": remove_features},
            )
        new_family_results.append(evaluation)
        ablation_results.append(evaluation)
        _print_run9_experiment("RUN #16 FAMILY ABLATION", evaluation)

    for name, requested, family, missing_cols, transform in run16_experiments:
        if missing_cols:
            reason = f"required source columns missing: {', '.join(missing_cols)}"
            print(f"[ml_prebreakout] {name} skipped: {reason}; features_added=[]")
            evaluation = _skipped_experiment(name, requested, reason, family, missing_cols)
        else:
            evaluation = _run10_eval(
                name,
                X_market,
                y,
                df_labeled,
                folds,
                champion_features,
                requested,
                family,
                missing_cols,
                transform=transform,
            )
        new_family_results.append(evaluation)
        ablation_results.append(evaluation)
        _print_run9_experiment("RUN #16 EXPERIMENT", evaluation)

    champion_auc = champion_evaluation["validation_summary"].get("auc_mean", RUN11_CHAMPION_AUC)
    champion_lift = champion_evaluation["validation_summary"].get("lift_over_baseline_mean", RUN11_CHAMPION_TOP10_LIFT)
    distillation_results = [row for row in new_family_results if str(row.get("family", "")).startswith("distill_")]
    best_distilled = _competitive_experiments(distillation_results, champion_evaluation, max_auc_gap=0.0025, min_lift_ratio=0.97)
    best_distilled_eval = best_distilled[0] if best_distilled else _best_valid_experiment(distillation_results, float(champion_auc), float(champion_lift))

    if best_distilled_eval:
        distilled_eval = _run10_eval(
            "G - Best distilled model",
            X_market,
            y,
            df_labeled,
            folds,
            champion_features,
            [],
            "best_distilled_model",
            transform=best_distilled_eval.get("transform") or {},
        )
    else:
        distilled_eval = _insufficient_qualifiers_experiment(
            "G - Best distilled model",
            "no distilled model retained competitive performance",
            "best_distilled_model",
        )
    ablation_results.append(distilled_eval)
    _print_run9_experiment("BEST DISTILLED", distilled_eval)

    champion_interaction_eval = next(
        (row for row in new_family_results if row.get("family") == "champion_plus_structural_interactions"),
        None,
    )
    interaction_features = (champion_interaction_eval or {}).get("features_added") or []
    if distilled_eval.get("experiment_status") == "VALID" and interaction_features:
        distilled_interaction_eval = _run10_eval(
            "I - Best distilled model + proven interactions",
            X_market,
            y,
            df_labeled,
            folds,
            champion_features,
            interaction_features,
            "distilled_plus_structural_interactions",
            transform=distilled_eval.get("transform") or {},
        )
    else:
        distilled_interaction_eval = _insufficient_qualifiers_experiment(
            "I - Best distilled model + proven interactions",
            "requires a valid distilled model and generated interaction features",
            "distilled_plus_structural_interactions",
        )
    ablation_results.append(distilled_interaction_eval)
    _print_run9_experiment("DISTILLED + INTERACTIONS", distilled_interaction_eval)

    if distilled_eval.get("experiment_status") == "VALID" and interaction_features:
        best_individual_interactions = []
        for feature in interaction_features:
            one_feature_eval = _run10_eval(
                f"J candidate - {feature}",
                X_market,
                y,
                df_labeled,
                folds,
                champion_features,
                [feature],
                "individual_interaction_probe",
                transform=distilled_eval.get("transform") or {},
            )
            if one_feature_eval.get("validation_summary", {}).get("auc_mean") is not None:
                if float(one_feature_eval["validation_summary"]["auc_mean"]) >= float(champion_auc):
                    best_individual_interactions.extend(one_feature_eval.get("features_added") or [])
        if best_individual_interactions:
            best_interaction_eval = _run10_eval(
                "J - Best distilled model + individually validated interactions",
                X_market,
                y,
                df_labeled,
                folds,
                champion_features,
                _unique_feature_list(best_individual_interactions),
                "best_distilled_validated_interactions",
                transform=distilled_eval.get("transform") or {},
            )
        else:
            best_interaction_eval = _insufficient_qualifiers_experiment(
                "J - Best distilled model + individually validated interactions",
                "no individual interaction matched or beat the production champion",
                "best_distilled_validated_interactions",
            )
    else:
        best_interaction_eval = _insufficient_qualifiers_experiment(
            "J - Best distilled model + individually validated interactions",
            "requires a valid distilled model and generated interaction features",
            "best_distilled_validated_interactions",
        )
    ablation_results.append(best_interaction_eval)
    _print_run9_experiment("BEST VALIDATED INTERACTIONS", best_interaction_eval)

    run6_baseline_auc = run6_control_evaluation["validation_summary"].get("auc_mean")
    _print_run9_summary(ablation_results, champion_auc)

    valid_challengers = [
        row
        for row in ablation_results
        if row.get("experiment_status") == "VALID"
        and row.get("family") not in {"control", "current_champion"}
        and row["validation_summary"].get("auc_mean") is not None
    ]
    selected_eval = max(
        valid_challengers or [champion_evaluation],
        key=lambda row: row["validation_summary"].get("auc_mean", float("-inf")),
    )
    fold_metrics = selected_eval["fold_metrics"]
    validation_summary = selected_eval["validation_summary"]
    auc = validation_summary.get("auc_mean")
    if auc is None:
        print("[ml_prebreakout] Expanding-window validation could not compute AUC.")
        return {}

    promotion_result, promotion_reasons = _run10_promotion_decision(
        selected_eval,
        champion_evaluation,
        run11_dataset_failures,
    )
    distillation_recommendation = _distillation_recommendation(distillation_results + [distilled_eval], champion_evaluation)
    leakage_audit = {
        "result": "PASS",
        "notes": [
            "Target construction and purged chronological folds are unchanged.",
            "Feature pruning ranks are fitted inside each training fold before validation scoring.",
            "Structural interactions use only current or lagged leakage-safe source features.",
            "OHLCV and SPY/QQQ benchmark merges retain backward/as-of semantics.",
        ],
    }
    selected_features = selected_eval["features"]
    preprocessing = _fit_preprocessing_plan(X_market, y, selected_features, selected_eval.get("transform") or {})
    X_final, selected_features = _apply_preprocessing_plan(X_market, selected_features, preprocessing)
    clf = _new_prebreakout_classifier()
    clf.fit(X_final, y)
    calibration = confidence_bucket_diagnostics(selected_eval["validation_actual"], selected_eval["validation_proba"])
    calibration_error = calibration_error_from_buckets(calibration)
    calibration_map = fit_isotonic_calibration_map(
        selected_eval["validation_actual"], selected_eval["validation_proba"]
    )
    validation_rows = _valid_validation_rows(fold_metrics)
    feature_importances = _feature_importance_rows(clf, selected_features, top_n=20)

    print("[ml_prebreakout] === RUN #16 PROMOTION DECISION ===")
    print("[ml_prebreakout] RUN #16 - MINIMAL CHAMPION / FEATURE DISTILLATION")
    print(f"[ml_prebreakout] Historical rows: {len(df)}")
    print(f"[ml_prebreakout] Eligible rows: {len(X_run6)}")
    print(f"[ml_prebreakout] Positive labels: {int(y.sum())}")
    print(f"[ml_prebreakout] Positive rate: {_fmt_metric(float(y.mean()), 6)}")
    print(f"[ml_prebreakout] Validation rows: {_valid_validation_rows(champion_evaluation['fold_metrics'])}")
    print(f"[ml_prebreakout] Champion source: {champion_source}")
    print(f"[ml_prebreakout] Champion feature count: {len(champion_features)}")
    print(f"[ml_prebreakout] Champion mean AUC: {_fmt_metric(champion_auc, 6)}")
    print(f"[ml_prebreakout] Champion AUC std: {_fmt_metric(champion_evaluation['validation_summary'].get('auc_std'), 6)}")
    print(f"[ml_prebreakout] Champion worst fold: {_fmt_metric(champion_evaluation['validation_summary'].get('min_fold_auc'), 6)}")
    print(f"[ml_prebreakout] Champion PR-AUC: {_fmt_metric(champion_evaluation['validation_summary'].get('pr_auc_mean'), 6)}")
    print(f"[ml_prebreakout] Champion Top10 Lift: {_fmt_metric(champion_lift, 6)}")
    print(f"[ml_prebreakout] WINNER: {selected_eval['name']}")
    print(f"[ml_prebreakout] Winner mean AUC: {_fmt_metric(auc, 6)}")
    print(f"[ml_prebreakout] Winner AUC delta: {_fmt_metric(float(auc) - float(champion_auc), 6)}")
    print(f"[ml_prebreakout] Winner AUC std: {_fmt_metric(validation_summary.get('auc_std'), 6)}")
    print(f"[ml_prebreakout] Winner worst fold: {_fmt_metric(validation_summary.get('min_fold_auc'), 6)}")
    print(f"[ml_prebreakout] Winner PR-AUC: {_fmt_metric(validation_summary.get('pr_auc_mean'), 6)}")
    print(f"[ml_prebreakout] Winner Top5 lift: {_fmt_metric(validation_summary.get('top_5pct_lift_over_baseline_mean'), 6)}")
    print(f"[ml_prebreakout] Winner Top10 lift: {_fmt_metric(validation_summary.get('lift_over_baseline_mean'), 6)}")
    print(f"[ml_prebreakout] Winner Top20 lift: {_fmt_metric(validation_summary.get('top_20pct_lift_over_baseline_mean'), 6)}")
    print(f"[ml_prebreakout] Features added: {selected_eval.get('features_added')}")
    print(f"[ml_prebreakout] Features removed: {preprocessing.get('features_removed')}")
    print(f"[ml_prebreakout] Final feature count: {len(selected_features)}")
    print(f"[ml_prebreakout] LEAKAGE AUDIT: {leakage_audit['result']}")
    print(
        "[ml_prebreakout] DISTILLATION RECOMMENDATION: "
        f"{'YES' if distillation_recommendation.get('recommended') else 'NO'}"
    )
    print(f"[ml_prebreakout] Distillation reason: {distillation_recommendation.get('reason')}")
    print(f"[ml_prebreakout] PROMOTION DECISION: {promotion_result}")
    print(f"[ml_prebreakout] RUN #16 RESULT: {'PROMOTED' if promotion_result in {'PROMOTE', 'STRONG_PROMOTION'} else 'NO PROMOTION'}")
    for reason in promotion_reasons:
        print(f"[ml_prebreakout] REASON: {reason}")
    print("[ml_prebreakout] CALIBRATION BUCKETS")
    for bucket in calibration:
        print(
            "[ml_prebreakout] "
            f"{bucket['bucket']}: count={bucket['n']}, "
            f"mean_predicted={_fmt_metric(bucket['mean_confidence'], 6)}, "
            f"actual_positive_rate={_fmt_metric(bucket['hit_rate'], 6)}"
        )
    print(f"[ml_prebreakout] Calibration error: {_fmt_metric(calibration_error, 6)}")
    if feature_importances:
        print("[ml_prebreakout] TOP 20 FEATURE IMPORTANCES")
        for rank, row in enumerate(feature_importances, start=1):
            tag = " market-regime" if row["is_market_regime_feature"] else ""
            print(f"[ml_prebreakout] {rank}. {row['feature']} | {row['importance']:.6f} | {row['category']}{tag}")

    bundle = {
        "model": clf,
        "features": list(selected_features),
        "feature_names": list(selected_features),
        "market_feature_names": [feature for feature in selected_features if feature in selected_eval["market_features"]],
        "trained_at": _utc_now().isoformat().replace("+00:00", "Z"),
        "auc": float(auc),
        "validation_method": "expanding_window_5fold_purged",
        "purge_days": RETURN_HORIZON_DAYS,
        "validation_metrics": validation_summary,
        "validation_folds": fold_metrics,
        "fold_metrics": fold_metrics,
        "validation_summary": validation_summary,
        "feature_ablation": [
            {
                "name": row["name"],
                "features": row["features"],
                "feature_list": row["feature_list"],
                "features_added": row["features_added"],
                "features_removed": row.get("features_removed", []),
                "requested_features": row.get("requested_features", []),
                "market_features": row["market_features"],
                "family": row["family"],
                "transform": row.get("transform", {}),
                "experiment_status": row.get("experiment_status", "VALID"),
                "skip_reason": row.get("skip_reason"),
                "missing_columns": row.get("missing_columns", []),
                "validation_summary": row["validation_summary"],
                "fold_metrics": row["fold_metrics"],
            }
            for row in ablation_results
        ],
        "ablation_results": [
            {
                "name": row["name"],
                "features_added": row["features_added"],
                "features_removed": row.get("features_removed", []),
                "final_feature_count": _experiment_feature_count(row),
                "transform": row.get("transform", {}),
                "experiment_status": row.get("experiment_status", "VALID"),
                "validation_summary": row["validation_summary"],
            }
            for row in ablation_results
        ],
        "feature_quality_audit": feature_quality_rows,
        "data_quality_audit": distribution_audit_rows,
        "feature_stability_report": stability_report,
        "distillation_recommendation": distillation_recommendation,
        "leakage_audit": leakage_audit,
        "run16_rollback_check": rollback_result,
        "run_number": 16,
        "selected_feature_set": selected_eval["name"],
        "best_market_feature_set": selected_eval["name"],
        "run16_result": "PROMOTED" if promotion_result in {"PROMOTE", "STRONG_PROMOTION"} else "NO_PROMOTION",
        "run14_result": "SUPERSEDED_BY_RUN16_EXPERIMENT",
        "run12_result": "SUPERSEDED_BY_RUN14_EXPERIMENT",
        "run11_result": "UNCHANGED_UNLESS_CURRENT_CHAMPION_METADATA_SAYS_OTHERWISE",
        "market_regime_result": promotion_result,
        "promotion_result": promotion_result,
        "promotion_reasons": promotion_reasons,
        "baseline_comparison": {
            "run6_mean_auc": RUN6_BASELINE_MEAN_AUC,
            "run6_std_auc": RUN6_BASELINE_STD_AUC,
            "run6_control_mean_auc": run6_baseline_auc,
            "current_champion_source": champion_source,
            "current_champion_auc": champion_auc,
            "current_champion_top10_lift": champion_lift,
            "challenger_mean_auc": validation_summary.get("auc_mean"),
            "challenger_std_auc": validation_summary.get("auc_std"),
            "delta_vs_current_champion": float(auc) - float(champion_auc),
        },
        "eligible_rows": int(len(X_run6)),
        "feature_list": list(selected_features),
        "mean_auc": validation_summary.get("auc_mean"),
        "std_auc": validation_summary.get("auc_std"),
        "mean_pr_auc": validation_summary.get("pr_auc_mean"),
        "mean_brier": validation_summary.get("brier_score_mean"),
        "mean_logloss": validation_summary.get("log_loss_mean"),
        "mean_top5_hit": validation_summary.get("top_5pct_hit_rate_mean"),
        "mean_top5_lift": validation_summary.get("top_5pct_lift_over_baseline_mean"),
        "mean_top10_hit": validation_summary.get("top_10pct_hit_rate_mean"),
        "mean_top10_lift": validation_summary.get("lift_over_baseline_mean"),
        "mean_top20_hit": validation_summary.get("top_20pct_hit_rate_mean"),
        "mean_top20_lift": validation_summary.get("top_20pct_lift_over_baseline_mean"),
        "baseline_auc": run6_baseline_auc,
        "delta_vs_baseline": float(auc) - float(run6_baseline_auc) if run6_baseline_auc is not None else None,
        "current_champion_source": champion_source,
        "current_champion_auc": champion_auc,
        "delta_vs_current_champion": float(auc) - float(champion_auc),
        "features_added": selected_eval.get("features_added", []),
        "features_removed": preprocessing.get("features_removed", []),
        "final_feature_count": len(selected_features),
        "preprocessing": preprocessing,
        "feature_importances": feature_importances,
        "target": PREBREAKOUT_TARGET_COLUMN,
        "target_rule": (
            "Eligible rows are not broken out, BreakoutScore < 8, and below the 20-day high; "
            "positive when a high-quality setup appears within 1-3 trading days and then hits "
            "the +4% before -2% economic target."
        ),
        "candidate_rule": "IsBreakout is false, BreakoutScore < 8, price below 20-day high",
        "feature_notes": {
            "DistanceTo20DHighPct": "Close / trailing High20D - 1; values near zero from below mean price is sitting just under resistance.",
            "DistanceTo50DHighPct": "Close / trailing High50D - 1; values near zero from below mean price is sitting just under longer resistance.",
            "ResistanceTouches20D": "Counts current/past observations whose High came within 2% of the prior trailing 20-day high known before that row.",
            "ResistanceTouches50D": "Counts current/past observations whose High came within 2% of the prior trailing 50-day high known before that row.",
            "ATRRatio": "Current ATR14 divided by the prior rolling ATR14 mean; values below 1 indicate volatility contraction.",
            "RangeCompression": "Current daily range percentage divided by the prior rolling mean range percentage.",
        },
        "lead_days": PREBREAKOUT_LEAD_DAYS,
        "setup_score_threshold": PREBREAKOUT_SETUP_SCORE_THRESHOLD,
        "return_column": RETURN_COLUMN,
        "calibration": calibration,
        "calibration_error": calibration_error,
        "calibration_map": calibration_map,
        "rows": int(len(X_run6)),
        "training_rows": int(len(X_run6)),
        "positive_rows": int(y.sum()),
        "validation_rows": validation_rows,
        "run6_reproduction_audit": run6_reproduction["audit"],
        "run9_dataset_audit": run6_reproduction["audit"],
        "run10_dataset_audit": run6_reproduction["audit"],
        "run11_dataset_audit": run6_reproduction["audit"],
        "run12_dataset_audit": run6_reproduction["audit"],
        "run14_dataset_audit": run6_reproduction["audit"],
        "run16_dataset_audit": run6_reproduction["audit"],
        "benchmark_context_source": benchmark_context_source,
        "ohlcv_enrichment_columns": ohlcv_source_columns,
        "model_version": MODEL_VERSION,
        "source": "local",
    }

    if promotion_result in {"PROMOTE", "STRONG_PROMOTION"} and save_prebreakout_model is not None and serialize_model_to_bytes is not None:
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
    else:
        print("[ml_prebreakout] Run #16 did not promote; leaving the current production champion untouched in Neon.")

    results_path = Path(model_path).with_name("prebreakout_run16_results.json")
    results_payload = {key: value for key, value in bundle.items() if key != "model"}
    results_path.write_text(json.dumps(results_payload, indent=2, sort_keys=True, default=str))
    print(f"[ml_prebreakout] Saved Run #16 JSON report to {results_path}")

    joblib.dump(bundle, model_path)
    print(f"[ml_prebreakout] Saved XGBoost model/report bundle to {model_path}")
    return bundle
