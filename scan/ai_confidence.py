"""AI confidence scoring for scanner result DataFrames."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xgb_breakout_model.joblib"
METADATA_PATH = ROOT / "models" / "xgb_breakout_metadata.json"
CONFIDENCE_COL = "AI Confidence"
WARNING_ATTR = "ai_confidence_warning"
TRAINED_AT_ATTR = "ai_confidence_trained_at"


def load_ai_confidence_metadata(metadata_path: Path = METADATA_PATH) -> dict[str, Any]:
    """Load optional XGBoost model metadata."""
    if not metadata_path.exists():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _warn(frame: pd.DataFrame, message: str, *, trained_at: str | None = None) -> pd.DataFrame:
    frame = frame.copy()
    frame.attrs[WARNING_ATTR] = message
    if trained_at:
        frame.attrs[TRAINED_AT_ATTR] = trained_at
    return frame


def _feature_names(model: Any, metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("feature_names")
    if raw is None:
        raw = getattr(model, "feature_names_in_", None)
    if raw is None:
        return []
    return [str(name) for name in list(raw) if str(name).strip()]


def score_ai_confidence(
    df: pd.DataFrame,
    *,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> pd.DataFrame:
    """Add AI Confidence from the trained XGBoost model when available.

    The function is intentionally non-fatal: missing model, metadata, features,
    or prediction failures return the original frame with a warning in attrs.
    """
    if df is None or df.empty:
        return df

    frame = df.copy()
    metadata = load_ai_confidence_metadata(metadata_path)
    trained_at = metadata.get("trained_at")
    trained_at = str(trained_at) if trained_at else None

    if joblib is None:
        return _warn(frame, "AI confidence model support is not installed.", trained_at=trained_at)
    if not model_path.exists():
        return _warn(frame, "AI confidence model is not available.", trained_at=trained_at)

    try:
        model = joblib.load(model_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return _warn(frame, f"AI confidence model could not be loaded: {type(exc).__name__}.", trained_at=trained_at)

    features = _feature_names(model, metadata)
    if not features:
        return _warn(frame, "AI confidence feature metadata is missing.", trained_at=trained_at)

    missing = [name for name in features if name not in frame.columns]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        return _warn(frame, f"AI confidence skipped; missing feature columns: {preview}{suffix}.", trained_at=trained_at)

    try:
        features_df = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        proba = model.predict_proba(features_df)
        frame[CONFIDENCE_COL] = (proba[:, 1] * 100.0).round(1)
        if trained_at:
            frame.attrs[TRAINED_AT_ATTR] = trained_at
        return frame.sort_values(CONFIDENCE_COL, ascending=False).reset_index(drop=True)
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
        return _warn(frame, f"AI confidence scoring failed: {type(exc).__name__}.", trained_at=trained_at)
