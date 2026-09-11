"""AI confidence scoring for scanner result DataFrames."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from db.ai_confidence_models import (
        load_latest_ai_confidence_model_bundle,
        save_ai_confidence_model,
        update_active_ai_confidence_model_metadata,
    )
except Exception:  # pragma: no cover - keep scanner import resilient
    load_latest_ai_confidence_model_bundle = None  # type: ignore[assignment]
    save_ai_confidence_model = None  # type: ignore[assignment]
    update_active_ai_confidence_model_metadata = None  # type: ignore[assignment]

try:
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xgb_breakout_model.joblib"
METADATA_PATH = ROOT / "models" / "xgb_breakout_metadata.json"
MODEL_VERSION = "ai-confidence-xgb-v1"
CONFIDENCE_COL = "AI Confidence"
WARNING_ATTR = "ai_confidence_warning"
TRAINED_AT_ATTR = "ai_confidence_trained_at"
SOURCE_ATTR = "ai_confidence_source"
CALIBRATION_ATTR = "ai_confidence_calibration"
TARGET_RULE_ATTR = "ai_confidence_target_rule"


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


# Process-level cache: loading re-downloaded the model from Neon and
# re-deserialized it on every scan. The model changes at most daily.
_BUNDLE_CACHE: dict[str, Any] = {}
_BUNDLE_CACHE_TTL_S = 900


def clear_bundle_cache() -> None:
    """Drop the cached bundle (tests; or force a reload after retrain)."""
    _BUNDLE_CACHE.clear()


def load_ai_confidence_bundle(
    *,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> tuple[Any | None, dict[str, Any], str | None]:
    """Load the AI Confidence model, preferring the database source (cached 15m)."""
    if joblib is None:
        return None, {}, "AI confidence model support is not installed."
    import time as _time

    cached = _BUNDLE_CACHE.get("bundle")
    if cached and (_time.time() - cached[0]) < _BUNDLE_CACHE_TTL_S:
        return cached[1]
    result = _load_ai_confidence_bundle_uncached(
        model_path=model_path, metadata_path=metadata_path
    )
    _BUNDLE_CACHE["bundle"] = (_time.time(), result)
    return result


def _load_ai_confidence_bundle_uncached(
    *,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> tuple[Any | None, dict[str, Any], str | None]:

    if load_latest_ai_confidence_model_bundle is not None:
        try:
            bundle = load_latest_ai_confidence_model_bundle(joblib)
            if bundle:
                metadata = dict(bundle.get("metadata") or {})
                metadata.setdefault("feature_names", bundle.get("feature_names") or [])
                metadata.setdefault("trained_at", bundle.get("trained_at"))
                metadata.setdefault("model_version", bundle.get("model_version"))
                metadata["source"] = "database"
                return bundle.get("model"), metadata, None
        except Exception as exc:
            db_warning = f"AI confidence database model load failed: {type(exc).__name__}."
        else:
            db_warning = None
    else:
        db_warning = None

    metadata = load_ai_confidence_metadata(metadata_path)
    if not model_path.exists():
        return None, metadata, db_warning or "AI confidence model is not available."

    try:
        model = joblib.load(model_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return None, metadata, f"AI confidence model could not be loaded: {type(exc).__name__}."

    metadata = dict(metadata)
    metadata.setdefault("source", "local")
    return model, metadata, db_warning


def save_ai_confidence_model_from_files(
    *,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    model_version: str = MODEL_VERSION,
) -> bool:
    """Save an existing local AI Confidence model artifact into the database."""
    if joblib is None or save_ai_confidence_model is None:
        return False
    if not model_path.exists():
        return False
    metadata = load_ai_confidence_metadata(metadata_path)
    model = joblib.load(model_path)
    feature_names = _feature_names(model, metadata)
    if not feature_names:
        return False
    import io

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return save_ai_confidence_model(
        model_bytes=buffer.getvalue(),
        feature_names=feature_names,
        trained_at=str(metadata.get("trained_at")) if metadata.get("trained_at") else None,
        model_version=str(metadata.get("model_version") or model_version),
        metadata=metadata,
    )


def _apply_calibration_map(proba: Any, calibration_map: Any) -> Any:
    """Map raw probabilities through a stored isotonic step-map via np.interp.

    Kept local (pure numpy) so the hot scoring path doesn't import the heavy
    ml_prebreakout module. Returns the input unchanged when no usable map is
    present, so scoring is safe on models trained before calibration existed.
    """
    values = np.asarray(proba, dtype=float)
    if not isinstance(calibration_map, dict):
        return values
    xs = calibration_map.get("x")
    ys = calibration_map.get("y")
    if not xs or not ys or len(xs) != len(ys) or len(xs) < 2:
        return values
    return np.clip(np.interp(values, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)), 0.0, 1.0)


def recalibrate_active_ai_confidence() -> dict:
    """Attach an isotonic calibration map to the live AI Confidence model.

    The live model may carry no calibration map (e.g. trained before calibration
    existed), so its displayed confidence is the raw, over-confident score. This
    fits a map from the model's own stored ``calibration`` buckets and patches
    the active row's metadata in place — no retrain, no model swap. Idempotent:
    skips when a map is already present. Returns a small status dict.
    """
    if update_active_ai_confidence_model_metadata is None:
        return {"ok": False, "reason": "db metadata update helper unavailable"}
    model, metadata, warning = load_ai_confidence_bundle()
    if model is None:
        return {"ok": False, "reason": warning or "no AI Confidence model available"}
    if metadata.get("source") != "database":
        return {"ok": False, "reason": f"active model source is {metadata.get('source')!r}, not database"}
    existing = metadata.get("calibration_map")
    if isinstance(existing, dict) and existing.get("x"):
        return {"ok": True, "skipped": "model already has a calibration_map", "method": existing.get("method")}
    try:
        from ml_prebreakout import fit_isotonic_calibration_map_from_buckets
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "reason": f"calibration fitter unavailable: {type(exc).__name__}"}
    calibration_map = fit_isotonic_calibration_map_from_buckets(metadata.get("calibration") or [])
    if not calibration_map:
        return {"ok": False, "reason": "model calibration buckets too sparse to fit a map"}
    if not update_active_ai_confidence_model_metadata({"calibration_map": calibration_map}):
        return {"ok": False, "reason": "metadata update failed (database unavailable?)"}
    _BUNDLE_CACHE.clear()
    return {
        "ok": True,
        "model_version": metadata.get("model_version"),
        "calibration_points": len(calibration_map["x"]),
        "n": calibration_map["n"],
    }


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
    model, metadata, load_warning = load_ai_confidence_bundle(
        model_path=model_path,
        metadata_path=metadata_path,
    )
    trained_at = metadata.get("trained_at")
    trained_at = str(trained_at) if trained_at else None

    if model is None:
        return _warn(frame, load_warning or "AI confidence model is not available.", trained_at=trained_at)

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
        # Isotonic calibration (fit on OOF validation) makes the displayed
        # confidence honest — a shown 45% actually hits ~45% — instead of the
        # raw, over-confident score. Monotonic, so it never changes ranking.
        # No-ops on models trained before calibration was attached.
        calibrated = _apply_calibration_map(proba[:, 1], metadata.get("calibration_map"))
        frame[CONFIDENCE_COL] = (calibrated * 100.0).round(1)
        if trained_at:
            frame.attrs[TRAINED_AT_ATTR] = trained_at
        if metadata.get("source"):
            frame.attrs[SOURCE_ATTR] = metadata.get("source")
        if metadata.get("calibration"):
            frame.attrs[CALIBRATION_ATTR] = metadata.get("calibration")
        if metadata.get("target_rule") or metadata.get("target"):
            frame.attrs[TARGET_RULE_ATTR] = metadata.get("target_rule") or metadata.get("target")
        return frame.sort_values(CONFIDENCE_COL, ascending=False).reset_index(drop=True)
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
        return _warn(frame, f"AI confidence scoring failed: {type(exc).__name__}.", trained_at=trained_at)
