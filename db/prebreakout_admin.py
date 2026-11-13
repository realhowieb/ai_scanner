"""Small read-only PreBreakout evidence queries; no model or training imports."""
from __future__ import annotations

import datetime as dt

from db.ai_confidence_admin import _as_utc, _conn, _num, _rows
from db.ml_model_registry import RegistryUnavailable

_MODEL_SQL = (
    "SELECT model_version, trained_at, created_at, metadata->>'dataset_version' AS dataset_version, "
    "metadata->>'target_rule' AS target_rule, metadata->'calibration' AS calibration, "
    "metadata->'calibration_map'->>'method' AS calibration_method, "
    "COALESCE(jsonb_array_length(metadata->'calibration_map'->'x'),0) AS calibration_points, "
    "jsonb_array_length(feature_names) AS feature_count, auc "
    "FROM prebreakout_models WHERE is_active=TRUE ORDER BY created_at DESC,id DESC LIMIT 1"
)
# Missing/failed captures may carry a zero placeholder. Never present that as a prediction.
_CAPTURED = "(s.raw_signal->'models'->'prebreakout'->>'status'='captured' OR " \
    "(s.raw_signal->'models'->'prebreakout'->>'status' IS NULL " \
    "AND s.raw_signal->>'prebreakout_model_version' IS NOT NULL AND s.prebreakout_prob>0))"
_NOT_SUPERSEDED = "AND NOT (s.signal_type='hsf_opportunity' AND EXISTS (SELECT 1 FROM signal_outcomes s2 " \
    "WHERE s2.source='opportunity' AND s2.signal_type='hsf_scanner' AND s2.fired_at=s.fired_at)) "
_LATEST_SQL = (
    "SELECT s.ticker, CASE WHEN " + _CAPTURED + " THEN s.prebreakout_prob END AS probability, "
    "s.raw_signal->>'hsf_score' AS hsf_score, s.setup_score, s.indicators->>'primary_setup' AS setup, "
    "COALESCE(s.raw_signal->'models'->'prebreakout'->'loaded_artifact'->>'version', "
    "s.raw_signal->>'prebreakout_model_version') AS model_version "
    "FROM signal_outcomes s WHERE s.source='opportunity' AND s.fired_at=%s "
    + _NOT_SUPERSEDED + "ORDER BY probability DESC NULLS LAST,s.ticker LIMIT 100"
)
_INPUT_SQL = (
    "SELECT s.ticker,s.fired_at,s.prebreakout_prob, "
    "s.raw_signal->'models'->'prebreakout'->'feature_names' AS names, "
    "s.raw_signal->'models'->'prebreakout'->'feature_values' AS values, "
    "s.raw_signal->'models'->'prebreakout'->'default_mask' AS defaults, "
    "s.raw_signal->'models'->'prebreakout'->'loaded_artifact'->>'version' AS model_version "
    "FROM signal_outcomes s WHERE s.source='opportunity' AND UPPER(s.ticker)=%s "
    "AND (%s::timestamptz IS NULL OR s.fired_at=%s::timestamptz) AND s.fired_at>=NOW()-INTERVAL '14 days' AND " + _CAPTURED + " " + _NOT_SUPERSEDED +
    "ORDER BY s.fired_at DESC,s.id DESC LIMIT 1"
)


def _read(fn):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            return fn(cur)
    except Exception as exc:
        raise RegistryUnavailable("PreBreakout evidence unavailable") from exc
    finally:
        conn.close()


def summary(now=None):
    now = now or dt.datetime.now(dt.timezone.utc)

    def load(cur):
        cur.execute("SELECT to_regclass('prebreakout_models') AS models, to_regclass('signal_outcomes') AS signals")
        models, signals = _rows_one(cur)
        model, latest_at, rows, warnings = None, None, [], []
        if models:
            cur.execute(_MODEL_SQL)
            saved = _rows(cur)
            if saved:
                version, trained, created, dataset, target, buckets, method, points, features, auc = saved[0]
                model = {"model_version": version, "trained_at": trained, "saved_at": created,
                         "dataset_version": dataset, "target_rule": target, "features": features,
                         "calibration_method": method if points else None, "validation_auc": _num(auc),
                         "training_buckets": [{"bucket": str(b.get("bucket")), "signals": int(b.get("n") or 0),
                            "mean_confidence": _num(b.get("mean_confidence")), "hit_rate": _num(b.get("hit_rate"))}
                            for b in (buckets if isinstance(buckets, list) else []) if isinstance(b, dict)]}
                if not model["calibration_method"]:
                    warnings.append("No calibration map recorded. Percentages may be raw model scores.")
                trained = _as_utc(trained)
                if trained and (now-trained).days > 120:
                    warnings.append(f"The active model was trained {(now-trained).days} days ago.")
        if not model:
            warnings.append("No active database model recorded. A local fallback may still score scans.")
        if signals:
            cur.execute("SELECT MAX(fired_at) AS scanned_at FROM signal_outcomes WHERE source='opportunity'")
            latest_at = _rows_one(cur)[0]
            if latest_at:
                cur.execute(_LATEST_SQL, (latest_at,))
                rows = [{"ticker": r[0], "probability": _num(r[1]), "hsf_score": _num(r[2]),
                         "breakout_score": _num(r[3]), "setup": r[4], "model_version": r[5]} for r in _rows(cur)]
                if (now-(_as_utc(latest_at) or now)).total_seconds() > 96*3600:
                    warnings.append("The latest saved scan is over four days old.")
        if not latest_at:
            warnings.append("No saved opportunity scan is available.")
        elif not any(r["probability"] is not None for r in rows):
            warnings.append("The latest scan has no captured PreBreakout probabilities.")
        if any(r["probability"] is not None and not r["model_version"] for r in rows):
            warnings.append("Some saved probabilities have no recorded model identity.")
        versions = sorted({r["model_version"] for r in rows if r["probability"] is not None and r["model_version"]})
        if model and any(v != model["model_version"] for v in versions):
            warnings.append("Saved scores include a different model version: " + ", ".join(versions))
        return {"model": model, "warnings": warnings, "latest": {"scanned_at": latest_at, "rows": rows},
                "live_validation": "Not established here. PreBreakout needs a future qualifying setup and its "
                "subsequent price path. Five-day high/low outcomes alone cannot verify that target. "
                "Training calibration is historical validation evidence, not live performance."}
    return _read(load)


def _rows_one(cur):
    row = cur.fetchone()
    return list(row.values()) if isinstance(row, dict) else list(row)


def inputs(ticker, scanned_at=None):
    def load(cur):
        cur.execute("SELECT to_regclass('signal_outcomes') AS signals")
        if not _rows_one(cur)[0]:
            return None
        cur.execute(_INPUT_SQL, (ticker.upper(), scanned_at, scanned_at))
        rows = _rows(cur)
        if not rows:
            return None
        sym, at, probability, names, values, defaults, version = rows[0]
        fields = []
        if isinstance(names, list) and isinstance(values, list) and len(names) == len(values):
            for i, (name, value) in enumerate(zip(names, values)):
                defaulted = defaults[i] if isinstance(defaults, list) and len(defaults) == len(names) else None
                fields.append({"name": str(name), "value": _num(value),
                               "defaulted": defaulted if isinstance(defaulted, bool) else None})
        return {"ticker": sym, "scanned_at": at, "probability": _num(probability), "model_version": version,
                "inputs": fields, "note": "Inputs captured at prediction time. Defaulted values are substitutes, "
                "not observed market data. These inputs do not establish each feature's contribution to the score."}
    return _read(load)
