"""Read-only AI Confidence evidence for the web Admin (Models tab).

Restores what the Streamlit admin showed under Scanner results: the model card
(trained, target, source, calibration, warnings), the latest scan's scores, and
the calibration buckets, plus a live check on matured signals.

Kept slim for the 512 MB API: only small SQL reads and aggregates. The model
file (model_bytes) is never selected and the model is never loaded here.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from db.engine import get_neon_conn
from db.ml_model_registry import RegistryUnavailable

LIVE_DAYS = 60
LATEST_LIMIT = 100
STALE_MODEL_DAYS = 120
TARGET_UP, TARGET_DOWN = 0.04, -0.02  # +4% before -2% in 5 trading days

# Same rule as db.signal_outcomes: a run with Scanner rows supersedes its Brief rows.
_NOT_SUPERSEDED = (
    "AND NOT (s.signal_type = 'hsf_opportunity' AND EXISTS ("
    "SELECT 1 FROM signal_outcomes s2 WHERE s2.source = 'opportunity' "
    "AND s2.signal_type = 'hsf_scanner' AND s2.fired_at = s.fired_at)) "
)

_MODEL_SQL = (
    "SELECT model_version, trained_at, created_at, metadata->>'dataset_version', "
    "COALESCE(metadata->>'target_rule', metadata->>'target'), metadata->'calibration', "
    "metadata->'calibration_map'->>'method', "
    "COALESCE(jsonb_array_length(metadata->'calibration_map'->'x'), 0), "
    "jsonb_array_length(feature_names) "
    "FROM ai_confidence_models WHERE is_active = TRUE ORDER BY created_at DESC, id DESC LIMIT 1"
)

_LATEST_SQL = (
    "SELECT s.ticker, s.ai_confidence, s.setup_score, s.raw_signal->>'hsf_score', "
    "s.indicators->>'primary_setup', s.indicators->>'chg_pct', "
    "COALESCE(s.raw_signal->'models'->'ai_confidence'->'loaded_artifact'->>'version', "
    "s.raw_signal->>'ai_confidence_model_version') "
    "FROM signal_outcomes s WHERE s.source = 'opportunity' AND s.ai_confidence IS NOT NULL "
    "AND s.fired_at = (SELECT MAX(fired_at) FROM signal_outcomes "
    "WHERE source = 'opportunity' AND ai_confidence IS NOT NULL) "
    + _NOT_SUPERSEDED
    + "ORDER BY s.ai_confidence DESC, s.ticker LIMIT %s"
)

_LATEST_AT_SQL = (
    "SELECT MAX(fired_at) FROM signal_outcomes WHERE source = 'opportunity' AND ai_confidence IS NOT NULL"
)

_LIVE_SQL = (
    "SELECT LEAST(FLOOR(s.ai_confidence / 10.0), 9)::int AS b, COUNT(*), AVG(s.ai_confidence), "
    "SUM(CASE WHEN s.mfe_5d >= %s AND s.mae_5d > %s THEN 1 ELSE 0 END), "
    "SUM(CASE WHEN s.mfe_5d < %s THEN 1 ELSE 0 END) "
    "FROM signal_outcomes s WHERE s.source = 'opportunity' AND s.ai_confidence IS NOT NULL "
    "AND s.fired_at >= NOW() - make_interval(days => %s) "
    "AND s.mfe_5d IS NOT NULL AND s.mae_5d IS NOT NULL "
    + _NOT_SUPERSEDED
    + "GROUP BY 1 ORDER BY 1"
)

_PENDING_SQL = (
    "SELECT COUNT(*) FROM signal_outcomes s WHERE s.source = 'opportunity' AND s.ai_confidence IS NOT NULL "
    "AND s.fired_at >= NOW() - make_interval(days => %s) AND (s.mfe_5d IS NULL OR s.mae_5d IS NULL) "
    + _NOT_SUPERSEDED
)

_TICKER_SQL = (
    "SELECT s.ticker, s.fired_at, s.ai_confidence, "
    "s.raw_signal->'models'->'ai_confidence'->'feature_names', "
    "s.raw_signal->'models'->'ai_confidence'->'feature_values' "
    "FROM signal_outcomes s WHERE s.source = 'opportunity' AND s.ai_confidence IS NOT NULL "
    "AND UPPER(s.ticker) = %s AND s.fired_at >= NOW() - make_interval(days => 14) "
    "ORDER BY s.fired_at DESC, s.id DESC LIMIT 1"
)


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def tier(pct: Optional[float]) -> str:
    """High / Medium / Low, the same cut points as the old admin explainer."""
    if pct is None:
        return "Unknown"
    return "High" if pct >= 40 else "Medium" if pct >= 25 else "Low"


def _as_utc(v: Any) -> Optional[dt.datetime]:
    if isinstance(v, dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    return None


def model_card(row: Optional[List[Any]], latest_versions: List[str], now: dt.datetime) -> Dict[str, Any]:
    """Shape the active model row (no model bytes) and plain-language warnings."""
    warnings: List[str] = []
    if not row:
        warnings.append("No active AI Confidence model in the database. Scans fall back to the local model file "
                        "if one ships with the code; otherwise AI Confidence is blank.")
        model = None
    else:
        version, trained_at, created_at, dataset, target, buckets, method, map_points, n_features = row
        buckets = buckets if isinstance(buckets, list) else []
        model = {
            "model_version": version, "trained_at": trained_at, "saved_at": created_at,
            "dataset_version": dataset, "target_rule": target, "features": n_features,
            "calibration_method": method if int(map_points or 0) > 0 else None,
            "calibration_points": int(map_points or 0),
            "training_buckets": [
                {"bucket": str(b.get("bucket")), "signals": int(b.get("n") or 0),
                 "mean_confidence": _num(b.get("mean_confidence")), "hit_rate": _num(b.get("hit_rate"))}
                for b in buckets if isinstance(b, dict)],
        }
        if not model["calibration_method"]:
            warnings.append("The live model has no calibration map, so the shown percentages are raw model "
                            "scores and run high. Run scripts/recalibrate_ai_confidence_model.py to attach one.")
        trained = _as_utc(trained_at)
        if trained and (now - trained).days > STALE_MODEL_DAYS:
            warnings.append(f"The model was trained {(now - trained).days} days ago.")
        served = sorted({v for v in latest_versions if v})
        if served and version and version not in served:
            warnings.append(f"The latest scan was scored by {', '.join(served)}, not the active database model "
                            f"{version}. The scanner may be using the local fallback file.")
    return {"model": model, "warnings": warnings}


def live_buckets(rows: List[List[Any]]) -> List[Dict[str, Any]]:
    """Matured signals by 10-point confidence bucket.

    Only the 5-day high/low is stored, not their order, so a signal that touched
    both +4% and -2% is "unclear". The hit rate is a range: clear hits only, up to
    clear hits plus unclear.
    """
    out = []
    for b, n, mean_conf, hits, misses in rows:
        n, hits, misses = int(n or 0), int(hits or 0), int(misses or 0)
        if n <= 0:
            continue
        unclear = max(n - hits - misses, 0)
        lo = int(b) * 10
        out.append({"bucket": f"{lo}-{lo + 10}%", "signals": n, "mean_confidence": _num(mean_conf),
                    "hits": hits, "misses": misses, "unclear": unclear,
                    "hit_rate_low": round(100.0 * hits / n, 1),
                    "hit_rate_high": round(100.0 * (hits + unclear) / n, 1)})
    return out


def _conn():
    try:
        conn = get_neon_conn()
    except Exception as exc:
        raise RegistryUnavailable("database unavailable") from exc
    if conn is None:
        raise RegistryUnavailable("database unavailable")
    return conn


def _tables(cur) -> tuple:
    cur.execute("SELECT to_regclass('ai_confidence_models'), to_regclass('signal_outcomes')")
    row = cur.fetchone()
    return tuple(row.values()) if isinstance(row, dict) else tuple(row)


def _rows(cur) -> List[List[Any]]:
    return [list(r.values()) if isinstance(r, dict) else list(r) for r in cur.fetchall()]


def summary(now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """Model card, latest scan scores and live calibration in one small read."""
    now = now or dt.datetime.now(dt.timezone.utc)
    conn = _conn()
    cur = None
    try:
        cur = conn.cursor()
        has_models, has_signals = _tables(cur)
        model_row = None
        if has_models:
            cur.execute(_MODEL_SQL)
            got = _rows(cur)
            model_row = got[0] if got else None
        latest_at, latest, live, pending = None, [], [], 0
        if has_signals:
            cur.execute(_LATEST_AT_SQL)
            got = _rows(cur)
            latest_at = got[0][0] if got else None
            if latest_at is not None:
                cur.execute(_LATEST_SQL, (LATEST_LIMIT,))
                latest = _rows(cur)
            cur.execute(_LIVE_SQL, (TARGET_UP, TARGET_DOWN, TARGET_UP, LIVE_DAYS))
            live = _rows(cur)
            cur.execute(_PENDING_SQL, (LIVE_DAYS,))
            got = _rows(cur)
            pending = int(got[0][0] or 0) if got else 0
    except RegistryUnavailable:
        raise
    except Exception as exc:
        raise RegistryUnavailable("AI Confidence evidence unavailable") from exc
    finally:
        if cur is not None:
            cur.close()
        conn.close()

    card = model_card(model_row, [r[6] for r in latest], now)
    if latest_at is None:
        card["warnings"].append("No scan has saved an AI Confidence score yet.")
    else:
        age_h = (now - (_as_utc(latest_at) or now)).total_seconds() / 3600
        if age_h > 96:
            card["warnings"].append(f"The last saved AI Confidence score is {int(age_h // 24)} days old.")
    scores = [{"ticker": r[0], "ai_confidence": _num(r[1]), "tier": tier(_num(r[1])),
               "breakout_score": _num(r[2]), "hsf_score": _num(r[3]), "setup": r[4],
               "chg_pct": _num(r[5])} for r in latest]
    buckets = live_buckets(live)
    return {
        **card,
        "latest": {"scanned_at": latest_at, "rows": scores,
                   "tiers": {t: sum(1 for s in scores if s["tier"] == t) for t in ("High", "Medium", "Low")}},
        "live": {"days": LIVE_DAYS, "matured": sum(b["signals"] for b in buckets), "pending": pending,
                 "buckets": buckets},
        "note": "Training buckets come from the validation set at train time (raw model scores). The live check "
                "uses AI Confidence as served on saved scan signals that have had 5 trading days to play out.",
    }


def ticker_features(ticker: str) -> Optional[Dict[str, Any]]:
    """The latest saved AI Confidence for one ticker with its captured model inputs."""
    conn = _conn()
    cur = None
    try:
        cur = conn.cursor()
        if not _tables(cur)[1]:
            return None
        cur.execute(_TICKER_SQL, (ticker.upper(),))
        got = _rows(cur)
    except Exception as exc:
        raise RegistryUnavailable("AI Confidence evidence unavailable") from exc
    finally:
        if cur is not None:
            cur.close()
        conn.close()
    if not got:
        return None
    sym, fired_at, conf, names, values = got[0]
    feats: Dict[str, float] = {}
    if isinstance(names, list) and isinstance(values, list) and len(names) == len(values):
        feats = {str(k): f for k, v in zip(names, values) if (f := _num(v)) is not None}
    return {"ticker": sym, "scanned_at": fired_at, "ai_confidence": _num(conf), "features": feats}
