"""Run 25 — canonical persistence + read-only aggregation for HSF opportunity
outcome intelligence. Separate from alert-quality (Run 24) and from the price
pipeline (Step 29/30). Idempotent, first-observation immutable, read side pure.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from db.engine import get_neon_conn

# Comparable outcome-rate denominators exclude these (Step 20). PENDING/
# UNAVAILABLE are never persisted, so only VERSION_CHANGED must be excluded.
_NON_COMPARABLE = ("VERSION_CHANGED",)
# Favorable = initial state held or improved; unfavorable = deteriorated/left.
_FAVORABLE = ("STRENGTHENED", "PERSISTED", "RECOVERED")
_UNFAVORABLE = ("WEAKENED", "FADED", "DROPPED")
MIN_OUTCOME_SAMPLE = 10  # mirrors analytics.alert_quality.MIN_QUALITY_SAMPLE

# SQL band expression mirrors analytics.opportunity_outcomes.SCORE_BANDS.
_BAND_CASE = (
    "CASE WHEN initial_score >= 80 THEN '80+' "
    "WHEN initial_score >= 70 THEN '70-79' "
    "WHEN initial_score >= 60 THEN '60-69' ELSE '<60' END")
_SIGNAL_CASE = (
    "CASE WHEN initial_signal_count >= 3 THEN '3+' "
    "WHEN initial_signal_count = 2 THEN '2' ELSE '1' END")


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hsf_opportunity_outcomes (
            id BIGSERIAL PRIMARY KEY,
            opportunity_observation_id BIGINT NOT NULL,
            ticker TEXT NOT NULL,
            source_snapshot_time TIMESTAMPTZ,
            evaluation_horizon TEXT NOT NULL,
            evaluation_time TIMESTAMPTZ,
            data_status TEXT NOT NULL,
            outcome_classification TEXT NOT NULL,
            initial_score DOUBLE PRECISION,
            initial_status TEXT,
            initial_score_version TEXT,
            initial_fading BOOLEAN,
            initial_signal_count INTEGER,
            initial_regime TEXT,
            subsequent_score DOUBLE PRECISION,
            subsequent_status TEXT,
            subsequent_score_version TEXT,
            subsequent_fading BOOLEAN,
            still_present BOOLEAN,
            score_delta INTEGER,
            status_transition TEXT,
            evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (opportunity_observation_id, evaluation_horizon)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hoo_status_horizon "
                "ON hsf_opportunity_outcomes (initial_status, evaluation_horizon)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hoo_source_time "
                "ON hsf_opportunity_outcomes (source_snapshot_time DESC)")
    conn.commit()
    cur.close()


def persisted_outcome_keys(days_back: int = 30) -> Set[Tuple[int, str]]:
    """Already-matured (observation_id, horizon) keys — so maturation never
    recomputes or rewrites a recorded outcome."""
    conn = get_neon_conn()
    if conn is None:
        return set()
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT opportunity_observation_id, evaluation_horizon "
            "FROM hsf_opportunity_outcomes "
            "WHERE source_snapshot_time >= NOW() - make_interval(days => %s)",
            (int(days_back),),
        )
        rows = cur.fetchall() or []
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return set()
    out: Set[Tuple[int, str]] = set()
    for r in rows:
        oid, hz = (r[0], r[1]) if not isinstance(r, dict) else tuple(r.values())
        if oid is not None:
            out.add((int(oid), hz))
    return out


def persist_opportunity_outcome(res: Dict[str, Any]) -> bool:
    """Idempotently persist one matured outcome. True only when a NEW row was
    written (ON CONFLICT DO NOTHING preserves the first observation)."""
    if not res.get("opportunity_observation_id") or not res.get("evaluation_horizon"):
        return False
    conn = get_neon_conn()
    if conn is None:
        return False
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO hsf_opportunity_outcomes
              (opportunity_observation_id, ticker, source_snapshot_time, evaluation_horizon,
               evaluation_time, data_status, outcome_classification, initial_score,
               initial_status, initial_score_version, initial_fading, initial_signal_count,
               initial_regime, subsequent_score, subsequent_status, subsequent_score_version,
               subsequent_fading, still_present, score_delta, status_transition)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (opportunity_observation_id, evaluation_horizon) DO NOTHING
            """,
            (
                int(res["opportunity_observation_id"]), str(res.get("ticker") or "").upper(),
                res.get("source_snapshot_time"), res.get("evaluation_horizon"),
                res.get("evaluation_time"), res.get("data_status"),
                res.get("outcome_classification"), res.get("initial_score"),
                res.get("initial_status"), res.get("initial_score_version"),
                res.get("initial_fading"), res.get("initial_signal_count"),
                res.get("initial_regime"), res.get("subsequent_score"),
                res.get("subsequent_status"), res.get("subsequent_score_version"),
                res.get("subsequent_fading"), res.get("still_present"),
                res.get("score_delta"), res.get("status_transition"),
            ),
        )
        wrote = cur.rowcount == 1
        conn.commit()
        cur.close()
        conn.close()
        return wrote
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return False


def _rate(num: int, comparable: int, min_sample: int):
    return (num / comparable) if comparable >= int(min_sample) and comparable else None


def _roll(rows, key_idx: int, min_sample: int) -> List[Dict[str, Any]]:
    """Collapse (key, classification, count) rows into per-key summaries with a
    single comparable denominator (excludes VERSION_CHANGED). One rule, one place."""
    grouped: Dict[Any, Dict[str, int]] = {}
    for r in rows:
        vals = list(r.values()) if isinstance(r, dict) else list(r)
        key, cls, n = vals[key_idx], vals[key_idx + 1], int(vals[key_idx + 2] or 0)
        grouped.setdefault(key, {})[cls] = n
    out = []
    for key, d in grouped.items():
        matured = sum(d.values())
        comparable = sum(n for k, n in d.items() if k not in _NON_COMPARABLE)
        fav = sum(d.get(k, 0) for k in _FAVORABLE)
        unfav = sum(d.get(k, 0) for k in _UNFAVORABLE)
        assessment = "INSUFFICIENT_SAMPLE" if comparable < int(min_sample) else "OK"
        out.append({
            "key": key, "matured": matured, "comparable": comparable,
            "strengthened": d.get("STRENGTHENED", 0), "persisted": d.get("PERSISTED", 0),
            "weakened": d.get("WEAKENED", 0), "faded": d.get("FADED", 0),
            "dropped": d.get("DROPPED", 0), "recovered": d.get("RECOVERED", 0),
            "neutral": d.get("NEUTRAL", 0), "version_changed": d.get("VERSION_CHANGED", 0),
            "favorable_rate": _rate(fav, comparable, min_sample),
            "unfavorable_rate": _rate(unfav, comparable, min_sample),
            "assessment": assessment,
        })
    return out


def get_opportunity_outcome_summary(*, days_back: int = 90, min_sample: int = MIN_OUTCOME_SAMPLE) -> Dict[str, Any]:
    """Read-only aggregation — NO scanning/freezing/maturation/delivery/writes/
    Claude. Derived from persisted outcomes only. Dimensions: overall, by initial
    status, score band, confirming-signal count, market regime, and horizon."""
    empty = {"available": False, "matured": 0, "comparable": 0, "by_status": [],
             "by_score_band": [], "by_signal_count": [], "by_regime": [],
             "by_horizon": [], "min_sample": int(min_sample)}
    conn = get_neon_conn()
    if conn is None:
        return empty
    where = "WHERE data_status = 'MATURED' AND source_snapshot_time >= NOW() - make_interval(days => %s)"
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(f"SELECT outcome_classification, COUNT(*) FROM hsf_opportunity_outcomes {where} "
                    "GROUP BY outcome_classification", (int(days_back),))
        overall_rows = cur.fetchall() or []
        cur.execute(f"SELECT initial_status, outcome_classification, COUNT(*) FROM hsf_opportunity_outcomes "
                    f"{where} GROUP BY initial_status, outcome_classification", (int(days_back),))
        status_rows = cur.fetchall() or []
        cur.execute(f"SELECT {_BAND_CASE} AS band, outcome_classification, COUNT(*) FROM hsf_opportunity_outcomes "
                    f"{where} GROUP BY band, outcome_classification", (int(days_back),))
        band_rows = cur.fetchall() or []
        cur.execute(f"SELECT {_SIGNAL_CASE} AS bucket, outcome_classification, COUNT(*) FROM hsf_opportunity_outcomes "
                    f"{where} GROUP BY bucket, outcome_classification", (int(days_back),))
        signal_rows = cur.fetchall() or []
        cur.execute(f"SELECT COALESCE(initial_regime, 'UNKNOWN'), outcome_classification, COUNT(*) "
                    f"FROM hsf_opportunity_outcomes {where} GROUP BY initial_regime, outcome_classification",
                    (int(days_back),))
        regime_rows = cur.fetchall() or []
        cur.execute(f"SELECT evaluation_horizon, outcome_classification, COUNT(*) FROM hsf_opportunity_outcomes "
                    f"{where} GROUP BY evaluation_horizon, outcome_classification", (int(days_back),))
        horizon_rows = cur.fetchall() or []
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return empty

    overall: Dict[str, int] = {}
    for r in overall_rows:
        k, n = (r[0], int(r[1] or 0)) if not isinstance(r, dict) else tuple(r.values())
        overall[k] = n
    matured = sum(overall.values())
    comparable = sum(n for k, n in overall.items() if k not in _NON_COMPARABLE)
    return {
        "available": True, "matured": matured, "comparable": comparable,
        "overall_classifications": overall,
        "by_status": _roll(status_rows, 0, min_sample),
        "by_score_band": _roll(band_rows, 0, min_sample),
        "by_signal_count": _roll(signal_rows, 0, min_sample),
        "by_regime": _roll(regime_rows, 0, min_sample),
        "by_horizon": _roll(horizon_rows, 0, min_sample),
        "min_sample": int(min_sample),
    }


def get_similar_state_outcomes(
    *, status: Optional[str], score: Optional[float], horizon: str = "H24",
    days_back: int = 180, min_sample: int = MIN_OUTCOME_SAMPLE,
) -> Dict[str, Any]:
    """Read-only cohort summary for Stock Intelligence: outcomes of observations
    that shared this ticker's HSF state (same status + score band) at one horizon.
    Descriptive HSF-state persistence only — never a price probability."""
    from analytics.opportunity_outcomes import score_band

    band = score_band(score)
    result = {"available": False, "status": status, "score_band": band, "horizon": horizon,
              "matured": 0, "comparable": 0, "favorable_rate": None, "assessment": "INSUFFICIENT_SAMPLE"}
    if not status or band is None:
        return result
    conn = get_neon_conn()
    if conn is None:
        return result
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            f"SELECT outcome_classification, COUNT(*) FROM hsf_opportunity_outcomes "
            f"WHERE data_status = 'MATURED' AND evaluation_horizon = %s AND initial_status = %s "
            f"AND {_BAND_CASE} = %s AND source_snapshot_time >= NOW() - make_interval(days => %s) "
            f"GROUP BY outcome_classification",
            (horizon, status, band, int(days_back)),
        )
        rows = cur.fetchall() or []
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return result
    d: Dict[str, int] = {}
    for r in rows:
        k, n = (r[0], int(r[1] or 0)) if not isinstance(r, dict) else tuple(r.values())
        d[k] = n
    matured = sum(d.values())
    comparable = sum(n for k, n in d.items() if k not in _NON_COMPARABLE)
    fav = sum(d.get(k, 0) for k in _FAVORABLE)
    result.update({
        "available": comparable >= int(min_sample), "matured": matured, "comparable": comparable,
        "favorable_rate": _rate(fav, comparable, min_sample),
        "assessment": "OK" if comparable >= int(min_sample) else "INSUFFICIENT_SAMPLE",
    })
    return result
