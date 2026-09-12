"""Storage for HSF Intelligence Alerts (state-change notifications) + prefs.

Separate from the static user_alerts system — never touches it. Dedupe is
state-aware + cooldown-windowed (not a hard UNIQUE), so a persistent state won't
re-fire every cron run yet a genuine later re-transition can notify again. Every
function is per-user scoped and non-fatal (returns a safe default when the DB is
unavailable).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from db.engine import get_neon_conn

_DEFAULT_COOLDOWN_HOURS = 6


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hsf_intelligence_alerts (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity TEXT,
            fingerprint TEXT NOT NULL,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            copy TEXT,
            delivery_status TEXT NOT NULL DEFAULT 'DETECTED',
            detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            delivered_at TIMESTAMPTZ
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hsf_ia_user_time "
                "ON hsf_intelligence_alerts (user_id, detected_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hsf_ia_user_fp "
                "ON hsf_intelligence_alerts (user_id, fingerprint, detected_at DESC)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hsf_alert_prefs (
            user_id TEXT PRIMARY KEY,
            prefs JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS intelligence_evaluation_runs (
            id BIGSERIAL PRIMARY KEY,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            status TEXT NOT NULL,
            duration_ms INTEGER,
            previous_snapshot_time TIMESTAMPTZ,
            current_snapshot_time TIMESTAMPTZ,
            snapshots_loaded INTEGER DEFAULT 0,
            events_detected INTEGER DEFAULT 0,
            users_evaluated INTEGER DEFAULT 0,
            watchlist_tickers_evaluated INTEGER DEFAULT 0,
            notifications_matched INTEGER DEFAULT 0,
            filtered_by_preferences INTEGER DEFAULT 0,
            deduped INTEGER DEFAULT 0,
            persisted INTEGER DEFAULT 0,
            delivered INTEGER DEFAULT 0,
            failed INTEGER DEFAULT 0,
            reason TEXT,
            error_stage TEXT,
            error_type TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hsf_eval_runs_time "
                "ON intelligence_evaluation_runs (started_at DESC)")
    conn.commit()
    cur.close()


_RUN_COUNT_FIELDS = (
    "snapshots_loaded", "events_detected", "users_evaluated", "watchlist_tickers_evaluated",
    "notifications_matched", "filtered_by_preferences", "deduped", "persisted", "delivered", "failed",
)


def _parse_ts(v):
    """Accept a datetime or a str timestamp; None/'None' -> NULL."""
    if v is None or v == "None" or v == "":
        return None
    return v


def record_evaluation_run(result: Dict[str, Any]) -> Optional[int]:
    """Persist one operational evaluation-run record. Returns id or None."""
    conn = get_neon_conn()
    if conn is None:
        return None
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO intelligence_evaluation_runs
              (status, duration_ms, previous_snapshot_time, current_snapshot_time,
               snapshots_loaded, events_detected, users_evaluated, watchlist_tickers_evaluated,
               notifications_matched, filtered_by_preferences, deduped, persisted, delivered, failed,
               reason, error_stage, error_type)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (
                result.get("status"), result.get("duration_ms"),
                _parse_ts(result.get("previous_snapshot_time")),
                _parse_ts(result.get("current_snapshot_time")),
                *[int(result.get(f) or 0) for f in _RUN_COUNT_FIELDS],
                result.get("reason"), result.get("error_stage"), result.get("error_type"),
            ),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return int(row[0] if not isinstance(row, dict) else row["id"]) if row else None
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None


def list_recent_evaluation_runs(limit: int = 15) -> List[Dict[str, Any]]:
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT started_at, status, events_detected, notifications_matched, "
            "delivered, failed, deduped, filtered_by_preferences, duration_ms, "
            "error_stage, reason FROM intelligence_evaluation_runs "
            "ORDER BY started_at DESC LIMIT %s",
            (int(limit),),
        )
        rows = cur.fetchall() or []
        cols = [d[0] for d in cur.description] if cur.description else []
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return []
    return [dict(r) if isinstance(r, dict) else dict(zip(cols, r)) for r in rows]


# A full weekday without a successful evaluation is genuinely stale given the
# ~4 scheduled cron runs/weekday (overnight/weekend gaps are expected, so the
# threshold is generous to avoid false staleness).
INTELLIGENCE_STALE_AFTER_MINUTES = 24 * 60


def get_intelligence_health(stale_after_minutes: int = INTELLIGENCE_STALE_AFTER_MINUTES) -> Dict[str, Any]:
    """Read-only operational health — NO evaluation, NO writes, NO delivery, NO
    market rebuild, NO Claude. Derived from persisted run records only."""
    conn = get_neon_conn()
    if conn is None:
        return {"status": "UNKNOWN", "reason": "database unavailable"}
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT started_at, status, events_detected, notifications_matched, delivered, "
            "failed, deduped, filtered_by_preferences, users_evaluated, error_stage, reason, duration_ms "
            "FROM intelligence_evaluation_runs ORDER BY started_at DESC LIMIT 1")
        latest = cur.fetchone()
        cur.execute("SELECT MAX(started_at) FROM intelligence_evaluation_runs WHERE status IN ('SUCCESS','PARTIAL')")
        last_success_row = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM intelligence_evaluation_runs "
                    "WHERE status = 'FAILED' AND started_at >= NOW() - make_interval(hours => 24)")
        recent_fail_row = cur.fetchone()
        cur.execute("SELECT EXTRACT(EPOCH FROM (NOW() - MAX(started_at)))/60 FROM intelligence_evaluation_runs")
        since_row = cur.fetchone()
        cur.execute("SELECT EXTRACT(EPOCH FROM (NOW() - MAX(started_at)))/60 "
                    "FROM intelligence_evaluation_runs WHERE status IN ('SUCCESS','PARTIAL')")
        since_success_row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return {"status": "UNKNOWN", "reason": "health query failed"}

    def g(row, i):
        if not row:
            return None
        return row[i] if not isinstance(row, dict) else list(row.values())[i]

    if not latest:
        return {"status": "UNKNOWN", "reason": "no evaluation has run yet",
                "last_run_at": None, "last_success_at": None}

    latest_status = g(latest, 1)
    last_success_at = g(last_success_row, 0)
    recent_failures = int(g(recent_fail_row, 0) or 0)
    mins_since_run = g(since_row, 0)
    mins_since_success = g(since_success_row, 0)
    latest_metrics = {
        "events_detected": g(latest, 2), "notifications_matched": g(latest, 3),
        "delivered": g(latest, 4), "failed": g(latest, 5), "deduped": g(latest, 6),
        "filtered_by_preferences": g(latest, 7), "users_evaluated": g(latest, 8),
    }

    # Deterministic status precedence.
    if last_success_at is None:
        status = "DEGRADED"
        reason = "runs recorded but none succeeded"
    elif mins_since_success is not None and float(mins_since_success) > float(stale_after_minutes):
        status = "STALE"
        reason = f"no successful evaluation in over {int(stale_after_minutes/60)}h"
    elif latest_status in ("FAILED", "PARTIAL") or recent_failures > 0:
        status = "DEGRADED"
        reason = g(latest, 10) or f"latest run {latest_status}"
    else:
        status = "HEALTHY"
        reason = None

    return {
        "status": status, "reason": reason,
        "last_run_at": g(latest, 0), "last_success_at": last_success_at,
        "minutes_since_last_run": round(float(mins_since_run), 1) if mins_since_run is not None else None,
        "minutes_since_last_success": round(float(mins_since_success), 1) if mins_since_success is not None else None,
        "latest_status": latest_status, "error_stage": g(latest, 9),
        "recent_failure_count": recent_failures,
        "delivery_failure_count": int(latest_metrics["failed"] or 0),
        "latest_metrics": latest_metrics,
    }


# ---------------------------------------------------------------------------
# Run 24 — alert QUALITY measurement persistence (separate from health).
# Stores subsequent-HSF-state outcomes for each alert/horizon. Idempotent per
# (alert_id, evaluation_horizon). Never price, never Claude. Read side is pure.
# ---------------------------------------------------------------------------


def _ensure_quality_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS intelligence_alert_outcomes (
            id BIGSERIAL PRIMARY KEY,
            alert_id BIGINT NOT NULL,
            ticker TEXT NOT NULL,
            event_type TEXT NOT NULL,
            alert_time TIMESTAMPTZ,
            evaluation_horizon TEXT NOT NULL,
            evaluation_time TIMESTAMPTZ,
            data_status TEXT NOT NULL,
            quality_classification TEXT NOT NULL,
            alert_score DOUBLE PRECISION,
            subsequent_score DOUBLE PRECISION,
            score_delta INTEGER,
            alert_status TEXT,
            subsequent_status TEXT,
            still_present BOOLEAN,
            fading BOOLEAN,
            score_version TEXT,
            evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (alert_id, evaluation_horizon)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_iao_event_horizon "
                "ON intelligence_alert_outcomes (event_type, evaluation_horizon)")
    conn.commit()
    cur.close()


def fetch_alerts_for_maturation(days_back: int = 30, limit: int = 2000) -> List[Dict[str, Any]]:
    """Frozen intelligence alerts (all users) for background quality maturation.

    Returns {id, ticker, event_type, alert_time, payload(note)} oldest-first.
    The payload already holds the signal-time HSF state (no reconstruction from
    current market data — leakage-safe). Never raises; [] if DB unavailable.
    """
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, ticker, event_type, detected_at, payload "
            "FROM hsf_intelligence_alerts "
            "WHERE detected_at >= NOW() - make_interval(days => %s) "
            "ORDER BY detected_at ASC LIMIT %s",
            (int(days_back), int(limit)),
        )
        rows = cur.fetchall() or []
        cols = [d[0] for d in cur.description] if cur.description else []
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return []
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r) if isinstance(r, dict) else dict(zip(cols, r))
        payload = d.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        out.append({
            "id": d.get("id"), "ticker": d.get("ticker"),
            "event_type": d.get("event_type"), "alert_time": d.get("detected_at"),
            "payload": payload if isinstance(payload, dict) else {},
        })
    return out


def persist_alert_outcome(res: Dict[str, Any]) -> bool:
    """Idempotently persist one matured alert/horizon outcome. Returns True only
    when a new row was written (ON CONFLICT DO NOTHING)."""
    if not res.get("alert_id") or not res.get("evaluation_horizon"):
        return False
    conn = get_neon_conn()
    if conn is None:
        return False
    try:
        _ensure_quality_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO intelligence_alert_outcomes
              (alert_id, ticker, event_type, alert_time, evaluation_horizon,
               evaluation_time, data_status, quality_classification, alert_score,
               subsequent_score, score_delta, alert_status, subsequent_status,
               still_present, fading, score_version)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (alert_id, evaluation_horizon) DO NOTHING
            """,
            (
                int(res["alert_id"]), str(res.get("ticker") or "").upper(),
                res.get("event_type"), _parse_ts(res.get("alert_time")),
                res.get("evaluation_horizon"), _parse_ts(res.get("evaluation_time")),
                res.get("data_status"), res.get("quality_classification"),
                res.get("alert_score"), res.get("subsequent_score"), res.get("score_delta"),
                res.get("alert_status"), res.get("subsequent_status"),
                res.get("still_present"), res.get("fading"), res.get("score_version"),
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


# Quality-rate numerator sets (deterministic). RECOVERED/DETERIORATED/NEUTRAL are
# reported on their own — they are neither a confirmation nor a clean reversal.
_CONFIRM_SET = ("CONFIRMED", "PERSISTED")
_REVERSE_SET = ("REVERSED",)
# Horizon offsets must mirror analytics.alert_quality.HORIZONS (hours).
_QUALITY_HORIZON_HOURS = (("NEXT", 0), ("D1", 24), ("D3", 72), ("D5", 120))


def get_alert_quality_summary(*, days_back: int = 60, min_sample: int = 10) -> Dict[str, Any]:
    """Read-only quality aggregation — NO scanning, NO freezing, NO evaluation,
    NO delivery, NO writes, NO Claude. Derived from persisted outcomes + frozen
    alerts only. Deterministic. Separate from operational health (Run 23)."""
    empty = {"available": False, "alerts_total": 0, "matured": 0, "pending": 0,
             "unavailable": 0, "confirmed": 0, "reversed": 0,
             "confirmation_rate": None, "reversal_rate": None,
             "by_event_type": [], "by_horizon": [], "frequency": {},
             "min_sample": int(min_sample)}
    conn = get_neon_conn()
    if conn is None:
        return empty
    try:
        _ensure_schema(conn)
        _ensure_quality_schema(conn)
        cur = conn.cursor()

        # Expected vs matured pairs -> pending / unavailable (per-horizon offsets).
        hz_values = ",".join(f"({h})" for _, h in _QUALITY_HORIZON_HOURS)
        cur.execute(
            f"""
            SELECT
              COUNT(*) AS total_pairs,
              SUM(CASE WHEN NOW() >= a.detected_at + make_interval(hours => hz.h)
                       THEN 1 ELSE 0 END) AS elapsed_pairs,
              COUNT(DISTINCT a.id) AS alerts_total
            FROM hsf_intelligence_alerts a
            CROSS JOIN (VALUES {hz_values}) AS hz(h)
            WHERE a.event_type <> 'VERSION_CHANGED'
              AND a.detected_at >= NOW() - make_interval(days => %s)
            """,
            (int(days_back),),
        )
        pair_row = cur.fetchone() or (0, 0, 0)

        # Overall matured outcomes by classification.
        cur.execute(
            "SELECT quality_classification, COUNT(*) FROM intelligence_alert_outcomes "
            "WHERE data_status = 'MATURED' AND alert_time >= NOW() - make_interval(days => %s) "
            "GROUP BY quality_classification",
            (int(days_back),),
        )
        overall_rows = cur.fetchall() or []

        # By event type (matured only).
        cur.execute(
            "SELECT event_type, quality_classification, COUNT(*) FROM intelligence_alert_outcomes "
            "WHERE data_status = 'MATURED' AND alert_time >= NOW() - make_interval(days => %s) "
            "GROUP BY event_type, quality_classification",
            (int(days_back),),
        )
        event_rows = cur.fetchall() or []

        # By horizon (matured only).
        cur.execute(
            "SELECT evaluation_horizon, quality_classification, COUNT(*) FROM intelligence_alert_outcomes "
            "WHERE data_status = 'MATURED' AND alert_time >= NOW() - make_interval(days => %s) "
            "GROUP BY evaluation_horizon, quality_classification",
            (int(days_back),),
        )
        horizon_rows = cur.fetchall() or []

        # Frequency / noise metrics.
        cur.execute(
            "SELECT event_type, COUNT(*) FROM hsf_intelligence_alerts "
            "WHERE detected_at >= NOW() - make_interval(days => %s) GROUP BY event_type",
            (int(days_back),),
        )
        dist_rows = cur.fetchall() or []
        cur.execute(
            "SELECT COUNT(DISTINCT ticker), "
            "COUNT(*) FILTER (WHERE delivery_status='DELIVERED') "
            "FROM hsf_intelligence_alerts WHERE detected_at >= NOW() - make_interval(days => %s)",
            (int(days_back),),
        )
        tick_row = cur.fetchone() or (0, 0)
        cur.execute(
            "SELECT COALESCE(SUM(notifications_matched),0), COALESCE(SUM(deduped),0), "
            "COALESCE(SUM(filtered_by_preferences),0), COUNT(*) "
            "FROM intelligence_evaluation_runs WHERE started_at >= NOW() - make_interval(days => %s)",
            (int(days_back),),
        )
        run_row = cur.fetchone() or (0, 0, 0, 0)
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return empty

    def _v(row, i):
        return (row[i] if not isinstance(row, dict) else list(row.values())[i]) if row else None

    total_pairs = int(_v(pair_row, 0) or 0)
    elapsed_pairs = int(_v(pair_row, 1) or 0)
    alerts_total = int(_v(pair_row, 2) or 0)

    def _pair(r):
        return (r[0], r[1], int(r[2] or 0)) if not isinstance(r, dict) else tuple(r.values())

    overall: Dict[str, int] = {}
    for r in overall_rows:
        k, n = (r[0], int(r[1] or 0)) if not isinstance(r, dict) else tuple(r.values())
        overall[k] = n
    matured = sum(overall.values())
    confirmed = sum(overall.get(k, 0) for k in _CONFIRM_SET)
    reversed_ = sum(overall.get(k, 0) for k in _REVERSE_SET)
    pending = max(total_pairs - elapsed_pairs, 0)
    unavailable = max(elapsed_pairs - matured, 0)

    def _assess(m: int, conf: int, rev: int) -> str:
        if m < int(min_sample):
            return "INSUFFICIENT_SAMPLE"
        cr = conf / m if m else 0
        rr = rev / m if m else 0
        if cr >= 0.6 and rr <= 0.25:
            return "Promising"
        if rr >= 0.5:
            return "High reversal"
        return "Mixed"

    by_event: Dict[str, Dict[str, int]] = {}
    for r in event_rows:
        et, cls, n = _pair(r)
        by_event.setdefault(et, {})[cls] = n
    by_event_type = []
    for et, d in sorted(by_event.items()):
        m = sum(d.values())
        conf = sum(d.get(k, 0) for k in _CONFIRM_SET)
        rev = sum(d.get(k, 0) for k in _REVERSE_SET)
        by_event_type.append({
            "event_type": et, "matured": m, "confirmed": conf, "reversed": rev,
            "recovered": d.get("RECOVERED", 0), "deteriorated": d.get("DETERIORATED", 0),
            "neutral": d.get("NEUTRAL", 0),
            "confirmation_rate": (conf / m) if m >= int(min_sample) else None,
            "assessment": _assess(m, conf, rev),
        })

    by_h: Dict[str, Dict[str, int]] = {}
    for r in horizon_rows:
        hz, cls, n = _pair(r)
        by_h.setdefault(hz, {})[cls] = n
    order = {name: i for i, (name, _) in enumerate(_QUALITY_HORIZON_HOURS)}
    by_horizon = []
    for hz, d in sorted(by_h.items(), key=lambda kv: order.get(kv[0], 99)):
        m = sum(d.values())
        conf = sum(d.get(k, 0) for k in _CONFIRM_SET)
        rev = sum(d.get(k, 0) for k in _REVERSE_SET)
        by_horizon.append({
            "horizon": hz, "matured": m, "confirmed": conf, "reversed": rev,
            "confirmation_rate": (conf / m) if m >= int(min_sample) else None,
            "assessment": _assess(m, conf, rev),
        })

    dist = {}
    for r in dist_rows:
        k, n = (r[0], int(r[1] or 0)) if not isinstance(r, dict) else tuple(r.values())
        dist[k] = n
    distinct_tickers = int(_v(tick_row, 0) or 0)
    delivered_total = int(_v(tick_row, 1) or 0)
    matched_sum = int(_v(run_row, 0) or 0)
    deduped_sum = int(_v(run_row, 1) or 0)
    filtered_sum = int(_v(run_row, 2) or 0)
    run_count = int(_v(run_row, 3) or 0)
    denom = matched_sum + deduped_sum
    frequency = {
        "alerts_total": alerts_total,
        "distinct_tickers": distinct_tickers,
        "alerts_per_ticker": (alerts_total / distinct_tickers) if distinct_tickers else None,
        "delivered_total": delivered_total,
        "event_type_distribution": dist,
        "dedupe_rate": (deduped_sum / denom) if denom else None,
        "filter_rate": (filtered_sum / matched_sum) if matched_sum else None,
        "runs_considered": run_count,
    }

    return {
        "available": True, "alerts_total": alerts_total, "matured": matured,
        "pending": pending, "unavailable": unavailable,
        "confirmed": confirmed, "reversed": reversed_,
        "confirmation_rate": (confirmed / matured) if matured >= int(min_sample) else None,
        "reversal_rate": (reversed_ / matured) if matured >= int(min_sample) else None,
        "by_event_type": by_event_type, "by_horizon": by_horizon,
        "frequency": frequency, "min_sample": int(min_sample),
        "overall_classifications": overall,
    }


def recent_fingerprints(user_id: str, *, hours: int = _DEFAULT_COOLDOWN_HOURS) -> Set[str]:
    """Fingerprints already recorded for this user within the cooldown window."""
    if not user_id:
        return set()
    conn = get_neon_conn()
    if conn is None:
        return set()
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT fingerprint FROM hsf_intelligence_alerts "
            "WHERE user_id = %s AND detected_at >= NOW() - make_interval(hours => %s)",
            (user_id, int(hours)),
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
    return {(r["fingerprint"] if isinstance(r, dict) else r[0]) for r in rows}


def record_intelligence_alert(
    *, user_id: str, note: Dict[str, Any], copy: str, fingerprint: str,
    delivery_status: str = "DETECTED",
) -> Optional[int]:
    """Persist one detected intelligence alert. Returns row id or None."""
    if not user_id or not note.get("ticker"):
        return None
    conn = get_neon_conn()
    if conn is None:
        return None
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO hsf_intelligence_alerts
              (user_id, ticker, event_type, severity, fingerprint, payload, copy, delivery_status)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            RETURNING id
            """,
            (user_id, str(note.get("ticker")).upper(), note.get("event_type"),
             note.get("severity"), fingerprint,
             json.dumps(_json_safe(note)), copy, delivery_status),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return int(row[0] if not isinstance(row, dict) else row["id"]) if row else None
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None


def update_delivery_status(alert_id: int, status: str) -> bool:
    conn = get_neon_conn()
    if conn is None or not alert_id:
        return False
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        delivered = ", delivered_at = NOW()" if status == "DELIVERED" else ""
        cur.execute(
            f"UPDATE hsf_intelligence_alerts SET delivery_status = %s{delivered} WHERE id = %s",
            (status, int(alert_id)),
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return False


def list_recent_intelligence_alerts(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    if not user_id:
        return []
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT ticker, event_type, severity, copy, delivery_status, detected_at "
            "FROM hsf_intelligence_alerts WHERE user_id = %s "
            "ORDER BY detected_at DESC LIMIT %s",
            (user_id, int(limit)),
        )
        rows = cur.fetchall() or []
        cols = [d[0] for d in cur.description] if cur.description else []
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return []
    return [dict(r) if isinstance(r, dict) else dict(zip(cols, r)) for r in rows]


def get_hsf_alert_prefs(user_id: str) -> Optional[Dict[str, Any]]:
    """User's intelligence-alert preferences, or None (caller applies defaults)."""
    if not user_id:
        return None
    conn = get_neon_conn()
    if conn is None:
        return None
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT prefs FROM hsf_alert_prefs WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None
    if not row:
        return None
    prefs = row["prefs"] if isinstance(row, dict) else row[0]
    if isinstance(prefs, str):
        try:
            prefs = json.loads(prefs)
        except json.JSONDecodeError:
            prefs = {}
    return prefs if isinstance(prefs, dict) else {}


def set_hsf_alert_prefs(user_id: str, prefs: Dict[str, Any]) -> bool:
    if not user_id:
        return False
    conn = get_neon_conn()
    if conn is None:
        return False
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO hsf_alert_prefs (user_id, prefs, updated_at) VALUES (%s, %s::jsonb, NOW()) "
            "ON CONFLICT (user_id) DO UPDATE SET prefs = EXCLUDED.prefs, updated_at = NOW()",
            (user_id, json.dumps(prefs or {})),
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return False


def list_alert_pref_user_ids() -> List[str]:
    """Users who explicitly opted into HSF intelligence alerts (have a prefs
    row). The conservative V1 subscription set. [] if DB unavailable."""
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM hsf_alert_prefs")
        rows = cur.fetchall() or []
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return []
    return [(r["user_id"] if isinstance(r, dict) else r[0]) for r in rows]


def intelligence_alert_health() -> Dict[str, Any]:
    """Read-only health: totals + last evaluation. Actual values only."""
    conn = get_neon_conn()
    if conn is None:
        return {"available": False}
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), MAX(detected_at), "
                    "COUNT(*) FILTER (WHERE delivery_status='DELIVERED'), "
                    "COUNT(*) FILTER (WHERE delivery_status='FAILED') "
                    "FROM hsf_intelligence_alerts")
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return {"available": False}
    g = (lambda i: row[i]) if not isinstance(row, dict) else (lambda i: list(row.values())[i])
    return {"available": True, "total": g(0), "last_detected_at": g(1),
            "delivered": g(2), "failed": g(3)} if row else {"available": True, "total": 0}


def _json_safe(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out
