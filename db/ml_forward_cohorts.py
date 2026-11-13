"""Prospective holdout quarantine. Collection never selects an outcome column."""
from __future__ import annotations

import datetime as dt
import hashlib
import json

from db.engine import get_neon_conn


def canonical(protocol):
    return json.dumps(protocol, sort_keys=True, separators=(",", ":"))


def validate(protocol):
    from analytics.market_calendar import is_trading_day

    start, end, release = (dt.datetime.fromisoformat(protocol[k]) for k in
                          ("observation_start", "observation_end_exclusive", "evaluation_not_before"))
    if any(d.tzinfo is None for d in (start, end, release)) or not start < end < release:
        raise ValueError("Cohort needs ordered timezone-aware start, end and earliest evaluation times")
    sessions, day = [], start.date()
    while day < end.date():
        if is_trading_day(day):
            sessions.append(day.isoformat())
        day += dt.timedelta(days=1)
    if len(sessions) != protocol["session_count"]:
        raise ValueError("Protocol trading-day count does not match its fixed window")
    if protocol["source"] != "opportunity" or protocol["signal_type"] != "hsf_scanner":
        raise ValueError("Only public Scanner observations are supported")
    return start, end, sessions


def _conn():
    conn = get_neon_conn()
    if conn is None:
        raise RuntimeError("Cohort database unavailable; refusing to continue")
    return conn


def _exists(cur):
    cur.execute("SELECT to_regclass('ml_forward_cohorts') AS cohorts")
    row = cur.fetchone()
    return bool(next(iter(row.values())) if isinstance(row, dict) else row[0])


def _get(cur, protocol):
    cur.execute("SELECT protocol, fingerprint FROM ml_forward_cohorts WHERE cohort_id=%s", (protocol["cohort_id"],))
    row = cur.fetchone()
    if not row:
        raise ValueError("Forward cohort must be registered before collecting")
    saved, digest = list(row.values()) if isinstance(row, dict) else row
    if canonical(saved) != canonical(protocol) or digest != hashlib.sha256(canonical(protocol).encode()).hexdigest():
        raise ValueError("Registered cohort definition is immutable; protocol mismatch")


def register(protocol, *, now=None):
    start, end, _ = validate(protocol)
    now = now or dt.datetime.now(dt.timezone.utc)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('ml-forward-cohorts'))")
            cur.execute("CREATE TABLE IF NOT EXISTS ml_forward_cohorts (cohort_id TEXT PRIMARY KEY, protocol JSONB NOT NULL, "
                        "fingerprint TEXT NOT NULL,start_at TIMESTAMPTZ NOT NULL,end_at TIMESTAMPTZ NOT NULL, "
                        "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())")
            cur.execute("CREATE TABLE IF NOT EXISTS ml_forward_members (cohort_id TEXT REFERENCES ml_forward_cohorts(cohort_id), "
                        "observation_id BIGINT NOT NULL,fired_at TIMESTAMPTZ NOT NULL,ticker TEXT NOT NULL,snapshot JSONB NOT NULL, "
                        "collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),PRIMARY KEY(cohort_id,observation_id))")
            cur.execute("SELECT cohort_id FROM ml_forward_cohorts WHERE cohort_id=%s", (protocol["cohort_id"],))
            if cur.fetchone():
                _get(cur, protocol)
            else:
                if now >= start:
                    raise ValueError("Cannot register a retrospective test cohort")
                cur.execute("SELECT id FROM signal_outcomes WHERE fired_at>=%s AND fired_at<%s LIMIT 1", (start,end))
                if cur.fetchone():
                    raise ValueError("Prospective window already has observations; cannot call it unseen")
                cur.execute("SELECT cohort_id FROM ml_forward_cohorts WHERE start_at<%s AND end_at>%s", (end, start))
                if cur.fetchone():
                    raise ValueError("Prospective cohort window overlaps an existing reservation")
                cur.execute("INSERT INTO ml_forward_cohorts(cohort_id,protocol,fingerprint,start_at,end_at) VALUES(%s,%s::jsonb,%s,%s,%s)",
                            (protocol["cohort_id"], canonical(protocol), hashlib.sha256(canonical(protocol).encode()).hexdigest(), start, end))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def collect(protocol, *, now=None):
    start, end, sessions = validate(protocol)
    now = now or dt.datetime.now(dt.timezone.utc)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _get(cur, protocol)
            # Frozen prediction fields only: no returns, extrema, labels or outcome timestamps.
            cur.execute("INSERT INTO ml_forward_members(cohort_id,observation_id,fired_at,ticker,snapshot) "
                        "SELECT %s,s.id,s.fired_at,s.ticker,jsonb_build_object('setup_score',s.setup_score,"
                        "'ai_confidence',s.ai_confidence,'prebreakout_prob',s.prebreakout_prob,'indicators',jsonb_build_object("
                        "'n_signals',s.indicators->'n_signals','signals',s.indicators->'signals','chg_pct',s.indicators->'chg_pct',"
                        "'gap_pct',s.indicators->'gap_pct','fading',s.indicators->'fading','primary_setup',s.indicators->'primary_setup',"
                        "'status',s.indicators->'status'),"
                        "'raw_signal',jsonb_build_object('models',s.raw_signal->'models','hsf_score',s.raw_signal->'hsf_score',"
                        "'score_components',s.raw_signal->'score_components','score_version',s.raw_signal->'score_version',"
                        "'source_scan_id',s.raw_signal->'source_scan_id','primary_setup',s.raw_signal->'primary_setup',"
                        "'status',s.raw_signal->'status','prebreakout_model_version',s.raw_signal->'prebreakout_model_version',"
                        "'ai_confidence_model_version',s.raw_signal->'ai_confidence_model_version')) "
                        "FROM signal_outcomes s WHERE s.source='opportunity' AND s.signal_type='hsf_scanner' AND s.user_id IS NULL "
                        "AND s.fired_at>=%s AND s.fired_at<%s AND s.fired_at<=%s "
                        "AND (s.fired_at AT TIME ZONE 'UTC')::date=ANY(%s::date[]) "
                        "ON CONFLICT(cohort_id,observation_id) DO NOTHING", (protocol["cohort_id"], start, end, now, sessions))
            added = cur.rowcount
            cur.execute("SELECT COUNT(*) AS observations,COUNT(DISTINCT (ticker,(fired_at AT TIME ZONE 'UTC')::date)) AS signal_days,"
                        "COUNT(DISTINCT (fired_at AT TIME ZONE 'UTC')::date) AS days FROM ml_forward_members WHERE cohort_id=%s",
                        (protocol["cohort_id"],))
            row = cur.fetchone()
            values = list(row.values()) if isinstance(row, dict) else list(row)
        conn.commit()
        return {"cohort_id": protocol["cohort_id"], "added": added, "observations": values[0], "signal_days": values[1],
                "observed_days": values[2], "status": "WAITING_FOR_START" if now < start else "COLLECTING" if now < end else "COLLECTION_CLOSED",
                "outcomes_read": False, "evaluation_not_before": protocol["evaluation_not_before"]}
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def development_ids(start, end):
    """IDs only, excluding every reserved interval even before its collector runs."""
    conn = _conn()
    try:
        with conn.cursor() as cur:
            if not _exists(cur):
                return None
            cur.execute("SELECT s.id FROM signal_outcomes s WHERE s.source='opportunity' AND s.fired_at>=%s AND s.fired_at<%s "
                        "AND NOT EXISTS(SELECT 1 FROM ml_forward_cohorts c WHERE s.fired_at>=c.start_at AND s.fired_at<c.end_at)", (start, end))
            return [int(next(iter(r.values())) if isinstance(r, dict) else r[0]) for r in cur.fetchall()]
    finally:
        conn.close()


def assert_development_allowed(ids):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            if not _exists(cur):
                return
            cur.execute("SELECT s.id FROM signal_outcomes s JOIN ml_forward_cohorts c "
                        "ON s.fired_at>=c.start_at AND s.fired_at<c.end_at WHERE s.id=ANY(%s) LIMIT 1", (list(ids),))
            if cur.fetchone():
                raise ValueError("Dataset includes quarantined forward-test observations; development access refused")
    finally:
        conn.close()
