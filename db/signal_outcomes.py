"""Immutable fired-signal rows plus settled market outcomes.

The row is created at signal fire time with the exact ticker, timestamp, price,
scores, probabilities, and indicator snapshot available then. Outcome columns
are filled later once the 1/3/5 trading-day windows have completed.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from db.engine import get_neon_conn


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_outcomes (
            id BIGSERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            source_event_id BIGINT,
            signal_type TEXT,
            user_id TEXT,
            alert_id BIGINT,
            ticker TEXT NOT NULL,
            fired_at TIMESTAMPTZ NOT NULL,
            entry_price DOUBLE PRECISION,
            setup_score DOUBLE PRECISION,
            ai_confidence DOUBLE PRECISION,
            prebreakout_prob DOUBLE PRECISION,
            indicators JSONB DEFAULT '{}'::jsonb,
            raw_signal JSONB DEFAULT '{}'::jsonb,
            return_1d DOUBLE PRECISION,
            return_3d DOUBLE PRECISION,
            return_5d DOUBLE PRECISION,
            mfe_5d DOUBLE PRECISION,
            mae_5d DOUBLE PRECISION,
            outcome_computed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (source, source_event_id, ticker, signal_type)
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_signal_outcomes_pending "
        "ON signal_outcomes (outcome_computed_at, fired_at)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_signal_outcomes_ticker_fired "
        "ON signal_outcomes (ticker, fired_at DESC)"
    )
    conn.commit()
    cur.close()


def freeze_signal(
    *,
    source: str,
    source_event_id: Optional[int],
    signal_type: Optional[str],
    user_id: Optional[str],
    alert_id: Optional[int],
    ticker: str,
    fired_at,
    entry_price: Optional[float],
    setup_score: Optional[float],
    ai_confidence: Optional[float],
    prebreakout_prob: Optional[float],
    indicators: Optional[Dict[str, Any]] = None,
    raw_signal: Optional[Dict[str, Any]] = None,
) -> bool:
    """Insert the frozen fire-time signal row once."""
    conn = get_neon_conn()
    if conn is None:
        return False
    _ensure_schema(conn)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO signal_outcomes (
            source, source_event_id, signal_type, user_id, alert_id, ticker,
            fired_at, entry_price, setup_score, ai_confidence, prebreakout_prob,
            indicators, raw_signal
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
        ON CONFLICT (source, source_event_id, ticker, signal_type) DO NOTHING
        """,
        (
            source,
            int(source_event_id) if source_event_id is not None else None,
            signal_type,
            user_id,
            int(alert_id) if alert_id is not None else None,
            (ticker or "").upper(),
            fired_at,
            entry_price,
            setup_score,
            ai_confidence,
            prebreakout_prob,
            json.dumps(indicators or {}, default=str),
            json.dumps(raw_signal or {}, default=str),
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    return True


def list_pending_outcomes(min_age_days: int = 8, limit: int = 1000) -> List[Dict[str, Any]]:
    """Frozen signals old enough to have a complete 5D window and no outcome."""
    conn = get_neon_conn()
    if conn is None:
        return []
    _ensure_schema(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, ticker, fired_at
        FROM signal_outcomes
        WHERE outcome_computed_at IS NULL
          AND fired_at < NOW() - make_interval(days => %s)
        ORDER BY fired_at ASC
        LIMIT %s
        """,
        (int(min_age_days), int(limit)),
    )
    rows = cur.fetchall() or []
    cur.close()
    conn.close()
    out: List[Dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            out.append(dict(r))
        else:
            out.append({"id": r[0], "ticker": r[1], "fired_at": r[2]})
    return out


def freeze_opportunity(snapshot_time: Any, opp: Dict[str, Any]) -> bool:
    """Freeze one HSF Top-Opportunity into signal_outcomes for calibration.

    Reuses the existing freeze + forward-outcome backfill pipeline (source=
    'opportunity'), so no duplicate outcome system. The frozen payload holds
    ONLY signal-time features (score, version, components, signal flags) — never
    any forward/outcome field; those land later in the return_/mfe_/mae_ columns
    via the same cron backfill. Idempotent per (snapshot, ticker).
    """
    if snapshot_time is None or not opp.get("ticker"):
        return False
    try:
        event_id = int(snapshot_time.timestamp()) if hasattr(snapshot_time, "timestamp") else None
    except Exception:
        event_id = None
    comps = opp.get("score_components") or {}
    # Frozen features (signal-time only — leakage-safe by construction).
    indicators = {
        "n_signals": opp.get("n_signals"),
        "signals": list(opp.get("signals") or []),
        "chg_pct": opp.get("chg_pct"),
        "gap_pct": opp.get("gap_pct"),
        "fading": bool(opp.get("fading")),
        "primary_setup": opp.get("primary_setup"),
        "status": opp.get("status"),
    }
    raw_signal = {
        "hsf_score": opp.get("score"),
        "score_version": opp.get("score_version"),
        "score_components": comps,
        "primary_setup": opp.get("primary_setup"),
        "status": opp.get("status"),
    }
    return freeze_signal(
        source="opportunity",
        source_event_id=event_id,
        signal_type="hsf_opportunity",
        user_id=None,
        alert_id=None,
        ticker=opp.get("ticker"),
        fired_at=snapshot_time,
        entry_price=None,
        setup_score=opp.get("breakout_score"),
        ai_confidence=None,
        prebreakout_prob=opp.get("prob"),
        indicators=indicators,
        raw_signal=raw_signal,
    )


def freeze_opportunities(snapshot_time: Any, opps: List[Dict[str, Any]]) -> int:
    """Freeze a snapshot's opportunities; returns how many rows were written."""
    return sum(1 for o in (opps or []) if freeze_opportunity(snapshot_time, o))


def fetch_opportunity_outcomes(days_back: int = 180, limit: int = 20000) -> List[Dict[str, Any]]:
    """Frozen HSF opportunities joined to their (maybe pending) forward outcomes.

    Returns raw rows for the calibration analytics; the analytics layer decides
    matured vs pending. Never raises — returns [] when the DB is unavailable.
    """
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ticker, fired_at, setup_score, prebreakout_prob,
                   indicators, raw_signal,
                   return_1d, return_3d, return_5d, mfe_5d, mae_5d,
                   outcome_computed_at
            FROM signal_outcomes
            WHERE source = 'opportunity'
              AND fired_at >= NOW() - make_interval(days => %s)
            ORDER BY fired_at ASC
            LIMIT %s
            """,
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
        out.append(dict(r) if isinstance(r, dict) else dict(zip(cols, r)))
    return out


def fetch_ticker_opportunity_history(
    ticker: str, days_back: int = 45, limit: int = 300
) -> List[Dict[str, Any]]:
    """Frozen HSF opportunity observations for one ticker, oldest-first — the
    lifecycle source. Reads signal-time fields only (score/version/status/
    signals) from the frozen payload. Also reports matured/positive counts for
    the ticker-specific history summary. Never raises; [] when DB unavailable.
    """
    t = str(ticker or "").strip().upper()
    if not t:
        return []
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT fired_at, raw_signal, indicators, setup_score, prebreakout_prob,
                   mfe_5d, outcome_computed_at
            FROM signal_outcomes
            WHERE source = 'opportunity' AND UPPER(ticker) = %s
              AND fired_at >= NOW() - make_interval(days => %s)
            ORDER BY fired_at ASC
            LIMIT %s
            """,
            (t, int(days_back), int(limit)),
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
        raw = d.get("raw_signal") or {}
        ind = d.get("indicators") or {}
        out.append({
            "time": d.get("fired_at"),
            "score": raw.get("hsf_score"),
            "status": raw.get("status") or ind.get("status"),
            "score_version": raw.get("score_version"),
            "signals": list(ind.get("signals") or []),
            "matured": d.get("outcome_computed_at") is not None,
            "mfe_5d": d.get("mfe_5d"),
        })
    return out


def fetch_opportunity_observations(days_back: int = 30, limit: int = 20000) -> List[Dict[str, Any]]:
    """Frozen HSF opportunity SIGNAL-TIME observations for Run 25 outcome
    intelligence — id + ticker + snapshot_time + signal-time HSF state ONLY.

    Deliberately returns NO forward/price columns (return_/mfe_/mae_): HSF-state
    outcome intelligence stays separate from the price pipeline. The row id is the
    canonical observation id; fired_at is the source snapshot time. Never raises.
    """
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, ticker, fired_at, raw_signal, indicators
            FROM signal_outcomes
            WHERE source = 'opportunity'
              AND fired_at >= NOW() - make_interval(days => %s)
            ORDER BY fired_at ASC
            LIMIT %s
            """,
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
        raw = d.get("raw_signal") or {}
        ind = d.get("indicators") or {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {}
        if isinstance(ind, str):
            try:
                ind = json.loads(ind)
            except json.JSONDecodeError:
                ind = {}
        out.append({
            "observation_id": d.get("id"),
            "ticker": d.get("ticker"),
            "snapshot_time": d.get("fired_at"),
            "score": raw.get("hsf_score"),
            "status": raw.get("status") or ind.get("status"),
            "score_version": raw.get("score_version"),
            "signals": list(ind.get("signals") or []),
            "n_signals": ind.get("n_signals"),
            "fading": bool(ind.get("fading")),
        })
    return out


def summarize_recent_outcomes(days_back: int = 7) -> Dict[str, Any]:
    """Scorecard over signals fired in the last `days_back` days.

    A "hit" is reaching +4% within the 5-day window (mfe_5d >= 0.04), matching
    the models' economic target. Winners/losers split on close-to-close 5D
    return. Returns zeroed fields (never raises) when the DB or data is absent.
    """
    empty = {
        "days_back": int(days_back), "completed": 0, "pending": 0, "hits": 0,
        "hit_rate": None, "winners": 0, "losers": 0, "avg_winner": None,
        "avg_loser": None, "best_ticker": None, "best_return": None,
    }
    conn = get_neon_conn()
    if conn is None:
        return empty
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
              COUNT(*) FILTER (WHERE outcome_computed_at IS NOT NULL) AS completed,
              COUNT(*) FILTER (WHERE outcome_computed_at IS NULL) AS pending,
              COUNT(*) FILTER (WHERE mfe_5d >= 0.04) AS hits,
              COUNT(*) FILTER (WHERE outcome_computed_at IS NOT NULL AND return_5d > 0) AS winners,
              COUNT(*) FILTER (WHERE outcome_computed_at IS NOT NULL AND return_5d <= 0) AS losers,
              AVG(return_5d) FILTER (WHERE outcome_computed_at IS NOT NULL AND return_5d > 0) AS avg_winner,
              AVG(return_5d) FILTER (WHERE outcome_computed_at IS NOT NULL AND return_5d <= 0) AS avg_loser
            FROM signal_outcomes
            WHERE fired_at >= NOW() - make_interval(days => %s)
            """,
            (int(days_back),),
        )
        row = cur.fetchone()
        cur.execute(
            """
            SELECT ticker, return_5d FROM signal_outcomes
            WHERE fired_at >= NOW() - make_interval(days => %s)
              AND outcome_computed_at IS NOT NULL AND return_5d IS NOT NULL
            ORDER BY return_5d DESC
            LIMIT 1
            """,
            (int(days_back),),
        )
        best = cur.fetchone()
        cur.close()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return empty

    def _g(r, key, idx):
        return r.get(key) if isinstance(r, dict) else r[idx]

    completed = int(_g(row, "completed", 0) or 0) if row else 0
    hits = int(_g(row, "hits", 2) or 0) if row else 0
    out = dict(empty)
    out.update({
        "completed": completed,
        "pending": int(_g(row, "pending", 1) or 0) if row else 0,
        "hits": hits,
        "hit_rate": (hits / completed) if completed else None,
        "winners": int(_g(row, "winners", 3) or 0) if row else 0,
        "losers": int(_g(row, "losers", 4) or 0) if row else 0,
        "avg_winner": float(_g(row, "avg_winner", 5)) if row and _g(row, "avg_winner", 5) is not None else None,
        "avg_loser": float(_g(row, "avg_loser", 6)) if row and _g(row, "avg_loser", 6) is not None else None,
        "best_ticker": (_g(best, "ticker", 0) if best else None),
        "best_return": (float(_g(best, "return_5d", 1)) if best and _g(best, "return_5d", 1) is not None else None),
    })
    return out


def summarize_outcomes_by_type(days_back: int = 7, min_completed: int = 5) -> List[Dict[str, Any]]:
    """Positive-outcome rate per signal_type over the window.

    Only returns types with at least `min_completed` scored signals so the
    dashboard never shows a rate off 1-2 samples. Never raises.
    """
    conn = get_neon_conn()
    if conn is None:
        return []
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COALESCE(signal_type, 'other') AS stype,
                   COUNT(*) FILTER (WHERE outcome_computed_at IS NOT NULL) AS completed,
                   COUNT(*) FILTER (WHERE outcome_computed_at IS NOT NULL AND mfe_5d >= 0.04) AS hits
            FROM signal_outcomes
            WHERE fired_at >= NOW() - make_interval(days => %s)
            GROUP BY COALESCE(signal_type, 'other')
            HAVING COUNT(*) FILTER (WHERE outcome_computed_at IS NOT NULL) >= %s
            ORDER BY hits::float / NULLIF(COUNT(*) FILTER (WHERE outcome_computed_at IS NOT NULL), 0) DESC
            """,
            (int(days_back), int(min_completed)),
        )
        rows = cur.fetchall() or []
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
        if isinstance(r, dict):
            stype, completed, hits = r.get("stype"), r.get("completed"), r.get("hits")
        else:
            stype, completed, hits = r[0], r[1], r[2]
        completed = int(completed or 0)
        hits = int(hits or 0)
        out.append({
            "signal_type": stype or "other",
            "completed": completed,
            "positive_rate": (hits / completed) if completed else None,
        })
    return out


def save_outcome(
    *,
    signal_id: int,
    return_1d: Optional[float],
    return_3d: Optional[float],
    return_5d: Optional[float],
    mfe_5d: Optional[float],
    mae_5d: Optional[float],
) -> bool:
    conn = get_neon_conn()
    if conn is None:
        return False
    _ensure_schema(conn)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE signal_outcomes
        SET return_1d = %s,
            return_3d = %s,
            return_5d = %s,
            mfe_5d = %s,
            mae_5d = %s,
            outcome_computed_at = NOW()
        WHERE id = %s
        """,
        (return_1d, return_3d, return_5d, mfe_5d, mae_5d, int(signal_id)),
    )
    conn.commit()
    cur.close()
    conn.close()
    return True
