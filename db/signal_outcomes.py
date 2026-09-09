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
