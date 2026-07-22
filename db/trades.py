"""Persistence for the user trade journal ("I took this trade").

Plain cursor + commit + close pattern like the rest of db/. Trades are logged
from a scan result / trade plan, optionally closed later with an exit price;
open positions get live P&L in the UI at render time.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from db.engine import get_neon_conn


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_trades (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            entry_price DOUBLE PRECISION NOT NULL,
            shares INTEGER NOT NULL DEFAULT 0,
            source TEXT,
            entered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            exit_price DOUBLE PRECISION,
            closed_at TIMESTAMPTZ
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_trades_user ON user_trades (user_id, entered_at DESC)"
    )
    # Paper-trading + provenance columns (added over time; all nullable so
    # existing manual trades are untouched).
    for col, ddl in (
        ("alpaca_order_id", "TEXT"),
        ("stop_price", "DOUBLE PRECISION"),
        ("target_price", "DOUBLE PRECISION"),
        ("breakout_score", "DOUBLE PRECISION"),
        ("ai_confidence", "DOUBLE PRECISION"),
    ):
        cur.execute(f"ALTER TABLE user_trades ADD COLUMN IF NOT EXISTS {col} {ddl}")
    conn.commit()
    cur.close()


def _get_conn():
    conn = get_neon_conn()
    if conn is None:
        raise RuntimeError("Neon is not available.")
    _ensure_schema(conn)
    return conn


def log_trade(
    user_id: str,
    ticker: str,
    entry_price: float,
    shares: int,
    source: str = "scan",
    *,
    alpaca_order_id: Optional[str] = None,
    stop_price: Optional[float] = None,
    target_price: Optional[float] = None,
    breakout_score: Optional[float] = None,
    ai_confidence: Optional[float] = None,
) -> None:
    """Record a trade. The keyword fields carry paper-order + provenance data
    (Alpaca order id, plan stop/target, and the scanner score / AI confidence at
    entry); all optional so manual logging is unchanged."""
    def _f(v):
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO user_trades
            (user_id, ticker, entry_price, shares, source,
             alpaca_order_id, stop_price, target_price, breakout_score, ai_confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            user_id, (ticker or "").upper(), float(entry_price), int(shares), source,
            (str(alpaca_order_id) if alpaca_order_id else None),
            _f(stop_price), _f(target_price), _f(breakout_score), _f(ai_confidence),
        ),
    )
    conn.commit()
    cur.close()


def _row_to_trade(r: Any) -> Dict[str, Any]:
    if isinstance(r, dict):
        return dict(r)
    return {
        "id": r[0],
        "ticker": r[1],
        "entry_price": r[2],
        "shares": r[3],
        "source": r[4],
        "entered_at": r[5],
        "exit_price": r[6],
        "closed_at": r[7],
    }


def list_trades(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """User's trades, open first then most recent."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, ticker, entry_price, shares, source, entered_at, exit_price, closed_at
        FROM user_trades
        WHERE user_id = %s
        ORDER BY (closed_at IS NOT NULL), entered_at DESC
        LIMIT %s
        """,
        (user_id, int(limit)),
    )
    rows = cur.fetchall()
    cur.close()
    return [_row_to_trade(r) for r in rows]


def close_trade(trade_id: int, user_id: str, exit_price: float) -> None:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE user_trades SET exit_price = %s, closed_at = NOW()
        WHERE id = %s AND user_id = %s AND closed_at IS NULL
        """,
        (float(exit_price), int(trade_id), user_id),
    )
    conn.commit()
    cur.close()


def delete_trade(trade_id: int, user_id: str) -> None:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM user_trades WHERE id = %s AND user_id = %s", (int(trade_id), user_id)
    )
    conn.commit()
    cur.close()


def journal_stats(user_id: str) -> Optional[Dict[str, Any]]:
    """Closed-trade summary: {closed, wins, avg_return_pct}."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE exit_price > entry_price),
               AVG((exit_price - entry_price) / NULLIF(entry_price, 0) * 100.0)
        FROM user_trades
        WHERE user_id = %s AND closed_at IS NOT NULL
        """,
        (user_id,),
    )
    row = cur.fetchone()
    cur.close()
    if not row:
        return None
    vals = list(row.values()) if isinstance(row, dict) else list(row)
    closed = int(vals[0] or 0)
    if closed == 0:
        return None
    return {
        "closed": closed,
        "wins": int(vals[1] or 0),
        "avg_return_pct": float(vals[2]) if vals[2] is not None else None,
    }
