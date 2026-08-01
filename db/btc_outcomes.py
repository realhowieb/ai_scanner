"""Outcome log for 15-min Kalshi BTC contracts (training data for #9).

Each row captures, for one KXBTC15M window: the decision-engine features + its
prediction at log time, the Kalshi implied price, and — once the window settles —
the ground-truth outcome (did BTC finish at/above the strike). Accumulated over
time this is the labeled dataset to replace the heuristic win-probability with a
calibrated model. Plain cursor + commit + close pattern like the rest of db/.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from db.engine import get_neon_conn


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS btc_outcomes (
            window_ticker TEXT PRIMARY KEY,
            logged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            close_time TIMESTAMPTZ,
            strike DOUBLE PRECISION,
            spot_at_log DOUBLE PRECISION,
            pred_direction TEXT,
            pred_confidence INTEGER,
            pred_win_prob INTEGER,
            kalshi_yes_pct DOUBLE PRECISION,
            features JSONB,
            settled BOOLEAN NOT NULL DEFAULT FALSE,
            result_up BOOLEAN,
            settle_value DOUBLE PRECISION,
            settled_at TIMESTAMPTZ
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_btc_outcomes_open "
        "ON btc_outcomes (settled, close_time)"
    )
    # Paper-P&L columns: the would-be trade at log time + its realized P&L on
    # settlement. bet_side/bet_price are set only when the engine said BUY.
    for col, ddl in (
        ("bet_side", "TEXT"),               # 'YES' | 'NO' | NULL (no trade / pass)
        ("bet_price", "DOUBLE PRECISION"),  # price paid, 0-1
        ("bet_pnl", "DOUBLE PRECISION"),    # payout(0/1) - bet_price, on settle
    ):
        cur.execute(f"ALTER TABLE btc_outcomes ADD COLUMN IF NOT EXISTS {col} {ddl}")
    conn.commit()
    cur.close()


def _get_conn():
    conn = get_neon_conn()
    if conn is None:
        raise RuntimeError("Neon is not available.")
    _ensure_schema(conn)
    return conn


def log_window(
    window_ticker: str,
    *,
    close_time,
    strike: Optional[float],
    spot: Optional[float],
    pred_direction: Optional[str],
    pred_confidence: Optional[int],
    pred_win_prob: Optional[int],
    kalshi_yes_pct: Optional[float],
    features: Dict[str, Any],
    bet_side: Optional[str] = None,
    bet_price: Optional[float] = None,
) -> bool:
    """Insert a pending outcome row for a window. Idempotent per window_ticker.

    bet_side/bet_price record the would-be trade when the engine said BUY, so
    P&L can be scored at settlement.
    """
    if not window_ticker:
        return False
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO btc_outcomes
                (window_ticker, close_time, strike, spot_at_log, pred_direction,
                 pred_confidence, pred_win_prob, kalshi_yes_pct, features,
                 bet_side, bet_price)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (window_ticker) DO NOTHING
            """,
            (
                window_ticker, close_time, strike, spot, pred_direction,
                pred_confidence, pred_win_prob, kalshi_yes_pct,
                json.dumps(features or {}), bet_side, bet_price,
            ),
        )
        conn.commit()
        cur.close()
        return True
    except Exception:
        return False


def pending_due(now: Optional[datetime] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Unsettled windows whose close_time has passed (ready to settle)."""
    now = now or datetime.now(timezone.utc)
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT window_ticker, close_time, strike FROM btc_outcomes
            WHERE settled = FALSE AND close_time IS NOT NULL AND close_time <= %s
            ORDER BY close_time ASC LIMIT %s
            """,
            (now, int(limit)),
        )
        rows = cur.fetchall() or []
        cur.close()
    except Exception:
        return []
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append(r)
        else:
            out.append({"window_ticker": r[0], "close_time": r[1], "strike": r[2]})
    return out


def record_result(window_ticker: str, result_up: bool, settle_value: Optional[float]) -> bool:
    """Mark a window settled with its ground-truth outcome."""
    if not window_ticker:
        return False
    try:
        conn = _get_conn()
        cur = conn.cursor()
        # Also score the paper trade: payout is $1 if the bet side won, else $0;
        # P&L = payout − price paid. Computed in SQL from the stored bet_side.
        cur.execute(
            """
            UPDATE btc_outcomes
               SET settled = TRUE, result_up = %s, settle_value = %s, settled_at = NOW(),
                   bet_pnl = CASE
                       WHEN bet_side IS NULL OR bet_price IS NULL THEN NULL
                       WHEN (bet_side = 'YES' AND %s) OR (bet_side = 'NO' AND NOT %s)
                           THEN 1.0 - bet_price
                       ELSE 0.0 - bet_price
                   END
             WHERE window_ticker = %s AND settled = FALSE
            """,
            (bool(result_up), settle_value, bool(result_up), bool(result_up), window_ticker),
        )
        conn.commit()
        cur.close()
        return True
    except Exception:
        return False


def outcome_stats() -> Optional[Dict[str, Any]]:
    """Summary for a readout: counts + how the engine's calls actually did."""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                COUNT(*) AS logged,
                COUNT(*) FILTER (WHERE settled) AS settled,
                COUNT(*) FILTER (WHERE settled AND pred_direction IN ('up','down')) AS decided,
                COUNT(*) FILTER (
                    WHERE settled AND pred_direction IN ('up','down')
                      AND ((pred_direction = 'up') = result_up)
                ) AS correct,
                COUNT(*) FILTER (WHERE settled AND bet_side IS NOT NULL) AS bets,
                COUNT(*) FILTER (WHERE settled AND bet_pnl > 0) AS bet_wins,
                COALESCE(SUM(bet_pnl) FILTER (WHERE settled AND bet_side IS NOT NULL), 0) AS pnl,
                COALESCE(SUM(bet_price) FILTER (WHERE settled AND bet_side IS NOT NULL), 0) AS staked
            FROM btc_outcomes
            """
        )
        row = cur.fetchone()
        cur.close()
    except Exception:
        return None
    if not row:
        return None
    vals = list(row.values()) if isinstance(row, dict) else list(row)
    logged, settled, decided, correct, bets, bet_wins = (int(v or 0) for v in vals[:6])
    pnl = float(vals[6] or 0.0)
    staked = float(vals[7] or 0.0)
    return {
        "logged": logged,
        "settled": settled,
        "decided": decided,
        "correct": correct,
        "accuracy": (correct / decided) if decided else None,
        "bets": bets,
        "bet_wins": bet_wins,
        "pnl": pnl,                              # in $ per $1 unit stakes
        "staked": staked,
        "roi": (pnl / staked) if staked > 0 else None,
    }
