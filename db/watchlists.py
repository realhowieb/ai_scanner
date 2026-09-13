from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from db.engine import get_neon_conn
from db.schema import ensure_neon_watchlists_schema

_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,7}$")


def normalize_watchlist_ticker(value: object) -> str:
    """Canonical persisted ticker shape: trimmed, upper-case, and valid-ish."""
    ticker = str(value or "").strip().upper()
    return ticker if _TICKER_RE.match(ticker) else ""


def normalize_watchlist_tickers(tickers: List[object]) -> List[str]:
    """De-dupe watchlist tickers deterministically after normalization."""
    cleaned = sorted({t for t in (normalize_watchlist_ticker(v) for v in (tickers or [])) if t})
    return cleaned


def _get_conn():
    """
    Always return a fresh Neon connection AND ensure schema exists.

    If Neon is not configured or reachable, raise a clear error so the
    watchlists UI can fall back gracefully instead of silently receiving None.
    """
    conn = get_neon_conn()
    if conn is None:
        raise RuntimeError("Neon is not available (missing URL or connection failed).")

    # Ensure schema every time we open a new connection (safe & idempotent)
    ensure_neon_watchlists_schema(conn)

    return conn


def _ensure_schema(conn) -> None:
    """Ensure the Neon watchlists schema exists."""
    ensure_neon_watchlists_schema(conn)


def list_watchlists(user_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, created_at FROM watchlists WHERE user_id = %s ORDER BY created_at DESC",
        (user_id,),
    )
    rows = cur.fetchall()
    cur.close()
    # Normalize rows so we support both dict_row (psycopg) and tuple-style rows.
    normalized: List[Dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            normalized.append(
                {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "created_at": r.get("created_at"),
                }
            )
        else:
            # Fallback for positional rows
            normalized.append(
                {
                    "id": r[0],
                    "name": r[1],
                    "created_at": r[2] if len(r) > 2 else None,
                }
            )
    return normalized


def create_watchlist(user_id: str, name: str) -> int:
    """Create a new watchlist for the user.

    We don't actually need the new ID on the caller side because the UI
    immediately re-queries all watchlists and refreshes. So we avoid
    depending on RETURNING id, which can be finicky on some drivers.
    """
    conn = _get_conn()
    cur = conn.cursor()
    # Simple insert without RETURNING
    cur.execute(
        "INSERT INTO watchlists (user_id, name) VALUES (%s, %s)",
        (user_id, name),
    )
    conn.commit()
    cur.close()
    # We don't use this return value in the UI, so a dummy value is fine.
    return -1


def delete_watchlist(watchlist_id: int, user_id: str) -> None:
    conn = _get_conn()
    cur = conn.cursor()
    # Ensure user owns the watchlist
    cur.execute(
        "DELETE FROM watchlists WHERE id = %s AND user_id = %s",
        (watchlist_id, user_id),
    )
    conn.commit()
    cur.close()


def get_watchlist_tickers(watchlist_id: int, user_id: str) -> List[str]:
    conn = _get_conn()
    cur = conn.cursor()
    # Check ownership
    cur.execute(
        "SELECT 1 FROM watchlists WHERE id = %s AND user_id = %s",
        (watchlist_id, user_id),
    )
    if cur.fetchone() is None:
        cur.close()
        return []

    cur.execute(
        "SELECT ticker FROM watchlist_items WHERE watchlist_id = %s ORDER BY ticker ASC",
        (watchlist_id,),
    )
    rows = cur.fetchall()
    cur.close()

    tickers: List[str] = []
    for r in rows:
        if isinstance(r, dict):
            # psycopg dict_row case
            val = r.get("ticker")
        else:
            # tuple-style row (e.g. sqlite or default cursor)
            val = r[0] if len(r) > 0 else None
        if val:
            ticker = normalize_watchlist_ticker(val)
            if ticker:
                tickers.append(ticker)

    return normalize_watchlist_tickers(tickers)


def get_user_watchlist(user_id: str) -> List[str]:
    """All persisted watchlist tickers for a user, loaded with one batch query."""
    if not user_id:
        return []
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT wi.ticker
        FROM watchlist_items wi
        JOIN watchlists wl ON wl.id = wi.watchlist_id
        WHERE wl.user_id = %s
        ORDER BY wi.ticker ASC
        """,
        (user_id,),
    )
    rows = cur.fetchall() or []
    cur.close()
    tickers: List[object] = []
    for r in rows:
        tickers.append(r.get("ticker") if isinstance(r, dict) else (r[0] if len(r) else None))
    return normalize_watchlist_tickers(tickers)


def _default_watchlist_id(user_id: str) -> Optional[int]:
    watchlists = list_watchlists(user_id)
    if not watchlists:
        create_watchlist(user_id, "My Watchlist")
        watchlists = list_watchlists(user_id)
    if not watchlists:
        return None
    return int(watchlists[0]["id"])


def add_to_watchlist(user_id: str, ticker: object, watchlist_id: Optional[int] = None) -> bool:
    """Add one ticker to the user's active/default watchlist without duplicates."""
    symbol = normalize_watchlist_ticker(ticker)
    if not user_id or not symbol:
        return False
    wid = int(watchlist_id) if watchlist_id is not None else _default_watchlist_id(user_id)
    if wid is None:
        return False
    current = get_watchlist_tickers(wid, user_id)
    updated = normalize_watchlist_tickers([*current, symbol])
    if updated == current:
        return True
    set_watchlist_tickers(wid, user_id, updated)
    return True


def remove_from_watchlist(user_id: str, ticker: object, watchlist_id: Optional[int] = None) -> bool:
    """Remove one ticker from the active watchlist, or every user watchlist."""
    symbol = normalize_watchlist_ticker(ticker)
    if not user_id or not symbol:
        return False
    conn = _get_conn()
    cur = conn.cursor()
    if watchlist_id is None:
        cur.execute(
            """
            DELETE FROM watchlist_items wi
            USING watchlists wl
            WHERE wi.watchlist_id = wl.id
              AND wl.user_id = %s
              AND UPPER(TRIM(wi.ticker)) = %s
            """,
            (user_id, symbol),
        )
    else:
        cur.execute(
            """
            DELETE FROM watchlist_items wi
            USING watchlists wl
            WHERE wi.watchlist_id = wl.id
              AND wl.id = %s
              AND wl.user_id = %s
              AND UPPER(TRIM(wi.ticker)) = %s
            """,
            (int(watchlist_id), user_id, symbol),
        )
    changed = cur.rowcount > 0
    conn.commit()
    cur.close()
    return bool(changed)


def set_watchlist_tickers(watchlist_id: int, user_id: str, tickers: List[str]) -> None:
    """Replace all tickers in a watchlist with the provided list."""
    conn = _get_conn()
    cur = conn.cursor()

    # Ownership check
    cur.execute(
        "SELECT 1 FROM watchlists WHERE id = %s AND user_id = %s",
        (watchlist_id, user_id),
    )
    if cur.fetchone() is None:
        cur.close()
        return

    cur.execute("DELETE FROM watchlist_items WHERE watchlist_id = %s", (watchlist_id,))
    if tickers:
        cur.executemany(
            "INSERT INTO watchlist_items (watchlist_id, ticker) VALUES (%s, %s)",
            [(watchlist_id, t) for t in normalize_watchlist_tickers(tickers)],
        )
    conn.commit()
    cur.close()
