from __future__ import annotations

from typing import List, Dict, Any

from db.engine import get_db_status, get_neon_conn


def _get_conn():
    """Watchlists are Neon-only. Raise if Neon is not available."""
    status = get_db_status()
    if status != "neon":
        raise RuntimeError("Watchlists require Neon DB (status: %s)" % status)
    return get_neon_conn()


def _ensure_schema(conn) -> None:
    """Create watchlist tables if they don't exist."""
    cur = conn.cursor()
    # Simple, Neon-friendly schema
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlists (
            id          BIGSERIAL PRIMARY KEY,
            user_id     TEXT NOT NULL,
            name        TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist_items (
            id            BIGSERIAL PRIMARY KEY,
            watchlist_id  BIGINT NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
            ticker        TEXT NOT NULL
        );
        """
    )
    conn.commit()
    cur.close()


def list_watchlists(user_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    _ensure_schema(conn)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, created_at FROM watchlists WHERE user_id = %s ORDER BY created_at DESC",
        (user_id,),
    )
    rows = cur.fetchall()
    cur.close()
    return [
        {"id": r[0], "name": r[1], "created_at": r[2]}
        for r in rows
    ]


def create_watchlist(user_id: str, name: str) -> int:
    """Create a new watchlist for the user.

    We don't actually need the new ID on the caller side because the UI
    immediately re-queries all watchlists and refreshes. So we avoid
    depending on RETURNING id, which can be finicky on some drivers.
    """
    conn = _get_conn()
    _ensure_schema(conn)
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
    _ensure_schema(conn)
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
    _ensure_schema(conn)
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
    return [r[0] for r in rows]


def set_watchlist_tickers(watchlist_id: int, user_id: str, tickers: List[str]) -> None:
    """Replace all tickers in a watchlist with the provided list."""
    conn = _get_conn()
    _ensure_schema(conn)
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
            [(watchlist_id, t.upper()) for t in tickers],
        )
    conn.commit()
    cur.close()