from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

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


def normalize_watchlist_name(name: object) -> str:
    """Canonical display-name validation for persisted watchlists."""
    cleaned = re.sub(r"\s+", " ", str(name or "").strip())
    return cleaned[:80]


def _row_get(row: object, key: str, index: int, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[index]  # type: ignore[index]
    except Exception:
        return default


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
    _repair_default_watchlist(user_id, conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT wl.id, wl.name, wl.created_at, wl.is_default, COUNT(wi.id) AS symbol_count
        FROM watchlists wl
        LEFT JOIN watchlist_items wi ON wi.watchlist_id = wl.id
        WHERE wl.user_id = %s
        GROUP BY wl.id, wl.name, wl.created_at, wl.is_default
        ORDER BY wl.is_default DESC, wl.created_at DESC, wl.id DESC
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    cur.close()
    # Normalize rows so we support both dict_row (psycopg) and tuple-style rows.
    normalized: List[Dict[str, Any]] = []
    for r in rows:
        normalized.append(
            {
                "id": _row_get(r, "id", 0),
                "name": _row_get(r, "name", 1),
                "created_at": _row_get(r, "created_at", 2),
                "is_default": bool(_row_get(r, "is_default", 3, False)),
                "symbol_count": int(_row_get(r, "symbol_count", 4, 0) or 0),
            }
        )
    return normalized


def _user_watchlist_count(user_id: str, conn) -> int:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM watchlists WHERE user_id = %s", (user_id,))
    row = cur.fetchone()
    cur.close()
    return int(_row_get(row, "count", 0, 0) or 0)


def _watchlist_owner_exists(watchlist_id: int, user_id: str, conn) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM watchlists WHERE id = %s AND user_id = %s",
        (int(watchlist_id), user_id),
    )
    found = cur.fetchone() is not None
    cur.close()
    return found


def _watchlist_name_exists(user_id: str, name: str, conn, *, exclude_id: Optional[int] = None) -> bool:
    cur = conn.cursor()
    if exclude_id is None:
        cur.execute(
            "SELECT 1 FROM watchlists WHERE user_id = %s AND LOWER(TRIM(name)) = LOWER(TRIM(%s))",
            (user_id, name),
        )
    else:
        cur.execute(
            """
            SELECT 1 FROM watchlists
            WHERE user_id = %s AND LOWER(TRIM(name)) = LOWER(TRIM(%s)) AND id <> %s
            """,
            (user_id, name, int(exclude_id)),
        )
    exists = cur.fetchone() is not None
    cur.close()
    return bool(exists)


def _repair_default_watchlist(user_id: str, conn=None) -> Optional[int]:
    """Guarantee at most one deterministic persisted default for a user."""
    if not user_id:
        return None
    own_conn = conn is None
    conn = conn or _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, is_default
        FROM watchlists
        WHERE user_id = %s
        ORDER BY is_default DESC, created_at DESC, id DESC
        """,
        (user_id,),
    )
    rows = cur.fetchall() or []
    if not rows:
        cur.close()
        if own_conn:
            conn.commit()
        return None
    default_id = None
    for row in rows:
        wid = int(_row_get(row, "id", 0))
        if bool(_row_get(row, "is_default", 1, False)) and default_id is None:
            default_id = wid
            break
    if default_id is None:
        default_id = int(_row_get(rows[0], "id", 0))
    cur.execute("UPDATE watchlists SET is_default = (id = %s) WHERE user_id = %s", (default_id, user_id))
    conn.commit()
    cur.close()
    return default_id


def get_default_watchlist_id(user_id: str) -> Optional[int]:
    conn = _get_conn()
    return _repair_default_watchlist(user_id, conn)


def set_default_watchlist(watchlist_id: int, user_id: str) -> bool:
    conn = _get_conn()
    if not _watchlist_owner_exists(watchlist_id, user_id, conn):
        return False
    cur = conn.cursor()
    cur.execute("UPDATE watchlists SET is_default = (id = %s) WHERE user_id = %s", (int(watchlist_id), user_id))
    conn.commit()
    cur.close()
    return True


def create_watchlist(user_id: str, name: str, *, make_default: bool = False) -> int:
    """Create a new watchlist for the user.

    The first watchlist is automatically default. Name validation happens here
    so every UI path shares the same persistence rules.
    """
    cleaned = normalize_watchlist_name(name)
    if not user_id or not cleaned:
        raise ValueError("Watchlist name is required.")
    conn = _get_conn()
    if _watchlist_name_exists(user_id, cleaned, conn):
        raise ValueError("A watchlist with that name already exists.")
    first_list = _user_watchlist_count(user_id, conn) == 0
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO watchlists (user_id, name, is_default) VALUES (%s, %s, %s) RETURNING id",
        (user_id, cleaned, bool(make_default or first_list)),
    )
    row = cur.fetchone()
    watchlist_id = int(_row_get(row, "id", 0, -1))
    if make_default or first_list:
        cur.execute("UPDATE watchlists SET is_default = (id = %s) WHERE user_id = %s", (watchlist_id, user_id))
    conn.commit()
    cur.close()
    return watchlist_id


def rename_watchlist(watchlist_id: int, user_id: str, new_name: str) -> bool:
    cleaned = normalize_watchlist_name(new_name)
    if not cleaned:
        raise ValueError("Watchlist name is required.")
    conn = _get_conn()
    if not _watchlist_owner_exists(watchlist_id, user_id, conn):
        return False
    if _watchlist_name_exists(user_id, cleaned, conn, exclude_id=int(watchlist_id)):
        raise ValueError("A watchlist with that name already exists.")
    cur = conn.cursor()
    cur.execute(
        "UPDATE watchlists SET name = %s WHERE id = %s AND user_id = %s",
        (cleaned, int(watchlist_id), user_id),
    )
    changed = cur.rowcount > 0
    conn.commit()
    cur.close()
    return bool(changed)


def _copy_items(source_id: int, dest_id: int, conn) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO watchlist_items (watchlist_id, ticker, date_added, price_when_added, note)
        SELECT %s, src.ticker, src.date_added, src.price_when_added, src.note
        FROM watchlist_items src
        WHERE src.watchlist_id = %s
          AND NOT EXISTS (
              SELECT 1 FROM watchlist_items dst
              WHERE dst.watchlist_id = %s AND UPPER(TRIM(dst.ticker)) = UPPER(TRIM(src.ticker))
          )
        """,
        (int(dest_id), int(source_id), int(dest_id)),
    )
    changed = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
    cur.close()
    return int(changed)


def duplicate_watchlist(watchlist_id: int, user_id: str, new_name: Optional[str] = None) -> Optional[int]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT name FROM watchlists WHERE id = %s AND user_id = %s", (int(watchlist_id), user_id))
    row = cur.fetchone()
    cur.close()
    if row is None:
        return None
    base = normalize_watchlist_name(new_name or f"{_row_get(row, 'name', 0, 'Watchlist')} Copy")
    name = base
    suffix = 2
    while _watchlist_name_exists(user_id, name, conn):
        name = normalize_watchlist_name(f"{base} {suffix}")
        suffix += 1
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO watchlists (user_id, name, is_default) VALUES (%s, %s, FALSE) RETURNING id",
        (user_id, name),
    )
    new_id = int(_row_get(cur.fetchone(), "id", 0, -1))
    cur.close()
    _copy_items(int(watchlist_id), new_id, conn)
    _repair_default_watchlist(user_id, conn)
    conn.commit()
    return new_id


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
    _repair_default_watchlist(user_id, conn)


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


def get_watchlist_items(watchlist_id: int, user_id: str) -> List[Dict[str, Any]]:
    conn = _get_conn()
    if not _watchlist_owner_exists(watchlist_id, user_id, conn):
        return []
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ticker, date_added, price_when_added, note
        FROM watchlist_items
        WHERE watchlist_id = %s
        ORDER BY ticker ASC
        """,
        (int(watchlist_id),),
    )
    rows = cur.fetchall() or []
    cur.close()
    items: List[Dict[str, Any]] = []
    for row in rows:
        ticker = normalize_watchlist_ticker(_row_get(row, "ticker", 0))
        if ticker:
            items.append(
                {
                    "ticker": ticker,
                    "date_added": _row_get(row, "date_added", 1),
                    "price_when_added": _row_get(row, "price_when_added", 2),
                    "note": _row_get(row, "note", 3),
                }
            )
    return items


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
    wid = get_default_watchlist_id(user_id)
    if wid is None:
        create_watchlist(user_id, "My Watchlist", make_default=True)
        wid = get_default_watchlist_id(user_id)
    return wid


def add_tickers_to_watchlist(
    user_id: str,
    tickers: Sequence[object],
    watchlist_id: Optional[int] = None,
    *,
    price_when_added: Optional[float] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Batch add tickers to one watchlist, preserving existing memberships."""
    symbols = normalize_watchlist_tickers(list(tickers or []))
    if not user_id or not symbols:
        return {"added": [], "already_present": [], "invalid": []}
    wid = int(watchlist_id) if watchlist_id is not None else _default_watchlist_id(user_id)
    if wid is None:
        return {"added": [], "already_present": symbols, "invalid": []}
    conn = _get_conn()
    if not _watchlist_owner_exists(wid, user_id, conn):
        return {"added": [], "already_present": symbols, "invalid": []}
    current = set(get_watchlist_tickers(wid, user_id))
    added = [s for s in symbols if s not in current]
    already = [s for s in symbols if s in current]
    cur = conn.cursor()
    for symbol in added:
        cur.execute(
            """
            INSERT INTO watchlist_items (watchlist_id, ticker, price_when_added, note)
            VALUES (%s, %s, %s, %s)
            """,
            (wid, symbol, price_when_added, note),
        )
    conn.commit()
    cur.close()
    return {"added": added, "already_present": already, "invalid": []}


def add_to_watchlist(user_id: str, ticker: object, watchlist_id: Optional[int] = None) -> bool:
    """Add one ticker to the user's active/default watchlist without duplicates."""
    if not normalize_watchlist_ticker(ticker):
        return False
    result = add_tickers_to_watchlist(user_id, [ticker], watchlist_id)
    return bool(result["added"] or result["already_present"])


def remove_tickers_from_watchlist(
    user_id: str, tickers: Sequence[object], watchlist_id: Optional[int] = None
) -> Dict[str, Any]:
    symbols = normalize_watchlist_tickers(list(tickers or []))
    if not user_id or not symbols:
        return {"removed": [], "missing": []}
    placeholders = ", ".join(["%s"] * len(symbols))
    params: List[Any] = []
    conn = _get_conn()
    cur = conn.cursor()
    if watchlist_id is None:
        params = [user_id, *symbols]
        cur.execute(
            f"""
            DELETE FROM watchlist_items wi
            USING watchlists wl
            WHERE wi.watchlist_id = wl.id
              AND wl.user_id = %s
              AND UPPER(TRIM(wi.ticker)) IN ({placeholders})
            RETURNING wi.ticker
            """,
            tuple(params),
        )
    else:
        params = [int(watchlist_id), user_id, *symbols]
        cur.execute(
            f"""
            DELETE FROM watchlist_items wi
            USING watchlists wl
            WHERE wi.watchlist_id = wl.id
              AND wl.id = %s
              AND wl.user_id = %s
              AND UPPER(TRIM(wi.ticker)) IN ({placeholders})
            RETURNING wi.ticker
            """,
            tuple(params),
        )
    rows = cur.fetchall() or []
    removed = normalize_watchlist_tickers([_row_get(r, "ticker", 0) for r in rows])
    conn.commit()
    cur.close()
    return {"removed": removed, "missing": [s for s in symbols if s not in set(removed)]}


def remove_from_watchlist(user_id: str, ticker: object, watchlist_id: Optional[int] = None) -> bool:
    """Remove one ticker from one watchlist, or every user watchlist when unspecified."""
    symbol = normalize_watchlist_ticker(ticker)
    if not user_id or not symbol:
        return False
    return bool(remove_tickers_from_watchlist(user_id, [symbol], watchlist_id)["removed"])


def copy_tickers_between_watchlists(
    user_id: str, source_watchlist_id: int, dest_watchlist_id: int, tickers: Sequence[object]
) -> Dict[str, Any]:
    symbols = normalize_watchlist_tickers(list(tickers or []))
    if not symbols or int(source_watchlist_id) == int(dest_watchlist_id):
        return {"copied": [], "already_present": symbols, "missing": []}
    conn = _get_conn()
    if not _watchlist_owner_exists(source_watchlist_id, user_id, conn) or not _watchlist_owner_exists(
        dest_watchlist_id, user_id, conn
    ):
        return {"copied": [], "already_present": [], "missing": symbols}
    source = {item["ticker"]: item for item in get_watchlist_items(source_watchlist_id, user_id)}
    dest = set(get_watchlist_tickers(dest_watchlist_id, user_id))
    copied: List[str] = []
    already: List[str] = []
    missing: List[str] = []
    cur = conn.cursor()
    for symbol in symbols:
        item = source.get(symbol)
        if item is None:
            missing.append(symbol)
        elif symbol in dest:
            already.append(symbol)
        else:
            cur.execute(
                """
                INSERT INTO watchlist_items (watchlist_id, ticker, date_added, price_when_added, note)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    int(dest_watchlist_id),
                    symbol,
                    item.get("date_added"),
                    item.get("price_when_added"),
                    item.get("note"),
                ),
            )
            copied.append(symbol)
    conn.commit()
    cur.close()
    return {"copied": copied, "already_present": already, "missing": missing}


def move_tickers_between_watchlists(
    user_id: str, source_watchlist_id: int, dest_watchlist_id: int, tickers: Sequence[object]
) -> Dict[str, Any]:
    copied = copy_tickers_between_watchlists(user_id, source_watchlist_id, dest_watchlist_id, tickers)
    removable = [*copied.get("copied", []), *copied.get("already_present", [])]
    removed = remove_tickers_from_watchlist(user_id, removable, source_watchlist_id) if removable else {"removed": []}
    return {**copied, "moved": removed.get("removed", [])}


def update_watchlist_item_note(watchlist_id: int, user_id: str, ticker: object, note: object) -> bool:
    symbol = normalize_watchlist_ticker(ticker)
    if not symbol:
        return False
    cleaned = str(note or "").strip()[:500] or None
    conn = _get_conn()
    if not _watchlist_owner_exists(watchlist_id, user_id, conn):
        return False
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE watchlist_items
        SET note = %s
        WHERE watchlist_id = %s AND UPPER(TRIM(ticker)) = %s
        """,
        (cleaned, int(watchlist_id), symbol),
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

    desired = normalize_watchlist_tickers(tickers)
    cur.execute("SELECT ticker FROM watchlist_items WHERE watchlist_id = %s", (watchlist_id,))
    existing = {normalize_watchlist_ticker(_row_get(r, "ticker", 0)) for r in (cur.fetchall() or [])}
    desired_set = set(desired)
    remove = sorted(existing - desired_set)
    add = [t for t in desired if t not in existing]
    if remove:
        placeholders = ", ".join(["%s"] * len(remove))
        cur.execute(
            f"DELETE FROM watchlist_items WHERE watchlist_id = %s AND UPPER(TRIM(ticker)) IN ({placeholders})",
            (watchlist_id, *remove),
        )
    if add:
        cur.executemany(
            "INSERT INTO watchlist_items (watchlist_id, ticker) VALUES (%s, %s)",
            [(watchlist_id, t) for t in add],
        )
    conn.commit()
    cur.close()
