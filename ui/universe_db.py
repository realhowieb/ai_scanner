from __future__ import annotations

import datetime as dt
from typing import Any


UNIVERSE_DB_ERRORS = (
    RuntimeError,
    TypeError,
    ValueError,
    AttributeError,
    OSError,
    ImportError,
)


def try_import(path: str, attr: str | None = None) -> Any:
    """Safely import a module or attribute used by optional universe helpers."""
    try:
        mod = __import__(path, fromlist=[path.split(".")[-1]])
        return getattr(mod, attr) if attr else mod
    except UNIVERSE_DB_ERRORS:
        return None


def get_db_conn() -> Any:
    """Return a live DB connection if available, otherwise None."""
    for mod_path in ("db.engine", "ai_scanner.db.engine", "db.core", "ai_scanner.db.core"):
        mod = try_import(mod_path)
        if not mod:
            continue
        for fn_name in ("get_neon_conn", "get_sqlite_conn", "get_conn", "get_connection"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                try:
                    conn = fn()
                    if conn is not None:
                        return conn
                except UNIVERSE_DB_ERRORS:
                    continue
    return None


def db_get_universe(
    universe: str,
    *,
    max_age_hours: float = 24.0,
) -> tuple[list[str] | None, dict[str, Any] | None]:
    """Fetch universe from DB if refreshed recently."""
    conn = get_db_conn()
    if conn is None:
        return None, None

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT universe, source, refreshed_utc_date, ticker_count, updated_at
                FROM universe_refresh_log
                WHERE universe = %s
                """,
                (universe,),
            )
            row = cur.fetchone()
            if not row:
                return None, None

            _univ, source, refreshed_utc_date, ticker_count, updated_at = row
            now_utc = dt.datetime.now(dt.timezone.utc)
            if isinstance(updated_at, dt.date) and not isinstance(updated_at, dt.datetime):
                updated_at_dt = dt.datetime.combine(updated_at, dt.time(0, 0), tzinfo=dt.timezone.utc)
            else:
                updated_at_dt = updated_at
                if updated_at_dt is not None and updated_at_dt.tzinfo is None:
                    updated_at_dt = updated_at_dt.replace(tzinfo=dt.timezone.utc)

            if updated_at_dt is None:
                return None, None

            age_hours = (now_utc - updated_at_dt).total_seconds() / 3600.0
            if age_hours > max_age_hours:
                return None, None

            cur.execute(
                """
                SELECT ticker
                FROM universe_symbols
                WHERE universe = %s AND is_active = TRUE
                ORDER BY ticker
                """,
                (universe,),
            )
            tickers = [row[0] for row in cur.fetchall() if row and row[0]]

        meta = {
            "universe": universe,
            "source": source,
            "refreshed_utc_date": refreshed_utc_date,
            "ticker_count": ticker_count,
            "updated_at": updated_at_dt,
        }
        if tickers:
            return tickers, meta
        return None, meta
    except UNIVERSE_DB_ERRORS:
        return None, None


def db_upsert_universe(
    universe: str,
    tickers: list[str],
    *,
    source: str,
) -> bool:
    """Upsert tickers and refresh metadata into DB."""
    conn = get_db_conn()
    if conn is None:
        return False

    deduped = _dedupe_tickers(tickers)
    try:
        utc_date = dt.datetime.now(dt.timezone.utc).date()
        with conn.cursor() as cur:
            _ensure_universe_tables(cur)
            cur.execute(
                """
                UPDATE universe_symbols
                SET is_active = FALSE, updated_at = NOW()
                WHERE universe = %s
                """,
                (universe,),
            )

            rows = [(universe, ticker, ticker, source) for ticker in deduped]
            cur.executemany(
                """
                INSERT INTO universe_symbols (universe, ticker, raw_ticker, source, is_active)
                VALUES (%s, %s, %s, %s, TRUE)
                ON CONFLICT (universe, ticker)
                DO UPDATE SET
                    is_active = TRUE,
                    source = EXCLUDED.source,
                    raw_ticker = EXCLUDED.raw_ticker,
                    updated_at = NOW()
                """,
                rows,
            )
            cur.execute(
                """
                INSERT INTO universe_refresh_log (universe, source, refreshed_utc_date, ticker_count, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (universe)
                DO UPDATE SET
                    source = EXCLUDED.source,
                    refreshed_utc_date = EXCLUDED.refreshed_utc_date,
                    ticker_count = EXCLUDED.ticker_count,
                    updated_at = NOW()
                """,
                (universe, source, utc_date, len(deduped)),
            )

        _commit_conn(conn)
        return True
    except UNIVERSE_DB_ERRORS:
        _rollback_conn(conn)
        return False


def _dedupe_tickers(tickers: list[str]) -> list[str]:
    seen = set()
    deduped: list[str] = []
    for raw in tickers:
        if not isinstance(raw, str):
            continue
        ticker = raw.strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            deduped.append(ticker)
    return deduped


def _ensure_universe_tables(cur: Any) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS universe_symbols (
          universe TEXT NOT NULL,
          ticker TEXT NOT NULL,
          raw_ticker TEXT,
          source TEXT NOT NULL DEFAULT 'nasdaqtrader',
          is_active BOOLEAN NOT NULL DEFAULT TRUE,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          PRIMARY KEY (universe, ticker)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS universe_refresh_log (
          universe TEXT PRIMARY KEY,
          source TEXT NOT NULL,
          refreshed_utc_date DATE NOT NULL,
          ticker_count INT NOT NULL,
          updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_universe_symbols_universe_active
        ON universe_symbols(universe, is_active)
        """
    )


def _commit_conn(conn: Any) -> None:
    try:
        conn.commit()
    except UNIVERSE_DB_ERRORS:
        return


def _rollback_conn(conn: Any) -> None:
    try:
        conn.rollback()
    except UNIVERSE_DB_ERRORS:
        return
