

"""Earnings calendar support.

Goal: populate a slow-moving earnings calendar once per day (or on-demand), then JOIN it
into scan results / filters without doing per-ticker API calls during scans.

Table:
  earnings_calendar(symbol PRIMARY KEY, earnings_date DATE, earnings_time TEXT, updated_at TIMESTAMP)

Notes:
- This module is designed to be called from your daily snapshot job or an admin-only button.
- Uses yfinance best-effort (dates can be missing/estimated). Safe failures return None.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import time
from typing import Iterable, Optional, Dict, Tuple


# -------------------------
# DB connection helpers
# -------------------------

def _get_conn(provided_conn=None):
    """Return a DB connection.

    Prefer an injected connection, otherwise try common project helpers.
    """
    if provided_conn is not None:
        return provided_conn

    # Try typical project patterns
    try:
        from db.core import get_conn  # type: ignore

        return get_conn()
    except Exception:
        pass

    try:
        from db.connection import get_conn  # type: ignore

        return get_conn()
    except Exception:
        pass

    raise RuntimeError(
        "No DB connection available. Pass conn=... or add db.core.get_conn()."
    )


# -------------------------
# Schema
# -------------------------

CREATE_EARNINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS earnings_calendar (
    symbol TEXT PRIMARY KEY,
    earnings_date DATE,
    earnings_time TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);
"""

UPSERT_EARNINGS_SQL = """
INSERT INTO earnings_calendar (symbol, earnings_date, earnings_time, updated_at)
VALUES (%s, %s, %s, NOW())
ON CONFLICT (symbol) DO UPDATE SET
  earnings_date = EXCLUDED.earnings_date,
  earnings_time = EXCLUDED.earnings_time,
  updated_at = NOW();
"""


def ensure_earnings_table(conn=None) -> None:
    c = _get_conn(conn)
    # Support both context-managed and plain connections
    with c.cursor() as cur:
        cur.execute(CREATE_EARNINGS_TABLE_SQL)
    try:
        c.commit()
    except Exception:
        # Some connection managers autocommit
        pass


# -------------------------
# Fetching (yfinance)
# -------------------------

@dataclass
class EarningsInfo:
    symbol: str
    earnings_date: Optional[date]
    earnings_time: Optional[str] = None  # 'AMC', 'BMO', or None


def fetch_next_earnings(symbol: str) -> EarningsInfo:
    """Fetch next earnings date (best-effort) for a symbol using yfinance.

    Returns EarningsInfo with earnings_date=None if not available.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return EarningsInfo(symbol=sym, earnings_date=None, earnings_time=None)

    try:
        import yfinance as yf  # local import to avoid import cost when unused

        t = yf.Ticker(sym)
        df = t.get_earnings_dates(limit=1)
        if df is None or getattr(df, "empty", True):
            return EarningsInfo(symbol=sym, earnings_date=None, earnings_time=None)

        # yfinance returns a DataFrame indexed by Timestamp
        ts = df.index[0]
        ed = ts.date() if hasattr(ts, "date") else None
        return EarningsInfo(symbol=sym, earnings_date=ed, earnings_time=None)
    except Exception:
        # Best-effort; do not raise
        return EarningsInfo(symbol=sym, earnings_date=None, earnings_time=None)


# -------------------------
# Populate (daily)
# -------------------------

def populate_earnings_calendar(
    symbols: Iterable[str],
    *,
    conn=None,
    sleep_s: float = 0.02,
    commit_every: int = 200,
    progress_cb=None,
) -> Dict[str, EarningsInfo]:
    """Populate / refresh earnings_calendar for given symbols.

    - symbols: universe list
    - sleep_s: gentle rate limit between yfinance calls
    - commit_every: commit every N rows
    - progress_cb: optional callback(progress:int, total:int, symbol:str)

    Returns a dict symbol -> EarningsInfo for what we attempted.
    """
    c = _get_conn(conn)
    ensure_earnings_table(c)

    syms = [str(s).strip().upper() for s in symbols if str(s).strip()]
    total = len(syms)
    out: Dict[str, EarningsInfo] = {}

    with c.cursor() as cur:
        for i, sym in enumerate(syms, start=1):
            info = fetch_next_earnings(sym)
            out[sym] = info
            cur.execute(UPSERT_EARNINGS_SQL, (info.symbol, info.earnings_date, info.earnings_time))

            if progress_cb:
                try:
                    progress_cb(i, total, sym)
                except Exception:
                    pass

            if commit_every and i % commit_every == 0:
                try:
                    c.commit()
                except Exception:
                    pass

            if sleep_s:
                time.sleep(sleep_s)

    try:
        c.commit()
    except Exception:
        pass

    return out


# -------------------------
# Load for joins / filters
# -------------------------

def load_earnings_map(
    symbols: Iterable[str],
    *,
    conn=None,
) -> Dict[str, Optional[date]]:
    """Return symbol -> earnings_date for the provided symbols."""
    c = _get_conn(conn)
    ensure_earnings_table(c)

    syms = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if not syms:
        return {}

    # Using = ANY(%s) requires a Python list for psycopg2
    sql = "SELECT symbol, earnings_date FROM earnings_calendar WHERE symbol = ANY(%s);"
    with c.cursor() as cur:
        cur.execute(sql, (syms,))
        rows = cur.fetchall() or []

    return {str(s).upper(): d for (s, d) in rows}


def load_earnings_details_map(
    symbols: Iterable[str],
    *,
    conn=None,
) -> Dict[str, Tuple[Optional[date], Optional[str]]]:
    """Return symbol -> (earnings_date, earnings_time)."""
    c = _get_conn(conn)
    ensure_earnings_table(c)

    syms = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if not syms:
        return {}

    sql = "SELECT symbol, earnings_date, earnings_time FROM earnings_calendar WHERE symbol = ANY(%s);"
    with c.cursor() as cur:
        cur.execute(sql, (syms,))
        rows = cur.fetchall() or []

    return {str(s).upper(): (d, t) for (s, d, t) in rows}