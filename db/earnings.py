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
from datetime import date, datetime, timezone
import time
from typing import Iterable, Optional, Dict, Tuple


def _norm_symbol(sym: str) -> str:
    """Normalize symbols for consistent DB keys and joins.

    Rules:
    - Uppercase
    - Trim whitespace
    - Strip common exchange suffixes (e.g. .US, .TO)
    - Remove leading '^' (e.g. ^GSPC -> GSPC)
    - Remove common non-alphanumeric separators (e.g. BRK.B -> BRKB, RDS-A -> RDSA)

    NOTE: This is intentionally conservative; it only aims to make joins stable.
    """
    if not sym:
        return ""

    s = str(sym).strip().upper()

    # Yahoo/finance index symbols
    if s.startswith("^"):
        s = s[1:]

    # Exchange suffixes
    if "." in s:
        s = s.split(".", 1)[0]

    # Normalize common separators used across feeds
    # Examples: BRK.B -> BRKB, RDS-A -> RDSA, BF.B -> BFB
    for ch in ("-", "_", " ", "/"):
        s = s.replace(ch, "")

    return s


def _norm_symbols(symbols: Iterable[str]) -> list[str]:
    return [s for s in (_norm_symbol(str(x)) for x in symbols) if s]


def _expand_join_keys(original: str) -> set[str]:
    """Generate a small set of equivalent keys for robust joins."""
    raw = str(original) if original is not None else ""
    raw_strip = raw.strip()
    norm = _norm_symbol(raw_strip)

    keys: set[str] = set()
    if raw_strip:
        keys.add(raw_strip)
        keys.add(raw_strip.upper())
    if norm:
        keys.add(norm)
        keys.add(norm.upper())
    return keys


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
  earnings_date = COALESCE(EXCLUDED.earnings_date, earnings_calendar.earnings_date),
  earnings_time = COALESCE(EXCLUDED.earnings_time, earnings_calendar.earnings_time),
  updated_at = NOW();
"""


def ensure_earnings_table(conn=None) -> None:
    c = _get_conn(conn)
    # Support both context-managed and plain connections
    with c.cursor() as cur:
        cur.execute(CREATE_EARNINGS_TABLE_SQL)
        # Best-effort cleanup: normalize existing symbols (avoid PK collisions)
        try:
            cur.execute(
                """
                WITH norm AS (
                    SELECT symbol AS orig, UPPER(TRIM(symbol)) AS norm
                    FROM earnings_calendar
                ),
                conflicts AS (
                    SELECT norm
                    FROM norm
                    GROUP BY norm
                    HAVING COUNT(*) > 1
                )
                UPDATE earnings_calendar e
                SET symbol = n.norm
                FROM norm n
                WHERE e.symbol = n.orig
                  AND e.symbol <> n.norm
                  AND n.norm NOT IN (SELECT norm FROM conflicts);
                """
            )
        except Exception:
            pass
        try:
            cur.execute(
                """
                UPDATE earnings_calendar
                SET symbol = UPPER(TRIM(symbol))
                WHERE symbol <> UPPER(TRIM(symbol));
                """
            )
        except Exception:
            pass
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

    Strategy:
    1) Prefer `Ticker.get_earnings_dates(limit=12)` when available; choose the next
       earnings datetime >= now (UTC).
    2) Fallback to `Ticker.calendar` (varies by ticker).

    Returns EarningsInfo with earnings_date=None if not available.
    """
    sym = _norm_symbol(symbol)
    if not sym:
        return EarningsInfo(symbol=sym, earnings_date=None, earnings_time=None)

    try:
        import yfinance as yf  # local import to avoid import cost when unused
        import pandas as pd

        t = yf.Ticker(sym)

        # ----------------------------
        # 1) Preferred: get_earnings_dates
        # ----------------------------
        try:
            df = t.get_earnings_dates(limit=12)
            if df is not None and not getattr(df, "empty", True):
                # yfinance often returns a DataFrame indexed by Timestamp
                idx = getattr(df, "index", None)
                if idx is not None and len(idx) > 0:
                    dts = pd.to_datetime(list(idx), errors="coerce", utc=True)
                    dts = [x for x in dts if pd.notna(x)]
                    if dts:
                        now = datetime.now(timezone.utc)
                        future = sorted([x for x in dts if x.to_pydatetime() >= now])
                        if future:
                            dt0 = future[0].to_pydatetime()
                            return EarningsInfo(symbol=sym, earnings_date=dt0.date(), earnings_time=None)

                # Some variants include an explicit column
                for col in ("Earnings Date", "earningsDate", "earnings_date"):
                    if col in getattr(df, "columns", []):
                        val = df[col].iloc[0]
                        dt = pd.to_datetime(val, errors="coerce", utc=True)
                        if pd.notna(dt):
                            return EarningsInfo(symbol=sym, earnings_date=dt.to_pydatetime().date(), earnings_time=None)
        except Exception as exc:
            print(f"WARNING: get_earnings_dates failed for {sym}: {exc}")

        # ----------------------------
        # 2) Fallback: calendar
        # ----------------------------
        try:
            cal = getattr(t, "calendar", None)
            if cal is not None and hasattr(cal, "empty") and not cal.empty:
                # Typical pattern: index contains 'Earnings Date'
                if hasattr(cal, "index") and "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"].values[0]
                    dt = pd.to_datetime(val, errors="coerce", utc=True)
                    if pd.notna(dt):
                        return EarningsInfo(symbol=sym, earnings_date=dt.to_pydatetime().date(), earnings_time=None)

                # Alternate pattern: columns contain 'Earnings Date'
                if hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                    val = cal["Earnings Date"].values[0]
                    dt = pd.to_datetime(val, errors="coerce", utc=True)
                    if pd.notna(dt):
                        return EarningsInfo(symbol=sym, earnings_date=dt.to_pydatetime().date(), earnings_time=None)
        except Exception as exc:
            print(f"WARNING: calendar earnings lookup failed for {sym}: {exc}")

        return EarningsInfo(symbol=sym, earnings_date=None, earnings_time=None)

    except Exception as exc:
        # Best-effort; do not raise
        print(f"WARNING: earnings lookup failed for {sym}: {exc}")
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

    syms = _norm_symbols(symbols)
    total = len(syms)
    out: Dict[str, EarningsInfo] = {}

    with c.cursor() as cur:
        for i, sym in enumerate(syms, start=1):
            info = fetch_next_earnings(sym)
            out[sym] = info
            sym_key = _norm_symbol(info.symbol or sym)
            if not sym_key:
                continue

            # Do not insert placeholder rows with NULL earnings_date.
            # A NULL first insert makes the refresh look successful while the UI
            # still returns 0 rows because fetch_earnings_this_week filters NULLs out.
            if info.earnings_date is not None:
                cur.execute(
                    UPSERT_EARNINGS_SQL,
                    (sym_key, info.earnings_date, info.earnings_time),
                )

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

    found_count = sum(1 for info in out.values() if info.earnings_date is not None)
    if total > 0 and found_count == 0:
        print(
            "WARNING: earnings refresh attempted "
            f"{total} symbols but found 0 earnings dates. "
            "Provider data may be unavailable, blocked, or parser output may have changed."
        )

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

    syms = _norm_symbols(symbols)
    if not syms:
        return {}

    # Using = ANY(%s) requires a Python list for psycopg2
    sql = """
        SELECT symbol AS sym_key, earnings_date
        FROM earnings_calendar
        WHERE symbol = ANY(%s);
    """
    with c.cursor() as cur:
        cur.execute(sql, (syms,))
        rows = cur.fetchall() or []

    # Base map from DB (normalized)
    base = {_norm_symbol(str(s)): d for (s, d) in rows if s is not None}

    # Expand keys so callers can join using either raw or normalized symbols
    out: Dict[str, Optional[date]] = {}
    for orig in symbols:
        for k in _expand_join_keys(str(orig)):
            nk = _norm_symbol(k)
            if nk and nk in base:
                out[k] = base[nk]

    # Always include normalized keys too
    out.update(base)
    return out


def load_earnings_details_map(
    symbols: Iterable[str],
    *,
    conn=None,
) -> Dict[str, Tuple[Optional[date], Optional[str]]]:
    """Return symbol -> (earnings_date, earnings_time)."""
    c = _get_conn(conn)
    ensure_earnings_table(c)

    syms = _norm_symbols(symbols)
    if not syms:
        return {}

    sql = """
        SELECT symbol AS sym_key, earnings_date, earnings_time
        FROM earnings_calendar
        WHERE symbol = ANY(%s);
    """
    with c.cursor() as cur:
        cur.execute(sql, (syms,))
        rows = cur.fetchall() or []

    base = {_norm_symbol(str(s)): (d, t) for (s, d, t) in rows if s is not None}

    out: Dict[str, Tuple[Optional[date], Optional[str]]] = {}
    for orig in symbols:
        for k in _expand_join_keys(str(orig)):
            nk = _norm_symbol(k)
            if nk and nk in base:
                out[k] = base[nk]

    out.update(base)
    return out


# -------------------------
# Refresh log (once per day)
# -------------------------

CREATE_EARNINGS_REFRESH_LOG_SQL = """
CREATE TABLE IF NOT EXISTS earnings_refresh_log (
    refresh_key TEXT PRIMARY KEY,
    refreshed_on DATE NOT NULL,
    refreshed_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

UPSERT_EARNINGS_REFRESH_LOG_SQL = """
INSERT INTO earnings_refresh_log (refresh_key, refreshed_on, refreshed_at)
VALUES (%s, %s, NOW())
ON CONFLICT (refresh_key) DO UPDATE SET
  refreshed_on = EXCLUDED.refreshed_on,
  refreshed_at = NOW();
"""


def ensure_earnings_refresh_log_table(conn=None) -> None:
    c = _get_conn(conn)
    with c.cursor() as cur:
        cur.execute(CREATE_EARNINGS_REFRESH_LOG_SQL)
    try:
        c.commit()
    except Exception:
        pass


def should_refresh_earnings_today(refresh_key: str, *, conn=None) -> bool:
    """Return True if we have NOT refreshed for this refresh_key today (UTC date)."""
    c = _get_conn(conn)
    ensure_earnings_refresh_log_table(c)

    key = (refresh_key or "").strip()
    if not key:
        return True

    sql = "SELECT refreshed_on FROM earnings_refresh_log WHERE refresh_key = %s;"
    with c.cursor() as cur:
        cur.execute(sql, (key,))
        row = cur.fetchone()

    if not row or not row[0]:
        return True

    try:
        # row[0] should be a date
        utc_today = datetime.now(timezone.utc).date()
        return row[0] != utc_today
    except Exception:
        return True



def mark_earnings_refreshed_today(refresh_key: str, *, conn=None) -> None:
    """Mark refresh_key as refreshed for today (UTC date)."""
    c = _get_conn(conn)
    ensure_earnings_refresh_log_table(c)

    key = (refresh_key or "").strip()
    if not key:
        return

    utc_today = datetime.now(timezone.utc).date()
    with c.cursor() as cur:
        cur.execute(UPSERT_EARNINGS_REFRESH_LOG_SQL, (key, utc_today))
    try:
        c.commit()
    except Exception:
        pass


# -------------------------
# UI helpers (read-only)
# -------------------------

def fetch_earnings_this_week(*, conn=None, days_ahead: int = 7):
    """Return upcoming earnings rows for the next N days (inclusive).

    This is DB-only (no Yahoo calls). Intended for UI panels.

    Returns: list[dict] with keys: symbol, earnings_date, earnings_time, updated_at
    """
    c = _get_conn(conn)
    ensure_earnings_table(c)

    utc_today = datetime.now(timezone.utc).date()
    end_day = utc_today.fromordinal(utc_today.toordinal() + max(0, int(days_ahead)))

    sql = """
        SELECT symbol, earnings_date, earnings_time, updated_at
        FROM earnings_calendar
        WHERE earnings_date IS NOT NULL
          AND earnings_date >= %s
          AND earnings_date <= %s
        ORDER BY earnings_date ASC, symbol ASC
        LIMIT 5000;
    """

    with c.cursor() as cur:
        cur.execute(sql, (utc_today, end_day))
        rows = cur.fetchall() or []

    out = []
    for (sym, d, t, upd) in rows:
        out.append(
            {
                "symbol": _norm_symbol(str(sym)) if sym is not None else None,
                "earnings_date": d,
                "earnings_time": t,
                "updated_at": upd,
            }
        )
    return out


def add_earnings_days_column(df, *, date_col: str = "earnings_date", out_col: str = "earnings_in_days"):
    """Add a computed 'days until earnings' column to a DataFrame.

    - df: pandas DataFrame
    - date_col: column containing a date/datetime/str
    - out_col: output column name

    Returns df (same object) for convenience.
    """
    try:
        import pandas as pd

        if df is None or getattr(df, "empty", True):
            return df
        if date_col not in df.columns:
            symbol_col = _find_symbol_column(df)
            if symbol_col is None:
                return df

            symbols = df[symbol_col].astype(str).str.strip().str.upper().tolist()
            details = load_earnings_details_map(symbols)
            if not details:
                return df

            df[date_col] = [_earnings_date_for_symbol(sym, details) for sym in symbols]
            if "earnings_time" not in df.columns:
                df["earnings_time"] = [_earnings_time_for_symbol(sym, details) for sym in symbols]

        today_ts = pd.Timestamp(date.today())
        earn_ts = pd.to_datetime(df[date_col], errors="coerce")
        df[out_col] = (earn_ts - today_ts).dt.days
        return df
    except Exception:
        # Best-effort: never break scans/UI
        return df


def _find_symbol_column(df) -> Optional[str]:
    for col in ("symbol", "Symbol", "Ticker", "ticker"):
        if col in getattr(df, "columns", []):
            return col
    return None


def _earnings_date_for_symbol(symbol: str, details: Dict[str, Tuple[Optional[date], Optional[str]]]) -> Optional[date]:
    detail = _details_for_symbol(symbol, details)
    return detail[0] if detail else None


def _earnings_time_for_symbol(symbol: str, details: Dict[str, Tuple[Optional[date], Optional[str]]]) -> Optional[str]:
    detail = _details_for_symbol(symbol, details)
    return detail[1] if detail else None


def _details_for_symbol(symbol: str, details: Dict[str, Tuple[Optional[date], Optional[str]]]):
    for key in _expand_join_keys(symbol):
        if key in details:
            return details[key]
    norm = _norm_symbol(symbol)
    if norm in details:
        return details[norm]
    return None
