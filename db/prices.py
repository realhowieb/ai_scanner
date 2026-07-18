"""Neon-backed OHLCV price cache shared across processes.

The scan engine fetches daily OHLCV history for the whole universe every run.
This stores each symbol's fetched DataFrame in Neon so separate processes — the
Streamlit app, and each cold-start cron run — can reuse a recent fetch instead
of re-hitting the price provider. In-process caches (data/prices.py) don't span
processes; this does.

psycopg throughout (matching the rest of db/*). Each DataFrame is stored as an
opaque serialized blob keyed by symbol, with an updated_at stamp for staleness.
All operations are best-effort: any failure degrades to "no cache" (the engine
then fetches fresh), never raising into a scan.
"""
from __future__ import annotations

from datetime import datetime
from io import StringIO
from typing import Any, Dict, Iterable, Optional, Tuple

from db.engine import get_neon_conn


def normalize_symbol(symbol: str) -> str:
    """Upper/strip only — class shares (BRK-B vs BRK-A) must stay distinct so a
    cached frame is never served for the wrong security."""
    return (symbol or "").strip().upper()


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS price_data_cache (
            symbol TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    conn.commit()
    cur.close()


def _serialize(df) -> Optional[str]:
    """DataFrame → JSON blob (split orient preserves index + columns)."""
    try:
        import pandas as pd

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        return df.to_json(orient="split", date_format="iso")
    except Exception:
        return None


def _deserialize(payload: Any):
    """JSON blob → DataFrame with a restored DatetimeIndex, or None."""
    try:
        import pandas as pd

        df = pd.read_json(StringIO(str(payload)), orient="split")
        if df is None or df.empty:
            return None
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
        return df
    except Exception:
        return None


def get_price_data_snapshot(
    symbols: Iterable[str],
    max_age_minutes: int = 15,
) -> Tuple[Dict[str, Any], set]:
    """Load cached OHLCV frames fresher than `max_age_minutes`.

    Returns ({original_symbol: DataFrame}, stale_set) where stale_set is every
    requested symbol without a fresh cached frame (missing or too old). Keys are
    the caller's original symbols so the engine's set math lines up.
    """
    orig_by_key: Dict[str, str] = {}
    keys = []
    for s in symbols:
        if not s:
            continue
        k = normalize_symbol(s)
        if k and k not in orig_by_key:
            orig_by_key[k] = s
            keys.append(k)
    if not keys:
        return {}, set()

    conn = get_neon_conn()
    if conn is None:
        return {}, set(orig_by_key.values())

    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT symbol, payload
            FROM price_data_cache
            WHERE symbol = ANY(%s)
              AND updated_at >= NOW() - (%s * INTERVAL '1 minute')
            """,
            (keys, int(max_age_minutes)),
        )
        rows = cur.fetchall()
        cur.close()
    except Exception:
        return {}, set(orig_by_key.values())

    cached: Dict[str, Any] = {}
    for r in rows:
        key = r["symbol"] if isinstance(r, dict) else r[0]
        payload = r["payload"] if isinstance(r, dict) else r[1]
        df = _deserialize(payload)
        if df is not None and not df.empty:
            cached[orig_by_key.get(key, key)] = df
    stale = set(orig_by_key.values()) - set(cached.keys())
    return cached, stale


def upsert_price_data_snapshot(
    data_dict: Dict[str, Any],
    as_of: Optional[datetime] = None,  # accepted for compat; write time is NOW()
) -> None:
    """Persist {symbol: DataFrame} OHLCV frames (best-effort, batched upsert)."""
    del as_of
    if not data_dict:
        return
    conn = get_neon_conn()
    if conn is None:
        return
    try:
        _ensure_schema(conn)
        params = []
        for sym, df in data_dict.items():
            key = normalize_symbol(sym)
            payload = _serialize(df)
            if key and payload is not None:
                params.append((key, payload))
        if params:
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT INTO price_data_cache (symbol, payload, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (symbol) DO UPDATE
                    SET payload = EXCLUDED.payload, updated_at = NOW()
                """,
                params,
            )
            conn.commit()
            cur.close()
    except Exception:
        pass
