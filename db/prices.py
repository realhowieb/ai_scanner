from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Dict, Optional

from sqlalchemy import text

from db.engine import get_neon_conn


def upsert_price_snapshot(
    symbol: str,
    price: float | None,
    as_of: datetime | None = None,
) -> None:
    """
    Store the latest known price for a symbol.
    Used for admin full-universe scans to avoid repeated yfinance calls.
    """
    if not symbol:
        return

    if as_of is None:
        as_of = datetime.now(timezone.utc)

    conn = get_neon_conn()
    if conn is None:
        return

    with conn.begin():
        conn.execute(
            text(
                """
                INSERT INTO price_snapshots (symbol, price, as_of)
                VALUES (:symbol, :price, :as_of)
                ON CONFLICT (symbol)
                DO UPDATE SET
                    price = EXCLUDED.price,
                    as_of = EXCLUDED.as_of
                """
            ),
            {
                "symbol": symbol.upper(),
                "price": price,
                "as_of": as_of,
            },
        )


def get_price_snapshots(
    symbols: Iterable[str],
) -> Dict[str, float | None]:
    """
    Fetch cached prices for a list of symbols.
    Returns a dict: {SYMBOL: price}
    """
    symbols = [s.upper() for s in symbols if s]
    if not symbols:
        return {}

    conn = get_neon_conn()
    if conn is None:
        return {}

    result = conn.execute(
        text(
            """
            SELECT symbol, price
            FROM price_snapshots
            WHERE symbol = ANY(:symbols)
            """
        ),
        {"symbols": symbols},
    )

    return {row.symbol: row.price for row in result.fetchall()}


def get_stale_symbols(
    symbols: Iterable[str],
    max_age_minutes: int = 15,
) -> list[str]:
    """
    Identify which symbols need fresh pricing.
    """
    symbols = [s.upper() for s in symbols if s]
    if not symbols:
        return []

    conn = get_neon_conn()
    if conn is None:
        return symbols

    result = conn.execute(
        text(
            """
            SELECT symbol
            FROM price_snapshots
            WHERE symbol = ANY(:symbols)
              AND as_of >= NOW() - (:mins * INTERVAL '1 minute')
            """
        ),
        {"symbols": symbols, "mins": max_age_minutes},
    )

    fresh = {row.symbol for row in result.fetchall()}
    return [s for s in symbols if s not in fresh]
