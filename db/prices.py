from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Dict

from sqlalchemy import text

from db.engine import get_neon_conn


def normalize_symbol(symbol: str) -> str:
    """Normalize symbols for caching + Yahoo compatibility.

    - Uppercase
    - Strip whitespace
    - Remove '$' prefix/markers sometimes present in symbol lists
    - Drop preferred/warrant suffixes like 'ADC-A' -> 'ADC'
    """
    s = (symbol or "").strip().upper()
    s = s.replace("$", "")
    if "-" in s:
        s = s.split("-", 1)[0]
    return s


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
                "symbol": normalize_symbol(symbol),
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
    symbols = [normalize_symbol(s) for s in symbols if s]
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
            WHERE symbol = ANY(:symbols::text[])
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
    symbols = [normalize_symbol(s) for s in symbols if s]
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
            WHERE symbol = ANY(:symbols::text[])
              AND as_of >= NOW() - (:mins * INTERVAL '1 minute')
            """
        ),
        {"symbols": symbols, "mins": max_age_minutes},
    )

    fresh = {row.symbol for row in result.fetchall()}
    return [s for s in symbols if s not in fresh]


def get_price_data_snapshot(
    symbols: Iterable[str],
    max_age_minutes: int = 15,
) -> tuple[Dict[str, float | None], set[str]]:
    """Engine-compatible snapshot loader.

    Returns:
      - price_data: {SYMBOL: price}
      - stale: set of SYMBOLs that are missing or older than `max_age_minutes`

    Note: This module currently stores *latest price only* in `price_snapshots`.
    """
    norm_symbols = [normalize_symbol(s) for s in symbols if s]
    if not norm_symbols:
        return {}, set()

    cached = get_price_snapshots(norm_symbols)

    # Stale means: missing from cache OR older than max_age_minutes
    stale_list = get_stale_symbols(norm_symbols, max_age_minutes=max_age_minutes)
    missing = [s for s in norm_symbols if s not in cached]
    stale = set(stale_list) | set(missing)

    return cached, stale


def upsert_price_data_snapshot(
    data_dict: Dict[str, object],
    as_of: datetime | None = None,
) -> None:
    """Engine-compatible snapshot writer.

    Accepts a dict keyed by symbol. Values may be:
      - float/int price
      - dict with 'price' key
      - None

    Examples:
      {'AAPL': 191.23, 'MSFT': {'price': 412.5}}
    """
    if not data_dict:
        return

    if as_of is None:
        as_of = datetime.now(timezone.utc)

    for sym, val in data_dict.items():
        price: float | None
        if val is None:
            price = None
        elif isinstance(val, (int, float)):
            price = float(val)
        elif isinstance(val, dict):
            p = val.get("price")
            if p is None:
                price = None
            elif isinstance(p, (int, float)):
                price = float(p)
            else:
                # Unknown payload; skip quietly
                continue
        else:
            # Unknown payload; skip quietly
            continue

        upsert_price_snapshot(sym, price, as_of=as_of)
