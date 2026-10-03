"""Provider helpers used by scan controls."""
from __future__ import annotations

import re
from typing import Any, List, Optional

import pandas as pd

from data.alpaca_config import get_alpaca_data_feed

try:
    import requests
    from requests import exceptions as requests_exc
except ImportError:  # pragma: no cover - requests is part of core requirements
    requests = None  # type: ignore[assignment]
    requests_exc = None  # type: ignore[assignment]


ALPACA_MAX_SNAPSHOT_BATCH = 50
ALPACA_SNAPSHOT_URL = "https://data.alpaca.markets/v2/stocks/snapshots"

_BAD_SYMBOL_RE = re.compile(r"\s+")
_VALID_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]*$")
_DROP_SUFFIXES = (
    "-U",
    "-W",
    "-WS",
    "-WT",
    "-R",
    "-RT",
    "-CV",
    "-CL",
)


def sanitize_universe_symbols(symbols: List[str]) -> List[str]:
    """Normalize and filter symbols so yfinance does not spam logs."""
    out: List[str] = []
    seen = set()
    for raw_symbol in symbols or []:
        symbol = str(raw_symbol or "").strip().upper()
        if not symbol:
            continue
        if "$" in symbol:
            continue
        if _BAD_SYMBOL_RE.search(symbol):
            continue
        if symbol.endswith(_DROP_SUFFIXES):
            continue
        if not _VALID_SYMBOL_RE.match(symbol):
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def get_alpaca_headers() -> Optional[dict[str, str]]:
    """Return Alpaca auth headers when configured.

    Delegates to data.alpaca_config (env-first, then guarded secrets) — the
    previous secrets-only read silently returned None in headless contexts.
    """
    from data.alpaca_config import get_alpaca_headers as _shared_headers

    return _shared_headers()


def _extract_snapshot_price(snapshot: Any) -> float | None:
    if not isinstance(snapshot, dict):
        return None

    last_trade = snapshot.get("latestTrade") or {}
    price = last_trade.get("p") if isinstance(last_trade, dict) else None

    if price is None:
        minute = snapshot.get("minuteBar") or {}
        price = minute.get("c") if isinstance(minute, dict) else None

    if price is None:
        return None

    try:
        return float(price)
    except (TypeError, ValueError):
        return None


def _normalize_snapshot_payload(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    raw_snapshots = data.get("snapshots")
    if isinstance(raw_snapshots, dict) and raw_snapshots:
        return raw_snapshots

    return data


def get_alpaca_extended_last_prices(symbols: List[str]) -> dict[str, float]:
    """Fetch extended-hours latest prices using Alpaca snapshots."""
    if requests is None:
        return {}

    headers = get_alpaca_headers()
    if not headers:
        return {}

    normalized_symbols = sanitize_universe_symbols(symbols)
    if not normalized_symbols:
        return {}

    out: dict[str, float] = {}
    for index in range(0, len(normalized_symbols), ALPACA_MAX_SNAPSHOT_BATCH):
        batch = normalized_symbols[index : index + ALPACA_MAX_SNAPSHOT_BATCH]
        params = {
            "symbols": ",".join(batch),
            "feed": get_alpaca_data_feed(),
        }
        try:
            resp = requests.get(ALPACA_SNAPSHOT_URL, headers=headers, params=params, timeout=5)
            resp.raise_for_status()
            snapshots = _normalize_snapshot_payload(resp.json())
        except (ValueError, requests_exc.RequestException):  # type: ignore[union-attr]
            continue

        for symbol in batch:
            price = _extract_snapshot_price(snapshots.get(symbol))
            if price is not None:
                out[symbol] = price

    return out


def _et_date(ts: Any):
    """ET calendar date of an Alpaca RFC3339 timestamp, or None."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    try:
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(ZoneInfo("America/New_York")).date()


def premarket_quote_from_snapshot(snapshot: Any, today) -> tuple[float, float] | None:
    """(latest pre-market trade, previous close) for `today` (an ET date), or None.

    Needs a trade printed today: a latest trade from an earlier day is last
    night's after-hours print, not pre-market. The previous close is the newest
    daily bar dated before today (Alpaca's dailyBar can already be today's).
    """
    if not isinstance(snapshot, dict):
        return None
    trade = snapshot.get("latestTrade") or {}
    if not isinstance(trade, dict) or _et_date(trade.get("t")) != today:
        return None
    prev = None
    for key in ("dailyBar", "prevDailyBar"):
        bar = snapshot.get(key) or {}
        d = _et_date(bar.get("t")) if isinstance(bar, dict) else None
        if d is not None and d < today:
            prev = bar.get("c")
            break
    try:
        last, prev = float(trade.get("p")), float(prev)
    except (TypeError, ValueError):
        return None
    return (last, prev) if last > 0 and prev > 0 else None


def get_alpaca_premarket_quotes(symbols: List[str], today) -> dict[str, tuple[float, float]]:
    """{symbol: (pre-market last, previous close)} from Alpaca snapshots."""
    if requests is None:
        return {}
    headers = get_alpaca_headers()
    if not headers:
        return {}
    normalized_symbols = sanitize_universe_symbols(symbols)
    out: dict[str, tuple[float, float]] = {}
    for index in range(0, len(normalized_symbols), ALPACA_MAX_SNAPSHOT_BATCH):
        batch = normalized_symbols[index : index + ALPACA_MAX_SNAPSHOT_BATCH]
        params = {"symbols": ",".join(batch), "feed": get_alpaca_data_feed()}
        try:
            resp = requests.get(ALPACA_SNAPSHOT_URL, headers=headers, params=params, timeout=5)
            resp.raise_for_status()
            snapshots = _normalize_snapshot_payload(resp.json())
        except (ValueError, requests_exc.RequestException):  # type: ignore[union-attr]
            continue
        for symbol in batch:
            quote = premarket_quote_from_snapshot(snapshots.get(symbol), today)
            if quote is not None:
                out[symbol] = quote
    return out


def apply_alpaca_extended_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Override the current price column using Alpaca extended-hours prices."""
    if df is None or df.empty:
        return df

    symbol_col = next((column for column in ("Ticker", "Symbol", "symbol") if column in df.columns), None)
    if symbol_col is None:
        return df

    quotes = get_alpaca_extended_last_prices(df[symbol_col].astype(str).tolist())
    if not quotes:
        return df

    target_col = next((column for column in ("Last", "Price", "Close") if column in df.columns), None)
    if target_col is None:
        target_col = "Last"
        if target_col not in df.columns:
            df[target_col] = None

    def _apply_row(row: pd.Series) -> object:
        symbol = str(row[symbol_col]).strip().upper()
        new_price = quotes.get(symbol)
        return float(new_price) if new_price is not None else row[target_col]

    df[target_col] = df.apply(_apply_row, axis=1)
    return df


def fetch_alpaca_snapshot_debug(symbol: str) -> tuple[int, Any]:
    """Fetch raw Alpaca snapshot details for the provider diagnostics UI."""
    if requests is None:
        raise RuntimeError("requests is not available")

    headers = get_alpaca_headers()
    if not headers:
        raise RuntimeError("Alpaca API keys are not configured")

    params = {
        "symbols": str(symbol).strip().upper(),
        "feed": get_alpaca_data_feed(),
    }
    response = requests.get(ALPACA_SNAPSHOT_URL, headers=headers, params=params, timeout=5)
    response.raise_for_status()
    try:
        return response.status_code, response.json() or {}
    except ValueError:
        return response.status_code, response.text or ""
