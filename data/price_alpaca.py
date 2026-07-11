"""Alpaca price-provider helpers for historical OHLCV downloads."""
from __future__ import annotations

import datetime as _dt
import logging
from typing import Dict, Sequence

import pandas as pd

from .price_utils import chunks, normalize_price_frame

try:
    import requests
    from requests import exceptions as requests_exc
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    requests_exc = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def get_alpaca_config() -> Dict[str, str] | None:
    """Return Alpaca Market Data configuration if it is available.

    Resolution (env-first, then guarded Streamlit secrets) is centralized in
    data.alpaca_config so every reader agrees; this wrapper only adds the
    requests-availability check that bar downloads need.
    """
    from data.alpaca_config import get_alpaca_config as _shared_config

    cfg = _shared_config()
    if cfg is None or requests is None:
        return None
    return cfg


def alpaca_timeframe_from_interval(interval: str) -> str | None:
    """Map yfinance-style intervals to Alpaca timeframes."""
    return {
        "1d": "1Day",
        "1D": "1Day",
    }.get(interval)


def alpaca_limit_from_period(period: str) -> int:
    """Convert a yfinance-style period string into an approximate bar limit."""
    default = 60
    try:
        normalized = period.lower()
        if normalized.endswith("d"):
            return max(1, int(normalized[:-1]))
        if normalized.endswith("mo"):
            return max(1, int(normalized[:-2]) * 21)
        if normalized.endswith("y"):
            return max(1, int(normalized[:-1]) * 252)
    except (AttributeError, TypeError, ValueError):
        pass
    return default


def alpaca_start_from_period(period: str) -> str:
    """Return an ISO date (UTC) far enough back to cover `period` trading days.

    Alpaca's /v2/stocks/bars endpoint requires a `start` date; without it the
    request 400s or returns no bars. We use ~1.7x calendar days per requested
    trading day (plus a buffer) so weekends/holidays don't starve the window.
    """
    days = 60
    try:
        n = (period or "").lower()
        if n.endswith("d"):
            days = max(1, int(n[:-1]))
        elif n.endswith("mo"):
            days = max(1, int(n[:-2])) * 31
        elif n.endswith("y"):
            days = max(1, int(n[:-1])) * 366
    except (AttributeError, TypeError, ValueError):
        pass
    start = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=int(days * 1.7) + 5)
    return start.strftime("%Y-%m-%d")


def download_multi_alpaca(
    tickers: Sequence[str],
    period: str,
    interval: str,
    prepost: bool,
    timeout_s: float,
) -> Dict[str, pd.DataFrame]:
    """Download bars for multiple symbols from Alpaca Market Data."""
    del prepost  # Daily Alpaca bars ignore extended-hours selection.

    cfg = get_alpaca_config()
    if cfg is None:
        raise RuntimeError("Alpaca configuration is not available.")

    timeframe = alpaca_timeframe_from_interval(interval)
    if timeframe is None:
        raise RuntimeError(f"Unsupported interval for Alpaca bars: {interval!r}")

    symbols = [symbol for symbol in dict.fromkeys(tickers) if isinstance(symbol, str) and symbol.strip()]
    if not symbols:
        return {}

    # Alpaca uses a dot for class shares (BRK.B), while our universe uses
    # Yahoo's dash form (BRK-B). Send the dot form and map responses back so a
    # single class-share symbol doesn't 400 the whole batch.
    orig_by_alpaca: Dict[str, str] = {}
    alpaca_symbols: list[str] = []
    for sym in symbols:
        a = str(sym).upper().replace("-", ".")
        orig_by_alpaca[a] = str(sym).upper()
        alpaca_symbols.append(a)
    symbols = alpaca_symbols

    url = f"{cfg['data_url']}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": cfg["api_key"],
        "APCA-API-SECRET-KEY": cfg["api_secret"],
        "Accept": "application/json",
    }

    start = alpaca_start_from_period(period)
    out: Dict[str, pd.DataFrame] = {}
    for chunk in chunks(symbols, 150):
        symbols_param = ",".join(sorted(chunk))
        # Accumulate bars across pages (a 150-symbol × ~60-bar response can
        # exceed Alpaca's 1000-bar page limit, so we must follow next_page_token).
        bars_by_symbol: Dict[str, list] = {}
        page_token: str | None = None
        failed = False
        for _ in range(50):  # hard page cap as a safety bound
            params = {
                "symbols": symbols_param,
                "timeframe": timeframe,
                "start": start,
                "limit": 10000,
                "adjustment": "raw",
                "feed": "iex",
            }
            if page_token:
                params["page_token"] = page_token

            try:
                resp = requests.get(url, headers=headers, params=params, timeout=timeout_s)  # type: ignore[union-attr]
                resp.raise_for_status()
                payload = resp.json()
            except requests_exc.RequestException as exc:  # type: ignore[union-attr]
                logger.warning("Alpaca request failed for %s symbols (%s): %s", len(chunk), symbols_param, exc)
                failed = True
                break
            except ValueError as exc:
                logger.warning("Alpaca returned invalid JSON for %s symbols (%s): %s", len(chunk), symbols_param, exc)
                failed = True
                break

            if not isinstance(payload, dict):
                logger.warning("Alpaca returned non-dict payload for %s symbols", len(chunk))
                failed = True
                break

            page_bars = payload.get("bars") or {}
            for sym, bars in page_bars.items():
                if bars:
                    bars_by_symbol.setdefault(sym, []).extend(bars)

            page_token = payload.get("next_page_token")
            if not page_token:
                break

        if failed:
            continue
        if not bars_by_symbol:
            logger.info("Alpaca returned no bars for %s symbols (%s)", len(chunk), symbols_param)
            continue

        for symbol, bars in (bars_by_symbol or {}).items():
            if not bars:
                logger.debug("Alpaca returned empty bars for symbol %s", symbol)
                continue
            df = pd.DataFrame(bars)
            if df.empty:
                logger.debug("Alpaca bars converted to empty DataFrame for symbol %s", symbol)
                continue

            if "t" in df.columns:
                df["t"] = pd.to_datetime(df["t"], errors="coerce")
                df = df.set_index("t")

            rename = {
                old: new
                for old, new in (("o", "Open"), ("h", "High"), ("l", "Low"), ("c", "Close"), ("v", "Volume"))
                if old in df.columns
            }
            if rename:
                df = df.rename(columns=rename)

            if "Adj Close" not in df.columns and "Close" in df.columns:
                df["Adj Close"] = df["Close"]

            # Map Alpaca's dot form back to the caller's original (dash) symbol.
            out_sym = orig_by_alpaca.get(str(symbol).upper(), str(symbol).upper())

            normalized = normalize_price_frame(df)
            try:
                normalized.attrs["source"] = "alpaca_multi"
                normalized.attrs["symbol"] = out_sym
            except (AttributeError, TypeError, ValueError):
                pass
            out[out_sym] = normalized

    return out
