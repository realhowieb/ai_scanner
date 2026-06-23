"""Alpaca price-provider helpers for historical OHLCV downloads."""
from __future__ import annotations

from typing import Dict, Sequence
import os
import logging

import pandas as pd

from .price_utils import chunks, normalize_price_frame


try:
    import requests
    from requests import exceptions as requests_exc
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    requests_exc = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _secret_get(secrets: object, key: str) -> str | None:
    """Safely read Streamlit secrets without failing when no secrets file exists."""
    try:
        if hasattr(secrets, "get"):
            value = secrets.get(key)  # type: ignore[attr-defined]
        else:
            value = secrets[key]  # type: ignore[index]
    except (AttributeError, KeyError, TypeError, RuntimeError, OSError):
        return None
    return str(value) if value else None


def get_alpaca_config() -> Dict[str, str] | None:
    """Return Alpaca Market Data configuration if it is available."""
    api_key: str | None = None
    api_secret: str | None = None
    data_url: str | None = None

    try:  # pragma: no cover - optional dependency
        import streamlit as st  # type: ignore

        secrets = getattr(st, "secrets", {})
        api_key = _secret_get(secrets, "ALPACA_API_KEY_ID") or api_key
        api_secret = _secret_get(secrets, "ALPACA_API_SECRET_KEY") or api_secret
        data_url = _secret_get(secrets, "ALPACA_DATA_URL") or data_url
    except (AttributeError, ImportError, RuntimeError):
        pass

    api_key = api_key or os.getenv("ALPACA_API_KEY_ID")
    api_secret = api_secret or os.getenv("ALPACA_API_SECRET_KEY")
    data_url = data_url or os.getenv("ALPACA_DATA_URL") or "https://data.alpaca.markets"

    if not api_key or not api_secret or requests is None:
        return None

    return {
        "api_key": api_key,
        "api_secret": api_secret,
        "data_url": data_url.rstrip("/"),
    }


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

    limit = alpaca_limit_from_period(period)
    symbols = [symbol for symbol in dict.fromkeys(tickers) if isinstance(symbol, str) and symbol.strip()]
    if not symbols:
        return {}

    url = f"{cfg['data_url']}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": cfg["api_key"],
        "APCA-API-SECRET-KEY": cfg["api_secret"],
        "Accept": "application/json",
    }

    out: Dict[str, pd.DataFrame] = {}
    for chunk in chunks(symbols, 150):
        params = {
            "symbols": ",".join(sorted(chunk)),
            "timeframe": timeframe,
            "limit": limit,
            "adjustment": "raw",
            "feed": "iex",
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout_s)  # type: ignore[union-attr]
            resp.raise_for_status()
            payload = resp.json()
        except requests_exc.RequestException as exc:  # type: ignore[union-attr]
            logger.warning(
                "Alpaca request failed for %s symbols (%s): %s",
                len(chunk),
                params["symbols"],
                exc,
            )
            continue
        except ValueError as exc:
            logger.warning(
                "Alpaca returned invalid JSON for %s symbols (%s): %s",
                len(chunk),
                params["symbols"],
                exc,
            )
            continue

        if not isinstance(payload, dict):
            logger.warning(
                "Alpaca returned non-dict payload for %s symbols (%s): %s",
                len(chunk),
                params["symbols"],
                type(payload).__name__,
            )
            continue

        bars_by_symbol = payload.get("bars") if isinstance(payload, dict) else {}
        if not bars_by_symbol and isinstance(payload, dict):
            bars_by_symbol = {key: value for key, value in payload.items() if isinstance(value, list)}

        if not bars_by_symbol:
            logger.warning(
                "Alpaca returned no bars for %s symbols (%s)",
                len(chunk),
                params["symbols"],
            )
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

            normalized = normalize_price_frame(df)
            try:
                normalized.attrs["source"] = "alpaca_multi"
                normalized.attrs["symbol"] = str(symbol).upper()
            except (AttributeError, TypeError, ValueError):
                pass
            out[str(symbol).upper()] = normalized

    return out
