"""
Centralized market data helpers for the AI Scanner.

This module is responsible for talking to Alpaca's Market Data API and returning
a simple, app-friendly structure for latest quotes that can be reused across:
  - price ticker strip
  - market snapshot (SPY / QQQ, etc.)
  - liquidity filters for large universes
  - extended-hours scans (premarket / after-hours)

It is intentionally written to be:
  - resilient (graceful failure and empty results on config/network issues)
  - cache-friendly (Streamlit @st.cache_data wrappers)
  - independent of the scan engine or UI details
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import streamlit as st

try:
    import requests  # type: ignore
    from requests import exceptions as requests_exc
except ImportError:  # pragma: no cover - requests import failure handled at runtime
    requests = None  # type: ignore
    requests_exc = None  # type: ignore


# ------------------------- Internal config helpers -------------------------


def _get_alpaca_base_urls() -> Dict[str, str]:
    """
    Read Alpaca API URLs from Streamlit secrets.

    Expected keys in .streamlit/secrets.toml:
      ALPACA_API_KEY_ID
      ALPACA_API_SECRET_KEY
      ALPACA_BASE_URL (optional, defaults to https://paper-api.alpaca.markets)
      ALPACA_DATA_URL (optional, defaults to https://data.alpaca.markets)
    """
    # Read env vars first (works headlessly, e.g. the scheduled cron), then fall
    # back to Streamlit secrets — guarded, since accessing st.secrets raises
    # StreamlitSecretNotFoundError when there's no secrets file (non-app context).
    def _secret(key: str, default: Optional[str] = None) -> Optional[str]:
        env_val = os.getenv(key)
        if env_val:
            return env_val
        try:
            val = getattr(st, "secrets", {}).get(key)
        except Exception:
            val = None
        return val if val else default

    api_key = _secret("ALPACA_API_KEY_ID")
    api_secret = _secret("ALPACA_API_SECRET_KEY")
    base_url = _secret("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    data_url = _secret("ALPACA_DATA_URL", "https://data.alpaca.markets")

    return {
        "api_key": api_key,
        "api_secret": api_secret,
        "base_url": base_url.rstrip("/"),
        "data_url": data_url.rstrip("/"),
    }


def _get_alpaca_headers() -> Optional[Dict[str, str]]:
    """Return Alpaca auth headers if configured, otherwise None."""
    cfg = _get_alpaca_base_urls()
    api_key = cfg.get("api_key")
    api_secret = cfg.get("api_secret")

    if not api_key or not api_secret:
        return None

    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "Accept": "application/json",
    }


# ------------------------------- Snapshots ---------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def fetch_alpaca_snapshots(symbols: List[str]) -> Dict[str, dict]:
    """
    Fetch snapshot data for a list of symbols from Alpaca.

    Returns a mapping:
      {
        "AAPL": { ...raw snapshot json... },
        "MSFT": { ... },
        ...
      }

    On any configuration or network error, returns an empty dict.
    """
    # Normalize and deduplicate symbols
    if not symbols:
        return {}

    symbols = [s.upper() for s in dict.fromkeys(symbols).keys()]

    headers = _get_alpaca_headers()
    if headers is None or requests is None:
        # Alpaca not configured or requests missing – caller should treat as "no data"
        return {}

    cfg = _get_alpaca_base_urls()
    base_data_url = cfg["data_url"]

    # Alpaca snapshots endpoint: /v2/stocks/snapshots?symbols=AAPL,MSFT,...
    url = f"{base_data_url}/v2/stocks/snapshots"
    params = {"symbols": ",".join(symbols)}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
    except requests_exc.RequestException:  # type: ignore[union-attr]
        return {}

    if resp.status_code != 200:
        return {}

    try:
        data = resp.json()
    except ValueError:
        return {}

    # Alpaca returns a top-level dict keyed by symbol.
    if not isinstance(data, dict):
        return {}

    # Ensure keys are uppercased for consistency.
    normalized: Dict[str, dict] = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        normalized[k.upper()] = v

    return normalized


# ------------------------------ Public API ---------------------------------


def get_latest_quotes(
    symbols: List[str],
    session_mode: str = "regular",
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Return the latest quotes for a list of symbols using Alpaca snapshots.

    Structure:
      {
        "AAPL": {
          "last": 178.23,
          "prev_close": 176.80,
          "volume": 12345678,
        },
        ...
      }

    Notes:
      - `session_mode` is accepted for future use (e.g., premkt/after-hours tuning)
        but currently we always take Alpaca's latestTrade price, which should
        already reflect extended-hours when available.
      - If Alpaca is not configured or returns no data, this function returns {}.
        Callers may layer their own yfinance fallback on top of this.
    """
    if not symbols:
        return {}

    snapshots = fetch_alpaca_snapshots(symbols)
    if not snapshots:
        return {}

    results: Dict[str, Dict[str, Optional[float]]] = {}

    for raw_symbol in symbols:
        sym = raw_symbol.upper()
        snap = snapshots.get(sym)
        if not isinstance(snap, dict):
            continue

        latest_trade = snap.get("latestTrade") or {}
        minute_bar = snap.get("minuteBar") or {}
        daily_bar = snap.get("dailyBar") or {}
        prev_daily_bar = snap.get("prevDailyBar") or {}

        # Determine "last" price preference:
        # - latestTrade.p (most real-time)
        # - else minuteBar.c
        # - else dailyBar.c
        last: Optional[float] = None
        for candidate in (
            latest_trade.get("p"),
            minute_bar.get("c"),
            daily_bar.get("c"),
        ):
            try:
                if candidate is not None:
                    last = float(candidate)
                    break
            except (TypeError, ValueError):
                continue

        # Previous close from prevDailyBar.c, if available
        prev_close: Optional[float] = None
        try:
            if prev_daily_bar.get("c") is not None:
                prev_close = float(prev_daily_bar["c"])
        except (TypeError, ValueError):
            prev_close = None

        # Volume preference: minuteBar.v, else dailyBar.v
        volume: Optional[float] = None
        for candidate in (minute_bar.get("v"), daily_bar.get("v")):
            try:
                if candidate is not None:
                    volume = float(candidate)
                    break
            except (TypeError, ValueError):
                continue

        if last is None:
            # No usable price – skip this symbol
            continue

        results[sym] = {
            "last": last,
            "prev_close": prev_close,
            "volume": volume,
        }

    return results
