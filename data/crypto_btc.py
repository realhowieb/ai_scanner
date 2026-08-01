"""BTC-USD OHLCV bars from Coinbase's public candles API (no key required).

Kept independent of the Alpaca market-data key so the Kalshi BTC scanner works
even when that key is rotated/down. Returns a pandas DataFrame with
Open/High/Low/Close/Volume and a UTC DatetimeIndex, oldest first. Best-effort —
returns None on any failure, never raises.
"""
from __future__ import annotations

from typing import Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]

_BASE = "https://api.exchange.coinbase.com"
_TIMEOUT = 12.0

# Coinbase candle granularities (seconds) → friendly label.
GRANULARITIES = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
}


def _fetch(granularity_s: int, product: str = "BTC-USD"):
    if requests is None:
        return None
    try:
        r = requests.get(
            f"{_BASE}/products/{product}/candles",
            params={"granularity": int(granularity_s)},
            headers={"User-Agent": "hsfinest-kalshi-scanner"},
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        return r.json() or []
    except Exception:
        return None


def _to_frame(rows):
    """Coinbase returns [time, low, high, open, close, volume], newest first."""
    try:
        import pandas as pd

        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["time", "Low", "High", "Open", "Close", "Volume"])
        df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("ts").sort_index()  # oldest first
        for col in ("Open", "High", "Low", "Close", "Volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return None


def fetch_btc_bars(timeframe: str = "5m"):
    """BTC-USD OHLCV bars for a timeframe key (see GRANULARITIES), or None."""
    gran = GRANULARITIES.get(timeframe, 300)
    return _to_frame(_fetch(gran))


def btc_24h_stats() -> Optional[dict]:
    """Live BTC-USD snapshot: {price, open_24h, change_pct, high_24h, low_24h}.

    Coinbase's 24-hour stats endpoint; used for the live price header. None on
    failure.
    """
    if requests is None:
        return None
    try:
        r = requests.get(
            f"{_BASE}/products/BTC-USD/stats",
            headers={"User-Agent": "hsfinest-kalshi-scanner"}, timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        d = r.json() or {}
    except Exception:
        return None

    def _f(x):
        try:
            return None if x is None else float(x)
        except (TypeError, ValueError):
            return None

    last, op = _f(d.get("last")), _f(d.get("open"))
    chg = ((last - op) / op * 100.0) if (last and op) else None
    return {
        "price": last,
        "open_24h": op,
        "change_pct": chg,
        "high_24h": _f(d.get("high")),
        "low_24h": _f(d.get("low")),
    }


def latest_btc_price() -> Optional[float]:
    """Spot BTC-USD price, or None. Cheap ticker endpoint."""
    if requests is None:
        return None
    try:
        r = requests.get(
            f"{_BASE}/products/BTC-USD/ticker",
            headers={"User-Agent": "hsfinest-kalshi-scanner"}, timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        px = (r.json() or {}).get("price")
        return float(px) if px is not None else None
    except Exception:
        return None
