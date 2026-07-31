"""Live Kalshi BTC event-contract market data (public, no auth).

Reads open Bitcoin markets from Kalshi's public Trade API and normalizes them for
the scanner: threshold, YES implied probability, spread, and time-to-close. These
are "BTC ≥ $K on <date>" contracts, so YES is the bullish side.

Read-only market data only — no account, no orders. Best-effort; returns [] on
any failure, never raises.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]

_BASE = "https://external-api.kalshi.com/trade-api/v2"
_TIMEOUT = 12.0
# BTC series on Kalshi: hourly, daily, and 15-minute price-threshold markets.
BTC_SERIES = ("KXBTC", "KXBTCD", "KXBTC15M")


def _d(v) -> Optional[float]:
    """Parse a Kalshi *_dollars string ('0.4500') to a float, or None."""
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def _ts(v) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


def _fetch_series(series: str, limit: int) -> List[Dict[str, Any]]:
    if requests is None:
        return []
    try:
        r = requests.get(
            f"{_BASE}/markets",
            params={"series_ticker": series, "status": "open", "limit": int(limit)},
            headers={"User-Agent": "hsfinest-kalshi-scanner"},
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return []
        return (r.json() or {}).get("markets", []) or []
    except Exception:
        return []


def _kind(subtitle: Optional[str]) -> str:
    """Classify a BTC contract from its subtitle: 'above' | 'below' | 'range'.

    'above' (BTC ≥ $K) → YES is bullish; 'below' (BTC ≤ $K) → YES is bearish;
    'range' ($A to $B) → non-directional bucket.
    """
    s = (subtitle or "").lower()
    if "above" in s:
        return "above"
    if "below" in s:
        return "below"
    return "range"


def _normalize(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ticker = m.get("ticker")
    if not ticker:
        return None
    yes_bid = _d(m.get("yes_bid_dollars"))
    yes_ask = _d(m.get("yes_ask_dollars"))
    # YES mid → implied probability (%). Fall back to whichever side exists.
    mids = [x for x in (yes_bid, yes_ask) if x is not None]
    yes_mid = sum(mids) / len(mids) if mids else None
    spread = (yes_ask - yes_bid) if (yes_bid is not None and yes_ask is not None) else None
    return {
        "ticker": str(ticker),
        "series": str(ticker).split("-", 1)[0],
        "title": m.get("title"),
        "threshold": m.get("subtitle"),        # e.g. "$50,000 or above"
        "kind": _kind(m.get("subtitle")),      # 'above' | 'below' | 'range'
        "floor_strike": _d(m.get("floor_strike")),
        "cap_strike": _d(m.get("cap_strike")),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "yes_prob_pct": round(yes_mid * 100, 1) if yes_mid is not None else None,
        "spread_cents": round(spread * 100, 1) if spread is not None else None,
        "last": _d(m.get("last_price_dollars")),
        "close_time": _ts(m.get("close_time")),
    }


def fetch_btc_markets(limit_per_series: int = 60, max_total: int = 40) -> List[Dict[str, Any]]:
    """Open Kalshi BTC markets, normalized, soonest-close first."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for series in BTC_SERIES:
        for raw in _fetch_series(series, limit_per_series):
            row = _normalize(raw)
            if row and row["ticker"] not in seen:
                seen.add(row["ticker"])
                out.append(row)
    out.sort(key=lambda r: (r["close_time"] is None, r["close_time"] or datetime.max.replace(tzinfo=timezone.utc)))
    return out[:max_total]


def nearest_the_money(markets: List[Dict[str, Any]], price: float) -> Optional[Dict[str, Any]]:
    """The directional (above/below) market whose strike is closest to price.

    Range buckets are skipped so the "nearest-the-money" pick is an actual
    directional contract the scanner read maps onto.
    """
    if price is None:
        return None
    cands = [
        m for m in markets
        if m.get("floor_strike") is not None and m.get("kind") in ("above", "below")
    ]
    if not cands:
        return None
    return min(cands, key=lambda m: abs(m["floor_strike"] - float(price)))
