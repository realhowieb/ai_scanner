"""Open in Webull" links.

Webull's quote pages need the listing exchange in the URL
(https://www.webull.com/quote/nasdaq-aapl, /quote/nysearca-spy, /quote/nyse-brk-b);
a wrong or missing exchange is a 404. We get the exchange from Alpaca's single-asset
endpoint (GET /v2/assets/{symbol}), one small call per ticker, cached for the process.

Fails soft: when the exchange can't be found the link goes to Webull's U.S. markets
page instead, so the button never lands on a 404.
"""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, Optional

WEBULL_FALLBACK_URL = "https://www.webull.com/quote/us"
# Alpaca exchange code -> Webull URL prefix (checked against live Webull pages 2026-11-20).
WEBULL_EXCHANGE = {
    "NASDAQ": "nasdaq",
    "NYSE": "nyse",
    "ARCA": "nysearca",
    "NYSEARCA": "nysearca",
    "AMEX": "amex",
    "NYSEAMERICAN": "amex",
    "BATS": "bats",
    "OTC": "otcpk",
}
_TICKER = re.compile(r"^[A-Z][A-Z0-9]{0,5}([.\-][A-Z]{1,2})?$")
_TTL_S = 24 * 3600
_MISS_TTL_S = 3600
_MAX = 4000
_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_LOCK = threading.Lock()


def clean_ticker(raw: str) -> Optional[str]:
    t = str(raw or "").strip().upper()
    return t if _TICKER.fullmatch(t) else None


def webull_quote_url(ticker: str, exchange: Optional[str]) -> Optional[str]:
    """Webull quote page for a ticker on an Alpaca exchange code, or None when unknown."""
    t = clean_ticker(ticker)
    slug = WEBULL_EXCHANGE.get(str(exchange or "").upper())
    if not t or not slug:
        return None
    return f"https://www.webull.com/quote/{slug}-{t.replace('.', '-').lower()}"


def _fetch_exchange(ticker: str) -> Optional[str]:
    """Alpaca's exchange code for a ticker (BRK-B is looked up as BRK.B), or None."""
    try:
        import requests

        from data.alpaca_config import get_alpaca_config, get_alpaca_headers

        cfg = get_alpaca_config()
        headers = get_alpaca_headers()
        if not cfg or not headers:
            return None
        base = (cfg.get("base_url") or "https://paper-api.alpaca.markets").rstrip("/")
        sym = ticker.replace("-", ".")
        r = requests.get(f"{base}/v2/assets/{sym}", headers=headers, timeout=5)
        if r.status_code != 200:
            return None
        body = r.json()
        exch = str((body or {}).get("exchange") or "").upper() if isinstance(body, dict) else ""
        return exch or None
    except Exception:
        return None


def exchange_for(ticker: str, *, fetch: Callable[[str], Optional[str]] = _fetch_exchange) -> Optional[str]:
    """Cached exchange lookup. Misses are cached for an hour so a bad symbol can't
    hammer Alpaca."""
    t = clean_ticker(ticker)
    if not t:
        return None
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(t)
        if hit and now < hit[1]:
            _CACHE.move_to_end(t)
            return hit[0]
    exch = fetch(t)
    with _LOCK:
        _CACHE[t] = (exch, now + (_TTL_S if exch else _MISS_TTL_S))
        _CACHE.move_to_end(t)
        while len(_CACHE) > _MAX:
            _CACHE.popitem(last=False)
    return exch


def webull_link(ticker: str, *, fetch: Callable[[str], Optional[str]] = _fetch_exchange) -> Dict[str, object]:
    """{ticker, exchange, url, exact}: exact is False when the link falls back to
    Webull's markets page."""
    t = clean_ticker(ticker) or ""
    exch = exchange_for(t, fetch=fetch) if t else None
    url = webull_quote_url(t, exch) if t else None
    return {"ticker": t, "exchange": exch, "url": url or WEBULL_FALLBACK_URL, "exact": bool(url)}


def _reset_cache() -> None:
    with _LOCK:
        _CACHE.clear()
