"""Company and ticker suggestions using the existing Alpaca asset provider."""
from __future__ import annotations

import re
import threading
import time

_lock = threading.Lock()
_assets: list[dict] = []
_expires = 0.0


def _rows() -> list[dict]:
    """The symbol directory, refreshed at most every 6 hours (a minute after a failure)."""
    global _assets, _expires
    # A single bounded refresh per process; stale names remain useful during an outage.
    with _lock:
        if time.monotonic() >= _expires:
            from api.operations import record_provider
            from data.us_market_universe import _fetch_us_equity_assets

            started = time.monotonic()
            assets = _fetch_us_equity_assets(timeout_s=5)
            record_provider("symbol_directory", bool(assets), (time.monotonic() - started) * 1000)
            if assets:
                _assets = [{"ticker": str(a.get("symbol", "")).upper(), "name": str(a.get("name") or a.get("symbol", ""))[:200]}
                           for a in assets if isinstance(a, dict) and a.get("tradable") and a.get("status") == "active"
                           and re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", str(a.get("symbol", "")).upper())]
            _expires = time.monotonic() + (21600 if assets else 60)
        return list(_assets)


def lookup(ticker: str) -> dict | None:
    """{ticker, name} for an active US-listed symbol, else None."""
    t = ticker.strip().upper()
    return next((a for a in _rows() if a["ticker"] == t), None)


def search(query: str) -> dict:
    q = query.strip().casefold()
    if not q:
        return {"results": [], "available": True}
    rows = _rows()
    matches = [a for a in rows if q in a["ticker"].casefold() or q in a["name"].casefold()]
    matches.sort(key=lambda a: (a["ticker"].casefold() != q, not a["ticker"].casefold().startswith(q), a["ticker"]))
    return {"results": matches[:8], "available": bool(rows)}
