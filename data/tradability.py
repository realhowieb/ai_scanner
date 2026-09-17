"""Live tradability filter — drop delisted / non-tradable symbols using Alpaca's
assets endpoint.

The scanner's static universe files (sp500.txt / nasdaq.txt) go stale: a company
that is acquired or delisted after the file was written keeps appearing (e.g. EA),
because the heuristic blocklist in data.symbols only catches known-bad / test
symbols, never a fresh delisting. This queries Alpaca once per process for the set
of ACTIVE, TRADABLE US equities and intersects the universe against it, so future
delistings drop out automatically with no manual list maintenance.

FAILS OPEN: on missing credentials, a network error, or an empty/garbled response
it returns the input unchanged, so a provider hiccup never empties a scan. The
active-symbol set is cached for the process so a multi-universe run hits the API
once.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Set

_CACHE: dict = {"symbols": None, "fetched_at": 0.0}
# Assets change slowly; one fetch per process (and re-fetch after this TTL if the
# same process lives long enough) is plenty.
_TTL_SECONDS = 6 * 3600
_TIMEOUT_SECONDS = 10


def _fetch_active_tradable_symbols() -> Optional[Set[str]]:
    """The set of active, tradable US-equity symbols from Alpaca, or None when it
    cannot be determined (caller then leaves the universe untouched)."""
    try:
        import requests

        from data.alpaca_config import get_alpaca_config, get_alpaca_headers

        cfg = get_alpaca_config()
        headers = get_alpaca_headers()
        if not cfg or not headers:
            return None
        base = (cfg.get("base_url") or "https://paper-api.alpaca.markets").rstrip("/")
        r = requests.get(
            f"{base}/v2/assets",
            params={"status": "active", "asset_class": "us_equity"},
            headers=headers, timeout=_TIMEOUT_SECONDS,
        )
        if r.status_code != 200:
            return None
        payload = r.json()
        if not isinstance(payload, list) or not payload:
            return None
        symbols = {
            str(a.get("symbol")).upper()
            for a in payload
            if isinstance(a, dict) and a.get("tradable") and str(a.get("symbol") or "").strip()
        }
        # A plausible response has thousands of names; a tiny set signals a bad
        # response we should not trust to filter the universe.
        return symbols if len(symbols) >= 500 else None
    except Exception:
        return None


def active_tradable_symbols(*, force: bool = False) -> Optional[Set[str]]:
    """Process-cached set of active, tradable symbols (None when unavailable)."""
    now = time.time()
    if not force and _CACHE["symbols"] is not None and (now - _CACHE["fetched_at"]) < _TTL_SECONDS:
        return _CACHE["symbols"]
    symbols = _fetch_active_tradable_symbols()
    if symbols is not None:
        _CACHE["symbols"] = symbols
        _CACHE["fetched_at"] = now
    return symbols


def filter_tradable_tickers(tickers: Sequence[str]) -> List[str]:
    """Return only symbols Alpaca reports as active + tradable, preserving order.

    Fails open: when the active set can't be determined, the input list is
    returned unchanged (never empties a scan over a provider issue).
    """
    items = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
    if not items:
        return list(items)
    active = active_tradable_symbols()
    if not active:
        return items  # unavailable -> leave universe untouched
    return [t for t in items if t in active]
