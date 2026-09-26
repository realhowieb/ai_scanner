"""Run 44 — canonical US_MARKET universe provider.

Builds the full currently-tradable U.S.-listed equity universe from the live
Alpaca assets endpoint (the authoritative tradability source already used by
data.tradability), with a last-known-good disk cache and an explicit fallback so
whole-market scans fail safely rather than silently shrinking to a partial list.

This module does NOT change any scoring/scanner/model logic — it only decides
WHICH symbols the scheduled scan attempts. The network fetch is isolated in
`_fetch_us_equity_assets` so tests mock it without live access.

Included:  active, tradable Alpaca `us_equity` assets on U.S. exchanges
           (NASDAQ / NYSE / NYSE American / ARCA / BATS), normalized + deduped.
Excluded:  inactive/delisted, non-tradable, malformed symbols, and SPAC
           units / warrants / rights (via data.symbols heuristics). OTC, crypto,
           options and non-U.S. assets are excluded by the endpoint query itself
           (`asset_class=us_equity`). Preferred shares are only excluded when the
           symbol carries a recognizable warrant/unit/rights marker — Alpaca does
           not expose a fine-grained security sub-type, documented as a limitation.
"""
from __future__ import annotations

import datetime as _dt
import json
import re as _re
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.symbols import is_probably_delisted, is_spac_unit_or_warrant, normalize_ticker

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "artifacts" / "universe" / "us_market.json"
# U.S. exchanges we accept from the provider (equities live on these).
US_EXCHANGES = {"NASDAQ", "NYSE", "NYSEARCA", "ARCA", "AMEX", "NYSEAMERICAN",
                "BATS", "IEX"}
# A plausible full-market response has thousands of names; anything tiny is a bad
# response we must not trust as "the market".
_MIN_PLAUSIBLE = 1000


def _fetch_us_equity_assets(timeout_s: float = 20.0) -> Optional[List[Dict[str, Any]]]:
    """Raw active us_equity assets from Alpaca, or None on any failure.
    Network is confined here so callers/tests can substitute it."""
    try:
        import requests

        from data.alpaca_config import get_alpaca_config, get_alpaca_headers

        cfg = get_alpaca_config()
        headers = get_alpaca_headers()
        if not cfg or not headers:
            return None
        base = (cfg.get("base_url") or "https://paper-api.alpaca.markets").rstrip("/")
        r = requests.get(f"{base}/v2/assets",
                         params={"status": "active", "asset_class": "us_equity"},
                         headers=headers, timeout=timeout_s)
        if r.status_code != 200:
            return None
        payload = r.json()
        return payload if isinstance(payload, list) and payload else None
    except Exception:
        return None


def _is_malformed(symbol: str) -> bool:
    if not symbol:
        return True
    # Normalized US equity tickers are short A–Z with an optional .X / -X class.
    core = symbol.replace(".", "").replace("-", "")
    return not core.isalnum() or len(symbol) > 10


# Preferred-share class suffix (Alpaca/Nasdaq convention): a "."/"-" separator then
# "PR" then an optional single class letter, e.g. PSA.PRF, PRIF.PRD, PSEC.PRA, X.PR.
# The required separator protects common stocks (PRE, PRI, PSPR never match).
_PREFERRED_RE = _re.compile(r"[.\-]PR[A-Z]?$")


def _is_preferred(symbol: str) -> bool:
    return bool(_PREFERRED_RE.search(str(symbol).upper()))


def symbol_exclusion_reason(symbol: str) -> Optional[str]:
    """Symbol-only US_MARKET rules usable without provider asset metadata
    (e.g. on stored observations): "preferred_share" / "malformed_symbol", else
    None. Deliberately excludes the SPAC unit/warrant heuristic, whose bare U/W
    suffix check also matches common stocks (MU, SNOW) outside this filter."""
    sym = normalize_ticker(str(symbol or ""))
    if _is_malformed(sym):
        return "malformed_symbol"
    if _is_preferred(sym):
        return "preferred_share"
    return None


def filter_assets(assets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the US_MARKET eligibility rules to raw provider assets. Pure —
    returns {symbols (sorted, deduped), exclusions (counts), provider_assets}."""
    excl = {"inactive": 0, "non_tradable": 0, "unsupported_asset_type": 0,
            "preferred_share": 0, "malformed_symbol": 0, "wrong_exchange": 0,
            "duplicate": 0}
    seen = set()
    out: List[str] = []
    for a in assets or []:
        if not isinstance(a, dict):
            excl["malformed_symbol"] += 1
            continue
        if str(a.get("status") or "").lower() != "active":
            excl["inactive"] += 1
            continue
        if a.get("tradable") is False:
            excl["non_tradable"] += 1
            continue
        if a.get("class") and str(a.get("class")).lower() not in ("us_equity", "equity"):
            excl["unsupported_asset_type"] += 1
            continue
        exch = str(a.get("exchange") or "").upper()
        if exch and US_EXCHANGES and exch not in US_EXCHANGES:
            excl["wrong_exchange"] += 1
            continue
        sym = normalize_ticker(str(a.get("symbol") or ""))
        if _is_malformed(sym):
            excl["malformed_symbol"] += 1
            continue
        if is_spac_unit_or_warrant(sym) or is_probably_delisted(sym):
            excl["unsupported_asset_type"] += 1
            continue
        # Exclude preferred shares (illiquid, fixed-income-like, not breakout-scan
        # targets; they were padding US_MARKET and causing provider read timeouts /
        # PRICE_DATA_UNAVAILABLE). See docs/US_MARKET_UNIVERSE.md.
        if _is_preferred(sym):
            excl["preferred_share"] += 1
            continue
        if sym in seen:
            excl["duplicate"] += 1
            continue
        seen.add(sym)
        out.append(sym)
    return {"symbols": sorted(out), "exclusions": excl,
            "provider_assets": len(assets or [])}


def _save_cache(result: Dict[str, Any]) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps({
            "generated_at": result.get("generated_at"),
            "symbols": result.get("symbols"),
            "provider_assets": result.get("provider_assets"),
            "exclusions": result.get("exclusions"),
        }))
    except Exception:
        pass


def _load_cache() -> Optional[Dict[str, Any]]:
    try:
        if not CACHE_PATH.exists():
            return None
        data = json.loads(CACHE_PATH.read_text())
        return data if data.get("symbols") else None
    except Exception:
        return None


def build_us_market_universe(
    *, fetch=None, allow_cache_fallback: bool = True,
) -> Dict[str, Any]:
    """Build the canonical US_MARKET universe.

    Returns a result with an explicit `source`:
      * "live"     — freshly fetched and filtered (also refreshes the cache),
      * "cached"   — provider failed; last-known-good cache used (FALLBACK),
      * "none"     — neither live nor cache available (scheduled scan must fail).
    Never silently substitutes a partial list. `fetch` is injectable for tests.
    """
    fetch = fetch or _fetch_us_equity_assets
    generated_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    assets = None
    try:
        assets = fetch()
    except Exception:
        assets = None

    if assets:
        filtered = filter_assets(assets)
        if len(filtered["symbols"]) >= _MIN_PLAUSIBLE:
            result = {
                "universe_name": "US_MARKET", "source": "live",
                "generated_at": generated_at, "symbols": filtered["symbols"],
                "symbol_count": len(filtered["symbols"]),
                "provider_assets": filtered["provider_assets"],
                "exclusions": filtered["exclusions"], "is_fallback": False,
            }
            _save_cache(result)
            return result

    # Provider failed or returned an implausibly small set → last-known-good.
    if allow_cache_fallback:
        cached = _load_cache()
        if cached:
            return {
                "universe_name": "US_MARKET", "source": "cached",
                "generated_at": cached.get("generated_at"),
                "cached_at": cached.get("generated_at"),
                "symbols": cached.get("symbols") or [],
                "symbol_count": len(cached.get("symbols") or []),
                "provider_assets": cached.get("provider_assets"),
                "exclusions": cached.get("exclusions"),
                "is_fallback": True,
                "fallback_reason": "provider unavailable — using last-known-good cache",
            }

    return {
        "universe_name": "US_MARKET", "source": "none", "generated_at": generated_at,
        "symbols": [], "symbol_count": 0, "is_fallback": True,
        "fallback_reason": "no live universe and no cached universe available",
    }
