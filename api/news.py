"""Company news from Alpaca's News API (Benzinga feed), with the market-data keys we already have.

Two reads, both shared across users through api.shared_cache:
- stock_news(ticker): recent headlines for the stock page (NEWS_TTL_S).
- news_flags(symbols): which of a set of tickers had news in the last NEWS_FLAG_HOURS,
  for the "News" marker on Scanner rows (one provider call per ticker set).

No keys, provider errors and HSF_NEWS_ENABLED=0 all answer an empty list; pages
just don't show news. Only http(s) links are passed on.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger("hsf_api")

NEWS_PATH = "/v1beta1/news"
NEWS_TTL_S = 300
NEWS_EMPTY_TTL_S = 120
NEWS_LOOKBACK_DAYS = 14
NEWS_FLAG_HOURS = 24
NEWS_MAX_ITEMS = 10
FLAG_MAX_SYMBOLS = 50
FLAG_MAX_PAGES = 3
_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_TAGS = re.compile(r"<[^>]+>")


def enabled() -> bool:
    return os.environ.get("HSF_NEWS_ENABLED", "1").strip() != "0"


def _clean_symbols(symbols: Sequence[str], cap: int) -> List[str]:
    out = sorted({str(s).strip().upper() for s in symbols if _TICKER.match(str(s).strip().upper())})
    return out[:cap]


def _text(v: Any, cap: int) -> Optional[str]:
    if not isinstance(v, str):
        return None
    v = " ".join(_TAGS.sub(" ", v).split())
    return v[:cap] or None


def _link(v: Any) -> Optional[str]:
    return v[:500] if isinstance(v, str) and v.startswith(("https://", "http://")) else None


def _item(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    headline = _text(raw.get("headline"), 300)
    when = raw.get("created_at") if isinstance(raw.get("created_at"), str) else None
    if not headline or not when:
        return None
    return {"id": str(raw.get("id") or "")[:40], "headline": headline, "summary": _text(raw.get("summary"), 500),
            "source": _text(raw.get("source"), 40), "author": _text(raw.get("author"), 80), "url": _link(raw.get("url")),
            "published_at": when, "symbols": _clean_symbols(raw.get("symbols") or [], 12)}


def _fetch(params: Dict[str, Any], *, client=None) -> Dict[str, Any]:
    """One page from Alpaca's news endpoint. Raises on HTTP or network errors."""
    from data.alpaca_config import get_alpaca_config, get_alpaca_headers

    cfg, headers = get_alpaca_config(), get_alpaca_headers()
    if not cfg or not headers:
        return {"news": []}
    import httpx

    own = client is None
    client = client or httpx.Client(timeout=6.0)
    try:
        r = client.get(cfg["data_url"] + NEWS_PATH, params=params, headers=headers)
        r.raise_for_status()
        body = r.json()
        return body if isinstance(body, dict) else {"news": []}
    finally:
        if own:
            client.close()


def _since(hours: float, now: Optional[dt.datetime] = None) -> str:
    now = now or dt.datetime.now(dt.timezone.utc)
    return (now - dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def stock_news(ticker: str, limit: int = NEWS_MAX_ITEMS) -> Dict[str, Any]:
    """{ticker, items: newest first, available}. available=False when news is off or unreachable."""
    syms = _clean_symbols([ticker], 1)
    if not syms or not enabled():
        return {"ticker": (ticker or "").upper()[:10], "items": [], "available": False}
    t = syms[0]

    def load() -> List[Dict[str, Any]]:
        body = _fetch({"symbols": t, "limit": NEWS_MAX_ITEMS, "sort": "desc", "include_content": "false",
                       "exclude_contentless": "false", "start": _since(NEWS_LOOKBACK_DAYS * 24)})
        return [i for i in (_item(n) for n in body.get("news") or [] if isinstance(n, dict)) if i]

    from api import shared_cache

    try:
        items = shared_cache.cached(f"news:{t}", load, NEWS_TTL_S, empty_ttl_s=NEWS_EMPTY_TTL_S,
                                    is_empty=lambda v: v is None)
    except Exception as e:
        log.warning(json.dumps({"event": "news_fetch_failed", "error": type(e).__name__}))
        return {"ticker": t, "items": [], "available": False}
    return {"ticker": t, "items": (items or [])[:max(1, min(limit, NEWS_MAX_ITEMS))], "available": True}


def news_flags(symbols: Sequence[str]) -> Dict[str, Any]:
    """{flags: {TICKER: {count, latest_at, headline}}} for tickers with news in the last
    NEWS_FLAG_HOURS. Up to FLAG_MAX_PAGES pages of 50 stories, so a very busy news day
    can leave a quiet ticker unflagged; never a false flag."""
    syms = _clean_symbols(symbols, FLAG_MAX_SYMBOLS)
    if not syms or not enabled():
        return {"flags": {}, "hours": NEWS_FLAG_HOURS}
    wanted = set(syms)

    def load() -> Dict[str, Any]:
        flags: Dict[str, Dict[str, Any]] = {}
        token = None
        for _ in range(FLAG_MAX_PAGES):
            params = {"symbols": ",".join(syms), "limit": 50, "sort": "desc", "include_content": "false",
                      "start": _since(NEWS_FLAG_HOURS)}
            if token:
                params["page_token"] = token
            body = _fetch(params)
            for raw in body.get("news") or []:
                item = _item(raw) if isinstance(raw, dict) else None
                if not item:
                    continue
                for s in item["symbols"]:
                    if s not in wanted:
                        continue
                    f = flags.setdefault(s, {"count": 0, "latest_at": item["published_at"], "headline": item["headline"]})
                    f["count"] += 1
            token = body.get("next_page_token")
            if not token:
                break
        return flags

    from api import shared_cache

    try:
        flags = shared_cache.cached(f"newsflags:{shared_cache.hashed(','.join(syms))}", load, NEWS_TTL_S,
                                    empty_ttl_s=NEWS_EMPTY_TTL_S, is_empty=lambda v: not v)
    except Exception as e:
        log.warning(json.dumps({"event": "news_flags_failed", "error": type(e).__name__}))
        flags = {}
    return {"flags": flags or {}, "hours": NEWS_FLAG_HOURS}
