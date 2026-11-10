"""Previews behind locked features: one real example and a count, from data the API
already keeps warm, so a Starter account sees what an upgrade shows before paying.

    day_trader  the top live mover by day-trade rank, and how many movers are on screen
    gappers     the biggest pre-market gap right now, and how many stocks are gapping
    earnings    the biggest company reporting this week, and how many companies report

Only the single top example is shown; the rest stay behind the lock. Everything comes
from shared caches (Day Trader movers are kept warm, gappers refresh every 3 min, the
earnings week once a day), so a preview adds no data fetches of its own and almost no
memory. Each answer is cached for CACHE_S. Best effort: an empty preview on any error.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Any, Callable, Dict, Optional

CACHE_S = 120
KINDS = ("day_trader", "gappers", "earnings")
_cache: Dict[str, tuple] = {}
_lock = threading.Lock()


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _pct(v: Optional[float]) -> str:
    return f"{v:+.1f}%" if v is not None else ""


def day_trader_preview() -> Dict[str, Any]:
    from api import market

    dt_ = market.day_trader("movers")
    rows = [r for r in dt_.get("rows") or [] if not r.get("quote_flags")]
    out: Dict[str, Any] = {"count": len(rows), "state": dt_.get("state")}
    if rows:
        r = rows[0]
        rvol = _num(r.get("rvol"))
        out["example"] = {"ticker": str(r.get("ticker") or "").upper(),
                          "detail": " · ".join(x for x in (_pct(_num(r.get("chg_pct"))),
                                                           f"{rvol:.1f}x volume" if rvol else "") if x)}
    return out


def gappers_preview() -> Dict[str, Any]:
    from api import market

    g = market.premarket_gappers(25)
    rows = list(g.get("rows") or [])
    out: Dict[str, Any] = {"count": len(rows), "state": "premarket" if g.get("active") else "closed"}
    if rows:
        r = max(rows, key=lambda x: abs(_num(x.get("pct")) or 0.0))
        out["example"] = {"ticker": str(r.get("ticker") or "").upper(), "detail": f"gapping {_pct(_num(r.get('pct')))}"}
    return out


def earnings_preview() -> Dict[str, Any]:
    from api import earnings_week as ew

    base = ew.cached_base()
    rows = [r for d in base.get("days") or [] for r in d.get("rows") or []]
    out: Dict[str, Any] = {"count": len(rows), "state": None}
    if rows:
        r = max(rows, key=lambda x: _num(x.get("dollar_volume")) or 0.0)
        when = {"bmo": "before the open", "amc": "after the close"}.get(str(r.get("time") or ""), "")
        try:
            day = dt.date.fromisoformat(str(r.get("date"))).strftime("%A")
        except ValueError:
            day = ""
        out["example"] = {"ticker": str(r.get("ticker") or "").upper(),
                          "detail": f"reports {day} {when}".strip() if day else "reports this week"}
    return out


_BUILDERS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "day_trader": day_trader_preview, "gappers": gappers_preview, "earnings": earnings_preview}


def preview(kind: str) -> Dict[str, Any]:
    now = time.monotonic()
    with _lock:
        hit = _cache.get(kind)
        if hit and now - hit[0] < CACHE_S:
            return hit[1]
    try:
        body = _BUILDERS[kind]()
    except Exception:
        body = {"count": 0, "state": None}
    out = {"kind": kind, "count": int(body.get("count") or 0), "state": body.get("state"),
           "example": body.get("example"), "as_of": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    with _lock:
        _cache[kind] = (now, out)
    return out
