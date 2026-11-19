"""The market condition badge and its per-day history (analytics.market_condition).

SPY's daily closes for about a year come from the same provider batch the
watchlist performance uses, shared for CLOSES_TTL_S. With no closes (provider
down, no keys) the condition is unavailable: it is never guessed.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

from analytics import market_condition as mc
from api.today import TTLCache

BENCHMARK = "SPY"
CLOSES_TTL_S = 1800
_cache = TTLCache(8, name="market_condition")


def _spy_closes() -> List[Tuple[str, float]]:
    def load() -> List[Tuple[str, float]]:
        from api.watchlist_perf import _closes

        return list(_closes([BENCHMARK]).get(BENCHMARK) or [])

    return _cache.get("spy", load, ttl_s=CLOSES_TTL_S, stale_s=6 * 3600)


def current() -> Dict[str, Any]:
    """{available, condition, label, advice, why, as_of, ...inputs}."""
    try:
        series = _spy_closes()
    except Exception:
        series = []
    c = mc.classify([v for _d, v in series]) if series else None
    if not c:
        return {"available": False, "condition": None, "label": None, "advice": None, "why": None,
                "as_of": None, "rule": mc.RULE_VERSION}
    return {"available": True, **c, "why": mc.why(c), "as_of": series[-1][0]}


def labels_by_day() -> Dict[str, str]:
    """{ISO day: condition} for the last year (empty when SPY closes are unavailable)."""
    def load() -> Dict[str, str]:
        return mc.by_day(_spy_closes())

    try:
        return _cache.get("by_day", load, ttl_s=CLOSES_TTL_S, stale_s=6 * 3600)
    except Exception:
        return {}


def condition_of(labels: Dict[str, str]):
    """A fast day -> condition lookup over `labels` (see mc.condition_on)."""
    days = sorted(labels)

    def of(day: Any) -> Optional[str]:
        if day is None:
            return None
        key = day.isoformat() if isinstance(day, (dt.date, dt.datetime)) else str(day)
        return mc.condition_on(key[:10], labels, days)

    return of
