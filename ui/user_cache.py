"""Run 71 — cached per-user reads for the per-rerun paths.

Watchlist reads are cached per (user, data version): `db.watchlists` bumps a
user's version on every write, so a change made on any page or session in this
process is visible on the very next rerun, while repeated reruns with no change
skip the database. The TTL only bounds staleness for writes made elsewhere.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

TTL_S = 120


def _version(user_id: str) -> int:
    # NB: the cached functions below take this as `version` (no leading
    # underscore) — st.cache_data does not hash underscore-prefixed parameters.
    from db.watchlists import data_version

    return data_version(user_id)


def _list_watchlists(user_id: str, version: int) -> List[Dict[str, Any]]:
    from db.watchlists import list_watchlists

    return list_watchlists(user_id)


def _watchlist_tickers(watchlist_id: int, user_id: str, version: int) -> List[str]:
    from db.watchlists import get_watchlist_tickers

    return get_watchlist_tickers(watchlist_id, user_id)


def _user_watchlist(user_id: str, version: int) -> List[str]:
    from db.watchlists import get_user_watchlist

    return get_user_watchlist(user_id)


def _default_watchlist_id(user_id: str, version: int) -> Optional[int]:
    from db.watchlists import get_default_watchlist_id

    return get_default_watchlist_id(user_id)


if st is not None:
    _cache = st.cache_data(ttl=TTL_S, show_spinner=False, max_entries=2000)
    _list_watchlists = _cache(_list_watchlists)
    _watchlist_tickers = _cache(_watchlist_tickers)
    _user_watchlist = _cache(_user_watchlist)
    _default_watchlist_id = _cache(_default_watchlist_id)


def list_watchlists(user_id: str) -> List[Dict[str, Any]]:
    return _list_watchlists(user_id, _version(user_id))


def get_watchlist_tickers(watchlist_id: int, user_id: str) -> List[str]:
    return _watchlist_tickers(watchlist_id, user_id, _version(user_id))


def get_user_watchlist(user_id: str) -> List[str]:
    return _user_watchlist(user_id, _version(user_id))


def get_default_watchlist_id(user_id: str) -> Optional[int]:
    return _default_watchlist_id(user_id, _version(user_id))


def _watchlist_summary(user_id: str, version: int) -> Dict[str, Any]:
    from analytics.watchlist_intelligence import build_watchlist_intelligence

    return dict((build_watchlist_intelligence(user_id) or {}).get("summary") or {})


if st is not None:
    _watchlist_summary = _cache(_watchlist_summary)


def watchlist_summary(user_id: str) -> Dict[str, Any]:
    """Needs-attention / strengthening / fading counts for the user's watchlist
    (cached per data version; the TTL bounds changes from new scans/alerts)."""
    return _watchlist_summary(user_id, _version(user_id))


def summary_line(summary: Dict[str, Any]) -> Optional[str]:
    tracked = int(summary.get("tracked") or 0)
    if not tracked:
        return None
    return (f"{tracked} watched · {int(summary.get('needs_attention') or 0)} need attention · "
            f"{int(summary.get('strengthening') or 0)} strengthening · {int(summary.get('fading') or 0)} fading")
