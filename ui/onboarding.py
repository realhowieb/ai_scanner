"""First-watchlist activation helpers retained after Run 75 UI consolidation.

The guided product tour is the single onboarding UI. These helpers support the
existing first-watchlist action without scanning, scoring, or other side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class FirstTickerResult:
    status: str
    ticker: str
    message: str
    opportunity: Optional[Dict[str, Any]] = None


def user_is_first_run(watchlist: List[object] | None) -> bool:
    """First-run is inferred from persisted watchlist emptiness."""
    return len(_normalize_tickers(watchlist or [])) == 0


def activation_reached(watchlist: List[object] | None, viewed_personal_intel: bool) -> bool:
    """Canonical Run 28 activation definition."""
    return bool(_normalize_tickers(watchlist or [])) and bool(viewed_personal_intel)


def current_opportunity_for_ticker(
    ticker: str,
    *,
    snapshots_loader: Optional[Callable[[], List[Dict[str, Any]]]] = None,
) -> Optional[Dict[str, Any]]:
    """Read the latest persisted Market Brief opportunity for one ticker."""
    symbol = _normalize_ticker(ticker)
    if not symbol:
        return None
    snapshots = snapshots_loader() if snapshots_loader else _load_recent_opportunity_snapshots()
    current = snapshots[0] if snapshots else {}
    for row in (current or {}).get("opportunities") or []:
        if _normalize_ticker(row.get("ticker") or row.get("Ticker") or row.get("Symbol")) == symbol:
            return dict(row)
    return None


def add_first_watch_ticker(
    user_id: str,
    ticker: object,
    *,
    existing_watchlist: Optional[List[object]] = None,
    watchlist_loader: Optional[Callable[[str], List[str]]] = None,
    add_fn: Optional[Callable[[str, str], bool]] = None,
    snapshots_loader: Optional[Callable[[], List[Dict[str, Any]]]] = None,
) -> FirstTickerResult:
    """Validate/add a first-run ticker and compose its immediate HSF context."""
    user = str(user_id or "").strip().lower()
    symbol = _normalize_ticker(ticker)
    if not symbol:
        return FirstTickerResult(
            status="invalid",
            ticker="",
            message="We couldn't recognize that ticker. Try a symbol like NVDA.",
        )

    watched = (
        _normalize_tickers(existing_watchlist)
        if existing_watchlist is not None
        else _normalize_tickers(watchlist_loader(user) if watchlist_loader else _load_user_watchlist(user))
    )
    duplicate = symbol in set(watched)
    if not duplicate:
        ok = add_fn(user, symbol) if add_fn else _add_to_persisted_watchlist(user, symbol)
        if not ok:
            return FirstTickerResult(
                status="save_failed",
                ticker=symbol,
                message="We could not save that ticker right now. Please try again.",
            )

    opportunity = current_opportunity_for_ticker(symbol, snapshots_loader=snapshots_loader)
    if duplicate:
        return FirstTickerResult(
            status="duplicate",
            ticker=symbol,
            message=f"{symbol} is already in your watchlist.",
            opportunity=opportunity,
        )
    return FirstTickerResult(
        status="added",
        ticker=symbol,
        message=f"{symbol} added to your watchlist.",
        opportunity=opportunity,
    )


def _normalize_ticker(value: object) -> str:
    try:
        from db.watchlists import normalize_watchlist_ticker

        return normalize_watchlist_ticker(value)
    except Exception:
        ticker = str(value or "").strip().upper()
        return ticker if ticker else ""


def _normalize_tickers(values: List[object] | None) -> List[str]:
    try:
        from db.watchlists import normalize_watchlist_tickers

        return normalize_watchlist_tickers(list(values or []))
    except Exception:
        return sorted({t for t in (_normalize_ticker(v) for v in (values or [])) if t})


def _load_user_watchlist(user: str) -> List[str]:
    try:
        from ui.user_cache import get_user_watchlist  # Run 71: cached per data version

        return get_user_watchlist(user)
    except Exception:
        return []


def _add_to_persisted_watchlist(user: str, ticker: str) -> bool:
    try:
        from db.watchlists import add_to_watchlist

        return bool(add_to_watchlist(user, ticker))
    except Exception:
        return False


def _load_recent_opportunity_snapshots() -> List[Dict[str, Any]]:
    try:
        from db.opportunity_snapshots import load_recent_snapshots

        return load_recent_snapshots(context="market_brief", limit=1)
    except Exception:
        return []
