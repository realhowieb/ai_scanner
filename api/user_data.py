"""Watchlists and alerts for the API (P1-59 step 6).

Thin layer over db.watchlists and db.alerts, the same functions the web app
uses, so ownership checks, name rules and duplicate checks are shared. Adds
the input checks the web app does in its forms, and turns "database not
available" into DatabaseUnavailable (503).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from api.store import DatabaseUnavailable

MAX_WATCHLISTS = 50          # per account; protects the database from scripted clients
MAX_TICKERS_PER_REQUEST = 200  # same as the web app's watchlist import
MAX_NOTE_LEN = 500


class NotFound(LookupError):
    """The watchlist or alert doesn't exist or belongs to someone else (404)."""


class Conflict(ValueError):
    """Duplicate name/alert or a plan limit (409 / 403 decided by the route)."""


class LimitReached(Conflict):
    pass


def _db(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        if "Neon is not available" in str(e):
            raise DatabaseUnavailable("database unavailable") from e
        raise


# ---- watchlists -----------------------------------------------------------------------------------
def list_watchlists(user: str) -> List[Dict[str, Any]]:
    from db import watchlists as wl

    return [{"id": int(w["id"]), "name": w["name"], "is_default": bool(w["is_default"]),
             "symbol_count": int(w["symbol_count"])} for w in _db(wl.list_watchlists, user)]


def _owned(user: str, watchlist_id: int) -> Dict[str, Any]:
    for w in list_watchlists(user):
        if w["id"] == int(watchlist_id):
            return w
    raise NotFound("watchlist")


def get_watchlist(user: str, watchlist_id: int) -> Dict[str, Any]:
    from db import watchlists as wl

    w = _owned(user, watchlist_id)
    items = _db(wl.get_watchlist_items, int(watchlist_id), user)
    return {**w, "items": [{"ticker": i["ticker"], "added_at": i.get("date_added"),
                            "price_when_added": i.get("price_when_added"), "note": i.get("note")}
                           for i in items]}


def create_watchlist(user: str, name: str, make_default: bool = False) -> Dict[str, Any]:
    from db import watchlists as wl

    try:
        new_id = _db(wl.create_watchlist, user, name, make_default=make_default, max_watchlists=MAX_WATCHLISTS)
    except wl.WatchlistLimitReached as e:
        raise LimitReached(str(e)) from e
    except ValueError as e:
        raise Conflict(str(e)) from e
    return get_watchlist(user, new_id)


def update_watchlist(user: str, watchlist_id: int, *, name: Optional[str], make_default: bool) -> Dict[str, Any]:
    from db import watchlists as wl

    _owned(user, watchlist_id)
    if name is not None:
        try:
            _db(wl.rename_watchlist, int(watchlist_id), user, name)
        except ValueError as e:
            raise Conflict(str(e)) from e
    if make_default:
        _db(wl.set_default_watchlist, int(watchlist_id), user)
    return get_watchlist(user, watchlist_id)


def delete_watchlist(user: str, watchlist_id: int) -> None:
    from db import watchlists as wl

    _owned(user, watchlist_id)
    _db(wl.delete_watchlist, int(watchlist_id), user)


def add_tickers(user: str, watchlist_id: int, tickers: List[str]) -> Dict[str, List[str]]:
    from db import watchlists as wl

    _owned(user, watchlist_id)
    cleaned = [str(t or "").strip().upper() for t in tickers]
    invalid = sorted({t for t in cleaned if not wl.normalize_watchlist_ticker(t)})
    result = _db(wl.add_tickers_to_watchlist, user, [t for t in cleaned if t not in invalid], int(watchlist_id))
    return {"added": list(result["added"]), "already_present": list(result["already_present"]),
            "invalid": invalid}


def remove_ticker(user: str, watchlist_id: int, ticker: str) -> None:
    from db import watchlists as wl

    _owned(user, watchlist_id)
    if not _db(wl.remove_from_watchlist, user, ticker, int(watchlist_id)):
        raise NotFound("ticker")


def set_note(user: str, watchlist_id: int, ticker: str, note: Optional[str]) -> None:
    from db import watchlists as wl

    _owned(user, watchlist_id)
    if not _db(wl.update_watchlist_item_note, int(watchlist_id), user, ticker, (note or "")[:MAX_NOTE_LEN]):
        raise NotFound("ticker")


def watchlists_with(user: str, ticker: str) -> List[Dict[str, Any]]:
    """The user's watchlists that contain `ticker` (id and name)."""
    from db import watchlists as wl

    out = []
    for w in list_watchlists(user):
        if ticker in _db(wl.get_watchlist_tickers, w["id"], user):
            out.append({"id": w["id"], "name": w["name"]})
    return out


# ---- alerts ---------------------------------------------------------------------------------------
# Input rules per type, as in the web app's alert forms (ui/alerts.py).
ALERT_RULES: Dict[str, Dict[str, Any]] = {
    "breakout": {"ticker": False, "threshold_min": 0.0, "directions": None},
    "watchlist": {"ticker": False, "threshold_min": None, "directions": None},
    "price": {"ticker": True, "threshold_min": 0.0, "threshold_gt": True, "directions": ("above", "below")},
    "move": {"ticker": True, "threshold_min": 0.5, "directions": None},
    "rvol": {"ticker": True, "threshold_min": 1.0, "directions": None},
    "ema_cross": {"ticker": True, "threshold_min": None, "directions": ("bullish", "bearish")},
    "ewo_cross": {"ticker": True, "threshold_min": None, "directions": ("up", "down")},
}
THRESHOLD_MAX = 1_000_000.0

# What each type means and the web form's defaults (ui/alerts.py, ui/alert_copy.py), so
# clients build their forms from the rules above instead of copying them.
ALERT_COPY: Dict[str, Dict[str, Any]] = {
    "breakout": {"label": "Breakout", "threshold_label": "Breakout Score at or above", "default_threshold": 8.0,
                 "description": "Fires when a scan finds a ticker whose Breakout Score is at or above your value "
                                "(the scanner's supporting technical score, not the 0-100 HSF Score). Lower "
                                "values fire more often."},
    "watchlist": {"label": "Watchlist", "description": "Fires when any ticker on your watchlist shows up in the "
                                                       "scan results."},
    "price": {"label": "Price", "threshold_label": "Price ($)", "default_threshold": None,
              "description": "Fires when a ticker crosses a price you set."},
    "move": {"label": "% move", "threshold_label": "Move at least (%)", "default_threshold": 5.0,
             "description": "Fires when a ticker moves more than ±X% vs yesterday's close. Checked live "
                            "(about every 60 s) during extended hours."},
    "rvol": {"label": "Relative volume", "threshold_label": "Times 20-day average volume", "default_threshold": 2.0,
             "description": "Fires when a ticker trades at X times its 20-day average volume. Checked live "
                            "(about every 60 s) during extended hours."},
    "ema_cross": {"label": "EMA cross", "description": "Fires when EMA 9 crosses EMA 21. Bullish is a short-term "
                                                       "Golden Cross; bearish is a short-term Death Cross."},
    "ewo_cross": {"label": "EWO cross", "description": "Fires when the Elliott Wave Oscillator (SMA 5 − SMA 35 of "
                                                       "close) crosses zero. Up is a bullish momentum flip; down "
                                                       "is bearish."},
}


def alert_types() -> List[Dict[str, Any]]:
    """The alert types and their input rules (the same ALERT_RULES validate_alert enforces)."""
    out = []
    for t, r in ALERT_RULES.items():
        copy = ALERT_COPY.get(t, {})
        threshold = None
        if r["threshold_min"] is not None:
            threshold = {"min": r["threshold_min"], "min_exclusive": bool(r.get("threshold_gt")),
                         "max": THRESHOLD_MAX, "label": copy.get("threshold_label") or "Threshold",
                         "default": copy.get("default_threshold")}
        out.append({"type": t, "label": copy.get("label") or t, "description": copy.get("description") or "",
                    "needs_ticker": bool(r["ticker"]), "threshold": threshold,
                    "directions": list(r["directions"] or []), "watchlist_only_option": t == "breakout"})
    return out


def validate_alert(alert_type: str, ticker: Optional[str], threshold: Optional[float],
                   direction: Optional[str], watchlist_only: bool) -> Dict[str, Any]:
    """Normalized create_alert kwargs, or ValueError with a message for the user."""
    from db.watchlists import normalize_watchlist_ticker

    rules = ALERT_RULES.get(alert_type)
    if rules is None:
        raise ValueError(f"Unknown alert type. Use one of: {', '.join(ALERT_RULES)}.")
    out: Dict[str, Any] = {"ticker": None, "threshold": None, "direction": None, "watchlist_only": False}
    if rules["ticker"]:
        t = normalize_watchlist_ticker(ticker)
        if not t:
            raise ValueError("Enter a valid ticker symbol.")
        out["ticker"] = t
    if rules["threshold_min"] is not None:
        if threshold is None:
            raise ValueError("Enter a threshold.")
        lo = rules["threshold_min"]
        if threshold < lo or (rules.get("threshold_gt") and threshold <= lo) or threshold > THRESHOLD_MAX:
            raise ValueError(f"Threshold must be {'greater than' if rules.get('threshold_gt') else 'at least'} {lo:g}.")
        out["threshold"] = float(threshold)
    if rules["directions"]:
        if direction not in rules["directions"]:
            raise ValueError(f"Direction must be one of: {', '.join(rules['directions'])}.")
        out["direction"] = direction
    if alert_type == "breakout":
        out["watchlist_only"] = bool(watchlist_only)
    return out


def _alert_out(a: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": int(a["id"]), "type": a["alert_type"], "ticker": a.get("ticker"),
            "threshold": a.get("threshold"), "direction": a.get("direction"),
            "watchlist_only": bool(a.get("watchlist_only")), "enabled": bool(a.get("enabled")),
            "last_fired_at": a.get("last_fired_at"), "created_at": a.get("created_at")}


def list_alerts(user: str) -> List[Dict[str, Any]]:
    from db import alerts

    return [_alert_out(a) for a in _db(alerts.list_alerts, user)]


def create_alert(user: str, alert_limit: int, alert_type: str, **fields: Any) -> Dict[str, Any]:
    """The limit and duplicate checks run with the insert under a per-user lock
    (db.alerts.create_alert), so concurrent requests can't exceed the plan limit."""
    from db import alerts

    clean = validate_alert(alert_type, fields.get("ticker"), fields.get("threshold"),
                           fields.get("direction"), bool(fields.get("watchlist_only")))
    try:
        new_id = _db(alerts.create_alert, user, alert_type, max_alerts=int(alert_limit), **clean)
    except alerts.AlertLimitReached as e:
        raise LimitReached(str(e)) from e
    except ValueError as e:
        raise Conflict(str(e)) from e
    return _owned_alert(user, int(new_id))


def _owned_alert(user: str, alert_id: int) -> Dict[str, Any]:
    for a in list_alerts(user):
        if a["id"] == int(alert_id):
            return a
    raise NotFound("alert")


def set_alert_enabled(user: str, alert_id: int, enabled: bool) -> Dict[str, Any]:
    from db import alerts

    _owned_alert(user, alert_id)
    _db(alerts.set_alert_enabled, int(alert_id), user, bool(enabled))
    return _owned_alert(user, alert_id)


def delete_alert(user: str, alert_id: int) -> None:
    from db import alerts

    _owned_alert(user, alert_id)
    _db(alerts.delete_alert, int(alert_id), user)


def alert_events(user: str, limit: int) -> List[Dict[str, Any]]:
    from db import alerts

    return [{"id": int(e["id"]), "alert_id": e.get("alert_id"), "ticker": e.get("ticker"),
             "message": str(e.get("message") or ""), "fired_at": e.get("fired_at")}
            for e in _db(alerts.list_recent_events, user, int(limit))]


def alerts_for(user: str, ticker: str) -> List[Dict[str, Any]]:
    return [a for a in list_alerts(user) if a.get("ticker") == ticker]
