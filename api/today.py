"""GET /today: the Today page as data (Before the open, Top setups, After the close,
last session recap), built with the same helpers as the Streamlit Today page."""
from __future__ import annotations

import datetime as dt
import json
import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from analytics import market_calendar as mc

CACHE_TTL_S = 60
_cache: Dict[Any, tuple[float, Any]] = {}
_lock = threading.Lock()


def _cached(key: Any, loader: Callable[[], Any]) -> Any:
    """Scans change a few times a day; one read per minute is plenty."""
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL_S:
            return hit[1]
    value = loader()
    with _lock:
        _cache[key] = (now, value)
    return value


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def _num(v: Any) -> Optional[float]:
    """JSON-safe number: NaN/inf and non-numbers become None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _iso(ts: Any) -> Optional[str]:
    return ts.isoformat() if isinstance(ts, (dt.datetime, dt.date)) else None


def market_runs() -> List[Dict[str, Any]]:
    from db.runs import list_runs
    from ui.market_scans import MARKET_USER
    from ui.market_scans import market_runs as _market_runs

    return _cached("market_runs", lambda: _market_runs(
        list_runs(limit=60, include_snapshots=True, username=MARKET_USER) or []))


def session_runs() -> List[Dict[str, Any]]:
    from db.runs import list_runs

    return _cached("session_runs", lambda: list_runs(limit=60, include_snapshots=False, username="scheduler") or [])


def run_df(run_id: int):
    """A saved run's results as a DataFrame (None when missing or empty)."""
    def load():
        import pandas as pd

        from db.runs import load_run_results

        raw = load_run_results(int(run_id))
        if not raw:
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        except (TypeError, ValueError):
            return None
        return None if df.empty else df

    return _cached(("run", int(run_id)), load)


def market_phase(now: dt.datetime) -> str:
    """premarket | open | afterhours | closed (US equities, ET)."""
    day = now.astimezone(mc.ET).date()
    if not mc.is_trading_day(day):
        return "closed"
    open_utc, close_utc = mc.session_bounds_utc(day)
    if open_utc <= now < close_utc:
        return "open"
    et_minutes = now.astimezone(mc.ET).hour * 60 + now.astimezone(mc.ET).minute
    if now < open_utc and et_minutes >= 4 * 60:
        return "premarket"
    if now >= close_utc and et_minutes < 20 * 60:
        return "afterhours"
    return "closed"


def _movers(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"ticker": m["ticker"], "pct": _num(m.get("pct")), "last": _num(m.get("last")),
             "score": m.get("score")} for m in rows]


def before_open(now: dt.datetime, entitled: bool) -> Optional[Dict[str, Any]]:
    """None outside the pre-open window or before this morning's scan exists."""
    from ui.before_open import current_premarket_run, premarket_movers

    run = current_premarket_run(session_runs(), now)
    if run is None:
        return None
    out = {"scan_at": _iso(run["created_at"]), "locked": not entitled, "movers": []}
    if entitled:
        out["movers"] = _movers(premarket_movers(run_df(int(run["id"]))))
    return out


def after_close(now: dt.datetime, entitled: bool) -> Optional[Dict[str, Any]]:
    """None during market hours or with no after-hours scan since the last close."""
    from ui.after_close import current_postmarket_run, session_movers

    run = current_postmarket_run(session_runs(), now)
    if run is None:
        return None
    out = {"scan_at": _iso(run["created_at"]), "locked": not entitled, "movers": []}
    if entitled:
        out["movers"] = _movers(session_movers(run_df(int(run["id"])), "AHPctChange", "AHLast"))
    return out


_SETUP_FIELDS = ("ticker", "score", "primary_setup", "status", "n_signals")
_SETUP_NUMBERS = ("last", "chg_pct", "gap_pct", "rvol", "prob")


def top_setups(entitlements: Dict[str, bool]) -> Dict[str, Any]:
    from ui.entitlement_view import redact_prebreakout_rows
    from ui.today import today_top_setups

    runs = market_runs()
    if not runs:
        return {"state": "empty_scan", "threshold": None, "scan_at": None, "setups": []}
    result = today_top_setups(run_df(int(runs[0]["id"])), n=5)
    rows = redact_prebreakout_rows(result["setups"], allowed=bool(entitlements.get("can_early_breakout")))
    setups = [{**{k: r.get(k) for k in _SETUP_FIELDS}, **{k: _num(r.get(k)) for k in _SETUP_NUMBERS}}
              for r in rows]
    return {"state": result["state"], "threshold": result["threshold"],
            "scan_at": _iso(runs[0]["created_at"]), "setups": setups}


def recap(now: dt.datetime) -> Optional[Dict[str, Any]]:
    from ui.market_scans import runs_on_day
    from ui.recap import build_recap, recap_day, session_scan_counts

    runs = market_runs()
    day = recap_day(runs, now)
    if day is None:
        return None
    day_runs = runs_on_day(runs, day)
    r = build_recap(day_runs, run_df(int(day_runs[-1]["id"])), run_df(int(day_runs[0]["id"])),
                    day=day, now=now, session_counts=session_scan_counts(session_runs(), day))
    return {**r, "day": r["day"].isoformat()}


def build_today(now: dt.datetime, entitlements: Dict[str, bool]) -> Dict[str, Any]:
    """Each section fails on its own: one bad read never blanks the whole page."""
    pro = bool(entitlements.get("can_day_trader"))
    sections: Dict[str, Callable[[], Any]] = {
        "before_open": lambda: before_open(now, pro),
        "top_setups": lambda: top_setups(entitlements),
        "after_close": lambda: after_close(now, pro),
        "recap": lambda: recap(now),
    }
    out: Dict[str, Any] = {"as_of": now.isoformat(), "market": {"phase": market_phase(now)}, "errors": []}
    for name, fn in sections.items():
        try:
            out[name] = fn()
        except Exception as e:  # report the section, never the details
            out[name] = None
            out["errors"].append({"section": name, "error": type(e).__name__})
    return out
