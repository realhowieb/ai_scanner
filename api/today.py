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
CACHE_MAX_ENTRIES = 32  # run lists + a handful of runs; old runs drop out
RUN_TTL_S = 6 * 3600  # a saved run's rows, re-validated every CACHE_TTL_S by its stamp
RUN_CACHE_MAX_ENTRIES = 8  # parsed scans are the biggest values; keep only the recent few


class TTLCache:
    """Scans change a few times a day; one read per minute is plenty. Expired
    entries are dropped on every write and the cache never exceeds max_entries,
    so a long-running instance doesn't grow.

    stale_s > 0 serves an expired value for up to stale_s more seconds while one
    background thread reloads it, so slow builders (the Brief, Day Trader movers)
    never make a visitor wait once warm. Concurrent misses on one key share a
    single load instead of each building it."""

    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self._items: Dict[Any, tuple[float, float, Any]] = {}  # key -> (expires at, stale until, value)
        self._lock = threading.Lock()
        self._loading: Dict[Any, threading.Lock] = {}  # key -> held while that key loads
        self._refreshing: set = set()

    def get(self, key: Any, loader: Callable[[], Any], ttl_s: float = CACHE_TTL_S, stale_s: float = 0) -> Any:
        from db.traffic import cache

        now = time.monotonic()
        with self._lock:
            hit = self._items.get(key)
            if hit and now < hit[0]:
                cache("api_results", hits=1)
                return hit[2]
            if hit and now < hit[1]:
                cache("api_results", stale_hits=1)
                if key not in self._refreshing:
                    self._refreshing.add(key)
                    threading.Thread(target=self._refresh, args=(key, loader, ttl_s, stale_s),
                                     name="cache-refresh", daemon=True).start()
                return hit[2]
            gate = self._loading.setdefault(key, threading.Lock())
        with gate:  # one load per key; later callers take its result
            with self._lock:
                hit = self._items.get(key)
                if hit and time.monotonic() < hit[0]:
                    cache("api_results", hits=1)
                    return hit[2]
            cache("api_results", misses=1)
            value = loader()
            self._put(key, value, ttl_s, stale_s)
        with self._lock:
            if self._loading.get(key) is gate and not gate.locked():
                del self._loading[key]
        return value

    def _refresh(self, key: Any, loader: Callable[[], Any], ttl_s: float, stale_s: float) -> None:
        try:
            from db.traffic import scope

            with scope("api.cache_refresh"):
                self._put(key, loader(), ttl_s, stale_s)
        except Exception:  # keep serving the stale value; the next request past stale_s loads in the foreground
            pass
        finally:
            with self._lock:
                self._refreshing.discard(key)

    def _put(self, key: Any, value: Any, ttl_s: float, stale_s: float) -> None:
        now = time.monotonic()
        with self._lock:
            for k in [k for k, (_, stale_until, _) in self._items.items() if now >= stale_until]:
                del self._items[k]
            self._items[key] = (now + ttl_s, now + ttl_s + max(0.0, stale_s), value)
            while len(self._items) > self.max_entries:
                del self._items[min(self._items, key=lambda k: self._items[k][1])]

    def clear_key(self, key: Any) -> None:
        with self._lock:
            self._items.pop(key, None)

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_cache = TTLCache(CACHE_MAX_ENTRIES)
_run_cache = TTLCache(RUN_CACHE_MAX_ENTRIES)


def _cached(key: Any, loader: Callable[[], Any], ttl_s: float = CACHE_TTL_S, stale_s: float = 0) -> Any:
    return _cache.get(key, loader, ttl_s, stale_s)


def _runs_or_outage(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """db.runs.list_runs answers [] when the database fails (it falls back to an
    empty local SQLite), which would read as "no scan yet". An empty list is
    only believed (and cached) when the database answers a ping; when it
    doesn't, DatabaseUnavailable propagates and nothing is cached."""
    if not runs:
        from api.store import ping

        ping()  # raises DatabaseUnavailable -> 503 / section error
    return runs


def cache_size() -> int:
    return _cache.size() + _run_cache.size()


def clear_cache() -> None:
    _cache.clear()
    _run_cache.clear()
    from api import scans  # stock pages have their own cache (api.scans)

    scans.stock_cache.clear()
    from api import outcomes  # Outcome Intelligence keeps its own cache too

    outcomes.clear_cache()


def _num(v: Any) -> Optional[float]:
    """JSON-safe number: NaN/inf and non-numbers become None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _iso(ts: Any) -> Optional[str]:
    """ISO 8601. Datetimes always carry a timezone: values from TIMESTAMP (no zone)
    columns are UTC (Neon runs in UTC), so a browser never reads them as local time."""
    if isinstance(ts, dt.datetime):
        return (ts if ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)).isoformat()
    return ts.isoformat() if isinstance(ts, dt.date) else None


def market_runs() -> List[Dict[str, Any]]:
    from db.runs import list_runs
    from ui.market_scans import MARKET_USER
    from ui.market_scans import market_runs as _market_runs

    return _cached("market_runs", lambda: _runs_or_outage(_market_runs(
        list_runs(limit=60, include_snapshots=True, username=MARKET_USER) or [])))


def session_runs() -> List[Dict[str, Any]]:
    from db.runs import list_runs

    return _cached("session_runs", lambda: _runs_or_outage(
        list_runs(limit=60, include_snapshots=False, username="scheduler") or []))


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

    # Saved runs are re-read by many endpoints; downloading results_json once a
    # minute was most of the Neon egress. Check the run's tiny stamp each minute
    # and keep the parsed rows until the stamp changes (an in-place snapshot rewrite).
    stamp = _cached(("run_stamp", int(run_id)), lambda: _run_stamp(int(run_id)))
    if stamp is None:
        return _cached(("run", int(run_id)), load)
    return _run_cache.get(("run", int(run_id), stamp), load, ttl_s=RUN_TTL_S)


def _run_stamp(run_id: int) -> Optional[str]:
    try:
        from db.runs import load_run_stamp

        return load_run_stamp(run_id)
    except Exception:
        return None


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


# A regular-session scan slot counts as missed once it is this late (System
# Health's SLOT_LATE); a scan may start this early and still count for it.
SCAN_SLOT_GRACE = dt.timedelta(minutes=45)
SCAN_SLOT_EARLY = dt.timedelta(minutes=10)


def _is_market_slot(slot: dt.datetime) -> bool:
    """Slots whose ET time falls in regular hours run the full-market scan
    (scheduler.cron_runner._resolve_session); the others run pre/post sessions."""
    et = slot.astimezone(mc.ET)
    return 9 * 60 + 30 <= et.hour * 60 + et.minute < 16 * 60


def last_due_market_scan(now: dt.datetime, days: int = 10) -> Optional[dt.datetime]:
    """The most recent full-market scan slot that should have produced a scan by now."""
    today = now.astimezone(mc.ET).date()
    for i in range(days):
        slots = [s for s in mc.expected_scan_slots(today - dt.timedelta(days=i))
                 if _is_market_slot(s) and s + SCAN_SLOT_GRACE <= now]
        if slots:
            return max(slots)
    return None


def scan_freshness(latest: Optional[dt.datetime], now: dt.datetime) -> Dict[str, Any]:
    """stale = a scheduled full-market scan was missed, so the latest scan is older
    than the schedule promises. Overnight, weekends and holidays are never stale
    on their own (the last scan of the session is current until the next one is due)."""
    due = last_due_market_scan(now)
    if latest is not None and latest.tzinfo is None:
        latest = latest.replace(tzinfo=dt.timezone.utc)
    stale = due is not None and (latest is None or latest < due - SCAN_SLOT_EARLY)
    return {"stale": stale, "expected_scan_at": _iso(due) if stale else None}


def market_status(now: dt.datetime) -> Dict[str, Any]:
    """Phase plus scan freshness for the Today page; freshness fails on its own."""
    out: Dict[str, Any] = {"phase": market_phase(now), "latest_scan_at": None, "stale": None,
                           "expected_scan_at": None}
    try:
        runs = market_runs()
        latest = runs[0]["created_at"] if runs else None
        out.update(latest_scan_at=_iso(latest), **scan_freshness(latest, now))
    except Exception:  # unknown freshness reads as null, never as stale
        pass
    return out


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


TOP_N = 5
_SETUP_FIELDS = ("ticker", "score", "primary_setup", "status", "n_signals")
_SETUP_NUMBERS = ("last", "chg_pct", "gap_pct", "rvol", "prob")


def top_setups(entitlements: Dict[str, bool]) -> Dict[str, Any]:
    from ui.entitlement_view import redact_prebreakout_rows
    from ui.market_scans import top_setups as ranked_setups
    from ui.recap import RECAP_MIN_SCORE
    from ui.today import today_top_setups

    runs = market_runs()
    if not runs:
        return {"state": "empty_scan", "threshold": None, "scan_at": None, "setups": []}
    df = run_df(int(runs[0]["id"]))
    result = today_top_setups(df, n=TOP_N)
    allowed = bool(entitlements.get("can_early_breakout"))

    def shape(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{**{k: r.get(k) for k in _SETUP_FIELDS}, **{k: _num(r.get(k)) for k in _SETUP_NUMBERS}}
                for r in redact_prebreakout_rows(rows, allowed=allowed)]

    # Fewer than TOP_N strong names: fill the card with the next ranked names (HSF 40+, the
    # recap's "ranked list" floor) so a quiet day doesn't read as an empty page.
    also: List[Dict[str, Any]] = []
    room = TOP_N - len(result["setups"])
    if room > 0 and result["state"] != "empty_scan":
        strong = {r.get("ticker") for r in result["setups"]}
        pool = ranked_setups(df, n=TOP_N * 2, minimum_score=RECAP_MIN_SCORE)
        also = [r for r in pool if r.get("ticker") not in strong][:room]
    return {"state": result["state"], "threshold": result["threshold"],
            "scan_at": _iso(runs[0]["created_at"]), "setups": shape(result["setups"]),
            "ranked_floor": RECAP_MIN_SCORE if also else None, "also_ranked": shape(also)}


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
    return {  # explicit shape (api.models.Recap), not build_recap's dict passed through
        "day": r["day"].isoformat(), "title": str(r["title"]), "scans": int(r["scans"]),
        "premarket_scans": int(r.get("premarket_scans") or 0),
        "postmarket_scans": int(r.get("postmarket_scans") or 0),
        "entered": [str(x) for x in r.get("entered") or []],
        "left": [str(x) for x in r.get("left") or []],
        "standouts": [{"ticker": str(o["ticker"]), "score": int(o["score"]), "setup": o.get("setup")}
                      for o in r.get("standouts") or []],
    }


# ---- market snapshot (the Streamlit trust banner + "Today's Market Snapshot") -------------------
SNAPSHOT_INDICES = (("SPY", "S&P 500"), ("QQQ", "Nasdaq 100"))
# The Streamlit price strip (ui.header.TICKER_STRIP) without VIX, which the stock
# quote provider doesn't carry. One provider call serves the strip and the snapshot.
TAPE_SYMBOLS = ("SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "TSLA")
QUOTE_TTL_S = 180   # the Streamlit strip caches quotes for 3 minutes too


def tape_quotes() -> List[Dict[str, Any]]:
    """Last price and change vs the previous close for TAPE_SYMBOLS, in that order, from
    the quote provider (Alpaca). Symbols without a price are left out; an empty answer
    (no provider keys, provider down) is not cached, so the next call retries."""
    def load():
        from market_data import get_latest_quotes

        quotes = get_latest_quotes(list(TAPE_SYMBOLS)) or {}
        out = []
        for sym in TAPE_SYMBOLS:
            q = quotes.get(sym)
            last = _num(q.get("last")) if isinstance(q, dict) else None
            prev = _num(q.get("prev_close")) if isinstance(q, dict) else None
            if last is not None:
                out.append({"symbol": sym, "last": last, "chg_pct": (last - prev) / prev * 100.0 if prev else None})
        return out

    try:
        out = _cached("tape_quotes", load, ttl_s=QUOTE_TTL_S)
    except Exception:
        return []
    if not out:
        _cache.clear_key("tape_quotes")
    return out


def _index_quotes() -> List[Dict[str, Any]]:
    by_symbol = {q["symbol"]: q for q in tape_quotes()}
    return [{**by_symbol[sym], "label": label} for sym, label in SNAPSHOT_INDICES if sym in by_symbol]


def _system_status(now: dt.datetime) -> Dict[str, Any]:
    """User-facing status and universe size from the latest health snapshot, read the
    same way as the Streamlit trust banner (ui.trust_banner.build_trust_info)."""
    from db.system_health import load_latest
    from ui.trust_banner import build_trust_info

    health = _cached("system_health", load_latest, ttl_s=QUOTE_TTL_S)
    info = build_trust_info([], health, now)
    return {"status": info["status"], "universe_symbols": info["universe_symbols"]}


def _scan_leader(df: Any, column: str) -> Optional[Dict[str, Any]]:
    """The row with the largest `column` value in a scan's results (one row per ticker)."""
    if df is None or column not in df.columns or "Ticker" not in df.columns:
        return None
    import pandas as pd

    work = df.assign(_v=pd.to_numeric(df[column], errors="coerce")).dropna(subset=["_v"])
    work = work[work["Ticker"].astype(str).str.strip() != ""]
    if work.empty:
        return None
    row = work.loc[work["_v"].idxmax()]
    chg = _num(row.get("PctChange")) if "PctChange" in work.columns else None
    return {"ticker": str(row["Ticker"]).strip().upper(), "chg_pct": chg,
            "last": _num(row.get("Last")) if "Last" in work.columns else None,
            "volume": _num(row.get("Volume")) if "Volume" in work.columns else None}


def snapshot(now: dt.datetime) -> Dict[str, Any]:
    """Universe, ranked count and system status for the status strip; SPY, QQQ, the
    scan's top gainer and most active name for the snapshot tiles. Each part degrades
    to null on its own."""
    out: Dict[str, Any] = {"universe_symbols": None, "ranked_count": None,
                           "status": {"level": "unknown", "label": "Status unavailable"},
                           "indices": _index_quotes(), "top_gainer": None, "most_active": None}
    try:
        out.update(_system_status(now))
    except Exception:
        pass
    runs = market_runs()
    if runs:
        rc = runs[0].get("row_count")
        out["ranked_count"] = int(rc) if isinstance(rc, (int, float)) and rc > 0 else None
        df = run_df(int(runs[0]["id"]))
        out["top_gainer"] = _scan_leader(df, "PctChange")
        out["most_active"] = _scan_leader(df, "Volume")
    return out


# ---- signed-in sections: new since your last visit, your watchlist ------------------------------
NEW_SHOWN_MAX = 50


def _run_scores(run_id: int) -> Dict[str, int]:
    """Ticker -> HSF Score for a saved market run (qualifying names only)."""
    def load():
        from ui.headline_score import hsf_scores_by_ticker

        df = run_df(run_id)
        return {} if df is None else hsf_scores_by_ticker(df.to_dict(orient="records"))

    return _cached(("scores", int(run_id)), load)


def new_since_visit(seen: Optional[int], baseline: Optional[int]) -> Dict[str, Any]:
    """The Streamlit "new since your last visit" rule (ui.last_visit) with the marker kept
    by the browser: it sends the run it last saw and its baseline, gets back the marker to
    store and the names in the latest market scan that weren't in the baseline scan. Only
    scheduled market runs count, so a run id can't read anyone's own scans."""
    from ui.last_visit import new_tickers, next_marker

    runs = market_runs()
    ids = {int(r["id"]) for r in runs}
    latest = int(runs[0]["id"]) if runs else None
    raw = f"{seen if seen in ids else ''}:{baseline if baseline in ids else ''}"
    base, marker = next_marker(raw if raw != ":" else None, latest)
    out: Dict[str, Any] = {"marker": marker or None, "baseline_scan_at": None, "tickers": [], "total": 0}
    if latest is None or base is None or base == latest or base not in ids:
        return out
    new = sorted(new_tickers(run_df(latest), run_df(base)))
    scores = _run_scores(latest)
    new.sort(key=lambda t: (-(scores.get(t) or -1), t))
    out.update(baseline_scan_at=_iso(next(r["created_at"] for r in runs if int(r["id"]) == base)),
               tickers=[{"ticker": t, "score": scores.get(t)} for t in new[:NEW_SHOWN_MAX]], total=len(new))
    return out


def watchlist_today(user: str) -> Dict[str, Any]:
    """The default watchlist against the latest market scan, like the Streamlit Today
    section (ui.today._section_watchlist): names in the scan with their HSF Score, names
    not in it, and the watchlist-intelligence counts."""
    from api import user_data

    lists = user_data.list_watchlists(user)
    wl = next((w for w in lists if w["is_default"]), lists[0] if lists else None)
    out: Dict[str, Any] = {"watchlist_id": None, "name": None, "summary": None, "in_scan": [], "missing": []}
    if wl is None:
        return out
    tickers = [str(i["ticker"]).strip().upper() for i in user_data.get_watchlist(user, wl["id"])["items"]]
    out.update(watchlist_id=wl["id"], name=wl["name"])
    try:
        from analytics.watchlist_intelligence import build_watchlist_intelligence

        summ = _cached(("wl_summary", user), lambda: dict((build_watchlist_intelligence(user) or {}).get("summary") or {}))
        if int(summ.get("tracked") or 0):
            out["summary"] = {k: int(summ.get(k) or 0) for k in ("tracked", "needs_attention", "strengthening", "fading")}
    except Exception:  # the counts are optional; the list still answers
        pass
    runs = market_runs()
    if not runs:
        out["missing"] = tickers
        return out
    from ui.market_scans import tickers_of

    in_scan = set(tickers_of(run_df(int(runs[0]["id"]))))
    scores = _run_scores(int(runs[0]["id"]))
    out["in_scan"] = sorted(({"ticker": t, "score": scores.get(t)} for t in tickers if t in in_scan),
                            key=lambda r: (-(r["score"] if r["score"] is not None else -1), r["ticker"]))
    out["missing"] = [t for t in tickers if t not in in_scan]
    return out


def build_personal(user: str, seen: Optional[int], baseline: Optional[int]) -> Dict[str, Any]:
    """Each section fails on its own, as in build_today."""
    sections: Dict[str, Callable[[], Any]] = {
        "new_since": lambda: new_since_visit(seen, baseline),
        "watchlist": lambda: watchlist_today(user),
    }
    out: Dict[str, Any] = {"errors": []}
    for name, fn in sections.items():
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = None
            out["errors"].append({"section": name, "error": type(e).__name__})
    return out


def build_today(now: dt.datetime, entitlements: Dict[str, bool]) -> Dict[str, Any]:
    """Each section fails on its own: one bad read never blanks the whole page."""
    pro = bool(entitlements.get("can_day_trader"))
    sections: Dict[str, Callable[[], Any]] = {
        "before_open": lambda: before_open(now, pro),
        "top_setups": lambda: top_setups(entitlements),
        "after_close": lambda: after_close(now, pro),
        "recap": lambda: recap(now),
        "snapshot": lambda: snapshot(now),
    }
    out: Dict[str, Any] = {"as_of": now.isoformat(), "market": market_status(now), "errors": []}
    for name, fn in sections.items():
        try:
            out[name] = fn()
        except Exception as e:  # report the section, never the details
            out[name] = None
            out["errors"].append({"section": name, "error": type(e).__name__})
    return out
