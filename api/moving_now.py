"""Day Trader "Moving now": stocks spotted as an intraday move starts, with when and
where HSF first spotted them today.

Rule (the one tested read-only in scripts/diag_early_watch_sim.py over 16 sessions,
Oct 29-Nov 19 2026): a stock is spotted the first time today, from 9:40 AM ET in the
regular session, that it trades at least 3% above yesterday's close AND today's volume
so far is at least 2x its 20-day average daily volume scaled to the share of the
session gone. Same liquidity floor as the scheduled scans: price >= $1 and 20-day
average dollar volume >= $5M, both on the scan's data feed.

That test found the rule spots stocks earlier in the day than the scans (a median
2h45m on the scans' own pickups, but only ~0.6% cheaper) with better same-day follow-
through and no proven multi-day edge, so this is a Day Trader aid only: it never feeds
the Scanner or the HSF Score.

Data: the whole-market snapshot pass Day Trader's movers lists already share
(api.market.dt_universe_rows), plus one daily-bar lookup per ET day for names up 3%+.
First-spotted times live in shared_cache (Render Key Value when configured) under one
key per ET day, so a deploy or a second instance keeps them.
"""
from __future__ import annotations

import datetime as dt
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence

MIN_CHG_PCT = 3.0
MIN_VOLUME_PACE = 2.0
MIN_PRICE = 1.0
MIN_DOLLAR_VOL = 5_000_000.0
MAX_CHG_PCT = 100.0          # a 100%+ move is the Day Trader "check quote" flag, not a start
MAX_GAP_PCT = 40.0           # splits / bad prints, as the test excluded
FIRST_SPOT_ET = dt.time(9, 40)
SESSION_MIN = 390.0
LOOKBACK = 20
REFRESH_TTL_S = 120
SPOTTED_TTL_S = 36 * 3600
STATS_CHUNK = 200

RULE_TEXT = ("Spotted the first time today (from 9:40 AM ET) a stock trades 3%+ above yesterday's close "
             "on at least 2x its usual volume for the time of day.")

_stats_lock = threading.Lock()
_stats_day: Optional[dt.date] = None
_stats: Dict[str, Optional[tuple]] = {}


def _et(now: dt.datetime) -> dt.datetime:
    from analytics.market_calendar import ET

    return now.astimezone(ET)


def volume_pace(volume: Any, avg_volume: Any, now: dt.datetime) -> Optional[float]:
    """Today's volume vs the 20-day average daily volume scaled to the session gone."""
    try:
        v, avg = float(volume or 0), float(avg_volume or 0)
    except (TypeError, ValueError):
        return None
    if v <= 0 or avg <= 0:
        return None
    et = _et(now)
    elapsed = (et - et.replace(hour=9, minute=30, second=0, microsecond=0)).total_seconds() / 60.0
    elapsed = min(SESSION_MIN, max(5.0, elapsed))
    return v / (avg * elapsed / SESSION_MIN)


def spotting_open(now: dt.datetime, state: str) -> bool:
    """True while new stocks can be spotted: regular session, 9:40 AM ET onward."""
    et = _et(now)
    return state == "open" and FIRST_SPOT_ET <= et.time() < dt.time(16, 0)


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def candidates(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Snapshot rows that could meet the price side of the rule (up 3%+, sane quote)."""
    out = []
    for r in rows or []:
        chg, last, gap = _num(r.get("chg_pct")), _num(r.get("last")), _num(r.get("gap_pct"))
        prev = _num(r.get("previous_close"))
        if not r.get("ticker") or chg is None or last is None or prev is None:
            continue
        if chg < MIN_CHG_PCT or chg >= MAX_CHG_PCT or prev < MIN_PRICE:
            continue
        if gap is not None and abs(gap) > MAX_GAP_PCT:
            continue
        out.append(r)
    return out


def _fetch_stats(symbols: Sequence[str], day: dt.date) -> Dict[str, Optional[tuple]]:
    """{sym: (avg_volume, avg_dollar_volume)} over the LOOKBACK sessions before `day`; None
    when there's too little history. Daily bars in chunks, reduced as they arrive."""
    from data.price_alpaca import fetch_minute_bars_multi

    start = (day - dt.timedelta(days=LOOKBACK * 2 + 10)).isoformat()
    end = (day - dt.timedelta(days=1)).isoformat()
    out: Dict[str, Optional[tuple]] = {}
    syms = list(symbols)
    for i in range(0, len(syms), STATS_CHUNK):
        chunk = syms[i:i + STATS_CHUNK]
        got = fetch_minute_bars_multi(chunk, start, end, timeframe="1Day")
        for s in chunk:
            bars = [b for b in (got.get(s) or []) if str(b.get("t", ""))[:10] < day.isoformat()][-LOOKBACK:]
            vols = [(_num(b.get("v")) or 0.0, _num(b.get("c")) or 0.0) for b in bars]
            if len(vols) < LOOKBACK // 2:
                out[s] = None
                continue
            out[s] = (sum(v for v, _ in vols) / len(vols), sum(v * c for v, c in vols) / len(vols))
    return out


def daily_stats(symbols: Sequence[str], day: dt.date, fetch=None) -> Dict[str, Optional[tuple]]:
    """Per-day cache over _fetch_stats: each symbol is looked up at most once per ET day."""
    global _stats_day, _stats
    fetch = fetch or _fetch_stats
    with _stats_lock:
        if _stats_day != day:
            _stats_day, _stats = day, {}
        missing = [s for s in dict.fromkeys(symbols) if s not in _stats]
    if missing:
        got = fetch(missing, day)
        with _stats_lock:
            if _stats_day == day:
                for s in missing:
                    _stats[s] = got.get(s)
    with _stats_lock:
        return {s: _stats.get(s) for s in symbols}


def triggered(rows: Iterable[Dict[str, Any]], stats: Dict[str, Optional[tuple]], now: dt.datetime) -> List[Dict[str, Any]]:
    """Rows meeting the whole rule now, each with its volume pace."""
    out = []
    for r in candidates(rows):
        st = stats.get(r["ticker"])
        if not st or st[1] < MIN_DOLLAR_VOL:
            continue
        pace = volume_pace(r.get("volume"), st[0], now)
        if pace is not None and pace >= MIN_VOLUME_PACE:
            out.append({**r, "volume_pace": pace})
    return out


def _key(day: dt.date) -> str:
    return f"moving_now:{day.isoformat()}"


def spotted(day: dt.date) -> Dict[str, Dict[str, Any]]:
    from api import shared_cache

    got = shared_cache.get(_key(day))
    return got if isinstance(got, dict) else {}


def record(day: dt.date, hits: Sequence[Dict[str, Any]], now: dt.datetime) -> Dict[str, Dict[str, Any]]:
    """Add first sightings; a stock already spotted today keeps its first time and price."""
    from api import shared_cache

    seen = spotted(day)
    new = {h["ticker"]: {"at": now.isoformat(), "price": round(float(h["last"]), 4),
                         "chg_pct": _num(h.get("chg_pct")), "pace": round(float(h["volume_pace"]), 2)}
           for h in hits if h["ticker"] not in seen}
    if new:
        seen = {**seen, **new}
        shared_cache.put(_key(day), seen, SPOTTED_TTL_S)
    return seen


def refresh(now: Optional[dt.datetime] = None, *, rows=None, state: Optional[str] = None) -> bool:
    """One spotting pass (keep-warm and visitors both call it, at most every REFRESH_TTL_S
    via moving_now). Returns True when it looked at live data."""
    from ui import day_trader as dtm

    now = now or dt.datetime.now(dt.timezone.utc)
    state = state or dtm.market_state(now)
    if not spotting_open(now, state):
        return False
    if rows is None:
        from api.market import dt_universe_rows

        rows = dt_universe_rows()
    cands = candidates(rows)
    if not cands:
        return True
    day = _et(now).date()
    stats = daily_stats([r["ticker"] for r in cands], day)
    record(day, triggered(cands, stats, now), now)
    return True


def moving_now(limit: int = 10, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """The web's Moving now card: today's spotted stocks with live prices, strongest
    volume pace first."""
    import logging

    from api.today import _cached
    from ui import day_trader as dtm

    now = now or dt.datetime.now(dt.timezone.utc)
    state = dtm.market_state(now)
    day = _et(now).date()
    if spotting_open(now, state):
        try:
            _cached(("moving_now_pass", day.isoformat()), lambda: refresh(now, state=state) or True,
                    ttl_s=REFRESH_TTL_S, stale_s=REFRESH_TTL_S * 3)
        except Exception as e:  # spotting failing still shows what was spotted
            logging.getLogger("hsf_api.market").warning("moving now pass failed: %s", type(e).__name__)
    seen = spotted(day)
    live: Dict[str, Dict[str, Any]] = {}
    if seen and state in ("open", "afterhours"):
        try:
            from api.market import dt_universe_rows

            live = {r["ticker"]: r for r in (dt_universe_rows() or []) if r.get("ticker") in seen}
        except Exception:
            live = {}
    stats = dict(_stats) if _stats_day == day else {}
    rows = []
    for t, s in seen.items():
        r = live.get(t) or {}
        last = _num(r.get("last"))
        st = stats.get(t)
        pace = volume_pace(r.get("volume"), st[0], now) if (st and r) else None
        if pace is None:
            pace = _num(s.get("pace"))
        rows.append({"ticker": t, "spotted_at": s.get("at"), "spotted_price": _num(s.get("price")),
                     "spotted_chg_pct": _num(s.get("chg_pct")), "last": last, "chg_pct": _num(r.get("chg_pct")),
                     "since_spotted_pct": (round((last / float(s["price"]) - 1) * 100, 2)
                                           if last and _num(s.get("price")) else None),
                     "volume_pace": round(pace, 2) if pace is not None else None})
    rows.sort(key=lambda x: -(x["volume_pace"] or 0))
    return {"active": spotting_open(now, state), "state": state, "as_of": now, "session_date": day.isoformat(),
            "spotted_count": len(rows), "rule": RULE_TEXT, "rows": rows[:limit]}


def with_spotted(rows: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Optional[Dict[str, Any]]]:
    """{ticker: pickup} -> the same pickups plus spotted_at/spotted_price when Moving now
    spotted the stock on the ET day of that pickup. Display only: nothing here changes a
    scan, a rank or the HSF Score. One spotted-list read per ET day; failures add nothing."""
    by_day: Dict[str, Dict[str, Dict[str, Any]]] = {}
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for t, p in rows.items():
        if not p or not p.get("at"):
            out[t] = p
            continue
        try:
            day = _et(dt.datetime.fromisoformat(str(p["at"]).replace("Z", "+00:00"))).date()
            key = day.isoformat()
            if key not in by_day:
                by_day[key] = spotted(day)
            s = by_day[key].get(str(t).upper())
        except Exception:
            s = None
        out[t] = {**p, "spotted_at": s.get("at"), "spotted_price": _num(s.get("price"))} if s else p
    return out
