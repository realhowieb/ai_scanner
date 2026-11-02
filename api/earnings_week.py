"""Week at a glance: who reports earnings on each of the next five trading days.

Shared by the web's Earnings calendar (GET /v1/earnings/week) and the Sunday weekly
earnings email (services/weekly_earnings.py). Each day lists its biggest reporters
first, by dollar volume (last close x volume from one batch of Alpaca snapshots), so
a busy day of 300 names opens with the ones most people trade. Sub-$5 names are only
counted, never shown.

The base (calendar + sizes + scan scores) is built once and cached for CACHE_TTL_S;
plan rules (which scores a plan may see) are applied per caller in `for_plan`.
Read-only and best-effort: a provider failure leaves names unsized, never raises.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

DAYS = 5
CACHE_TTL_S = 15 * 60
MIN_PRICE = 5.0
DEFAULT_PER_DAY = 16
MAX_PER_DAY = 40
LOGO_URL = "https://financialmodelingprep.com/image-stock/{}.png"


def _key(ticker: Any) -> str:
    return "".join(ch for ch in str(ticker or "").upper() if ch.isalnum())


def _time(raw: Any) -> Optional[str]:
    t = str(raw or "").strip().lower()
    return t if t in ("bmo", "amc") else None


def logo_url(ticker: str) -> str:
    return LOGO_URL.format(str(ticker).upper().replace(".", "-"))


def week_days(today: dt.date, n: int = DAYS) -> List[dt.date]:
    """The next n trading days starting today (a weekend or holiday starts at the next one)."""
    from analytics import market_calendar as mc

    out, d = [], today
    while len(out) < n:
        if mc.is_trading_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _f(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None   # NaN -> None


def size_from_snapshot(snap: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """{last, chg_pct, dollar_volume} from one Alpaca snapshot. The size is the larger of
    the latest and previous daily bars (close x volume), so a quiet pre-market morning
    still ranks by yesterday's full session."""
    day, prev = snap.get("dailyBar") or {}, snap.get("prevDailyBar") or {}
    trade = snap.get("latestTrade") or {}
    last = _f(trade.get("p")) or _f(day.get("c")) or _f(prev.get("c"))
    pc = _f(prev.get("c"))
    chg = None
    if last is not None and pc:
        chg = (last - pc) / pc * 100.0
    sizes = [(_f(b.get("c")) or 0.0) * (_f(b.get("v")) or 0.0) for b in (day, prev)]
    dv = max(sizes) if any(sizes) else None
    return {"last": last, "chg_pct": chg, "dollar_volume": dv}


def _sizes(tickers: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    try:
        from market_data import fetch_alpaca_snapshots

        snaps = fetch_alpaca_snapshots(tickers) or {}
    except Exception:
        return {}
    return {_key(t): size_from_snapshot(s) for t, s in snaps.items() if isinstance(s, dict)}


def _scan_scores() -> Dict[str, Any]:
    """{at, ranks: {key: (rank, score)}} from the latest market scan; empty on error."""
    try:
        from api.scans import _iso, market_runs, run_opportunities

        runs = market_runs()
        if not runs:
            return {"at": None, "ranks": {}}
        ranks: Dict[str, Any] = {}
        for i, o in enumerate(run_opportunities(int(runs[0]["id"]))):
            k = _key(o.get("ticker"))
            if k and k not in ranks:
                ranks[k] = (i + 1, o.get("score"))
        return {"at": _iso(runs[0]["created_at"]), "ranks": ranks}
    except Exception:
        return {"at": None, "ranks": {}}


def build_base(today: dt.date, *, sizes=None, scores=None, calendar=None) -> Dict[str, Any]:
    """Every reporter on the next DAYS trading days with size, price and scan rank,
    biggest first within each day. Injectable loaders keep it testable offline."""
    days = week_days(today)
    start, end = days[0], days[-1]
    if calendar is None:
        try:
            from db.earnings import fetch_earnings_this_week

            calendar = fetch_earnings_this_week(days_ahead=(end - start).days, start=start) or []
        except Exception:
            calendar = []
    wanted = set(days)
    rows: List[Dict[str, Any]] = []
    seen = set()
    for r in calendar:
        d = r.get("earnings_date")
        if isinstance(d, dt.datetime):
            d = d.date()
        k = _key(r.get("symbol"))
        if not k or k in seen or d not in wanted:
            continue
        seen.add(k)
        rows.append({"ticker": str(r.get("symbol")).strip().upper(), "date": d, "time": _time(r.get("earnings_time"))})
    size_map = (sizes or _sizes)([r["ticker"] for r in rows]) if rows else {}
    sc = scores() if scores else _scan_scores()
    ranks = sc.get("ranks") or {}
    for r in rows:
        s = size_map.get(_key(r["ticker"])) or {}
        rank = ranks.get(_key(r["ticker"]))
        r.update(last=s.get("last"), chg_pct=s.get("chg_pct"), dollar_volume=s.get("dollar_volume"),
                 rank=rank[0] if rank else None, scan_score=rank[1] if rank else None)
    out_days = []
    for d in days:
        group = [r for r in rows if r["date"] == d]
        group.sort(key=lambda r: (-(r["dollar_volume"] or 0.0), r["ticker"]))
        out_days.append({"date": d, "rows": group})
    return {"start": start, "end": end, "scan_at": sc.get("at"), "days": out_days}


def cached_base(now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    from api.scans import _et_date
    from api.today import _cached

    today = _et_date(now)
    return _cached(("earnings_week", today.isoformat()), lambda: build_base(today), ttl_s=CACHE_TTL_S)


def _penny(r: Dict[str, Any]) -> bool:
    return r.get("last") is not None and r["last"] < MIN_PRICE


def for_plan(base: Dict[str, Any], *, per_day: int = DEFAULT_PER_DAY, cap: Optional[int] = None,
             auto: Optional[Dict[str, Dict[str, Any]]] = None, watched: Optional[set] = None,
             all_scores: bool = False) -> Dict[str, Any]:
    """The base cut to `per_day` names a day, with the scores this caller may see:
    scan scores within the plan's row cap (`cap`; None = all), auto-scores from `auto`
    (Premium, or every email reader with all_scores), and watchlist names always shown."""
    per_day = max(1, min(int(per_day), MAX_PER_DAY))
    watched_keys = {_key(t) for t in watched or ()}
    auto = auto or {}
    days = []
    for day in base["days"]:
        eligible = [r for r in day["rows"] if not _penny(r)]
        mine = [r for r in eligible if _key(r["ticker"]) in watched_keys]
        top = eligible[:per_day]
        shown = top + [r for r in mine if r not in top]
        names = []
        for r in shown:
            k = _key(r["ticker"])
            score, source = None, None
            if r.get("rank") is not None and (all_scores or cap is None or r["rank"] <= cap):
                score, source = r.get("scan_score"), "scan"
            elif k in auto and auto[k].get("score") is not None:
                score, source = auto[k].get("score"), "auto"
            names.append({"ticker": r["ticker"], "time": r["time"], "last": r.get("last"), "chg_pct": r.get("chg_pct"),
                          "dollar_volume": r.get("dollar_volume"), "score": score, "score_source": source,
                          "on_watchlist": k in watched_keys, "logo_url": logo_url(r["ticker"])})
        days.append({"date": day["date"].isoformat(), "total": len(day["rows"]), "shown": len(names), "names": names})
    return {"start": base["start"].isoformat(), "end": base["end"].isoformat(), "scan_at": base.get("scan_at"),
            "per_day": per_day, "days": days}
