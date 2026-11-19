"""The public track record (/results on the web): every HSF top pick with how it did
over the next 5 trading days against the S&P 500 (SPY), losers included.

    Picks       each trading day's FIRST Scanner run, its top TOP_PER_DAY by HSF Score
                (score >= RANKED_MIN): the same rule as the weekly recap, frozen in
                signal_outcomes (source='opportunity', signal_type='hsf_scanner') when
                the scan ran, so nothing is re-picked with hindsight
    Delayed     a pick is shown once its 5-day window is over (it has matured), so the
                page never hands out today's setups
    Returns     close-to-close from the entry day's close, as stored by the outcome
                engine; SPY over the same window is the benchmark; no costs or slippage
    Verdict     a day-clustered bootstrap interval for the average excess return. Picks
                on the same day move together, so days, not picks, are resampled. The
                verdict only says "ahead" or "behind" when the whole interval is on one
                side of zero AND there are at least MIN_DAYS days; otherwise "no edge
                proven yet". The page states the verdict as computed, whatever it is.

Read-only for visitors; one query per CACHE_TTL_S.
"""
from __future__ import annotations

import datetime as dt
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from api.today import TTLCache

TOP_PER_DAY = 10
RANKED_MIN = 40      # same floor as the recap and "ranked" everywhere else
HORIZON = 5
MIN_DAYS = 20        # fewer entry days than this can't support a verdict either way
SHOW = 150
BOOTSTRAP = 2000
CACHE_TTL_S = 1800
_cache = TTLCache(8, name="public_record")

AHEAD, BEHIND, NO_EDGE = "ahead", "behind", "no_edge_yet"
VERDICT_TEXT = {
    AHEAD: "So far HSF's top picks have beaten the S&P 500 on average over 5 trading days, by more than chance alone would easily explain. It's still early, and past results don't guarantee future ones.",
    BEHIND: "So far HSF's top picks have trailed the S&P 500 on average over 5 trading days. We publish this anyway: you should see the losing stretches too.",
    NO_EDGE: "So far these results don't show a proven edge over the S&P 500: the difference is within what chance alone could explain. We publish every pick anyway, losers included, so you can judge for yourself.",
}


def _et():
    from analytics import market_calendar as mc

    return mc.ET


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _aware(t: Any) -> Optional[dt.datetime]:
    if not isinstance(t, dt.datetime):
        try:
            t = dt.datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        except ValueError:
            return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def daily_top(rows: Sequence[Dict[str, Any]], tz=None, *, top: int = TOP_PER_DAY) -> List[Dict[str, Any]]:
    """rows: dicts with ticker, fired_at, score, setup, return_5d, benchmark_5d, computed_at.
    The first run of each ET day, its top `top` by score (>= RANKED_MIN), one per ticker.
    Each pick gets day (ET), status matured/pending/unavailable and its excess return."""
    tz = tz or _et()
    clean = []
    for r in rows:
        fired = _aware(r.get("fired_at"))
        score = _num(r.get("score"))
        if fired is None or score is None:
            continue
        clean.append({**r, "fired_at": fired, "score": score})
    first: Dict[dt.date, dt.datetime] = {}
    for r in clean:
        d = r["fired_at"].astimezone(tz).date()
        if d not in first or r["fired_at"] < first[d]:
            first[d] = r["fired_at"]
    per_day: Dict[dt.date, List[Dict[str, Any]]] = {}
    for r in clean:
        d = r["fired_at"].astimezone(tz).date()
        if r["score"] >= RANKED_MIN and first.get(d) == r["fired_at"]:
            per_day.setdefault(d, []).append(r)
    out: List[Dict[str, Any]] = []
    for d in sorted(per_day):
        seen = set()
        for r in sorted(per_day[d], key=lambda x: -x["score"]):
            t = str(r.get("ticker") or "").strip().upper()
            if not t or t in seen:
                continue
            seen.add(t)
            if len(seen) > top:
                break
            computed = _aware(r.get("computed_at")) if r.get("computed_at") else None
            ret, bench = _num(r.get("return_5d")), _num(r.get("benchmark_5d"))
            if computed is None or computed <= r["fired_at"]:
                status = "pending"
            elif ret is None:
                status = "unavailable"
            else:
                status = "matured"
            ok = status == "matured"
            out.append({"ticker": t, "day": d.isoformat(), "score": round(r["score"]), "setup": r.get("setup"),
                        "status": status, "return_5d": ret if ok else None,
                        "spy_5d": bench if ok else None,
                        "excess_5d": (ret - bench) if ok and bench is not None else None})
    return out


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _median(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def day_bootstrap(picks: Sequence[Dict[str, Any]], *, n: int = BOOTSTRAP, seed: int = 7) -> Optional[Tuple[float, float]]:
    """95% interval for the average excess return, resampling whole entry days."""
    by_day: Dict[str, List[float]] = {}
    for p in picks:
        if p.get("excess_5d") is not None:
            by_day.setdefault(p["day"], []).append(p["excess_5d"])
    days = list(by_day.values())
    if len(days) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [days[rng.randrange(len(days))] for _ in days]
        flat = [x for day in sample for x in day]
        means.append(sum(flat) / len(flat))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def stats(picks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate numbers for matured picks (pending/unavailable only counted)."""
    mat = [p for p in picks if p["status"] == "matured"]
    rets = [p["return_5d"] for p in mat]
    ex = [p["excess_5d"] for p in mat if p["excess_5d"] is not None]
    spy = [p["spy_5d"] for p in mat if p["spy_5d"] is not None]
    ci = day_bootstrap(mat)
    days = len({p["day"] for p in mat})
    return {
        "picks": len(mat), "days": days,
        "pending": sum(1 for p in picks if p["status"] == "pending"),
        "unavailable": sum(1 for p in picks if p["status"] == "unavailable"),
        "first_day": min((p["day"] for p in mat), default=None),
        "last_day": max((p["day"] for p in mat), default=None),
        "avg_return": _mean(rets), "median_return": _median(rets),
        "avg_spy": _mean(spy), "avg_excess": _mean(ex), "median_excess": _median(ex),
        "beat_rate": (sum(1 for x in ex if x > 0) / len(ex)) if ex else None,
        "up_rate": (sum(1 for x in rets if x > 0) / len(rets)) if rets else None,
        "excess_ci95": {"low": ci[0], "high": ci[1]} if ci else None,
    }


def verdict(s: Dict[str, Any]) -> str:
    ci = s.get("excess_ci95")
    if not ci or s.get("days", 0) < MIN_DAYS:
        return NO_EDGE
    if ci["low"] > 0:
        return AHEAD
    if ci["high"] < 0:
        return BEHIND
    return NO_EDGE


def load_rows(start: Optional[dt.datetime] = None, end: Optional[dt.datetime] = None) -> List[Dict[str, Any]]:
    from db.engine import get_neon_conn

    conn = get_neon_conn()
    if conn is None:
        return []
    where, args = ["source = 'opportunity'", "signal_type = 'hsf_scanner'"], []
    if start is not None:
        where.append("fired_at >= %s")
        args.append(start)
    if end is not None:
        where.append("fired_at < %s")
        args.append(end)
    try:
        cur = conn.cursor()
        cur.execute("SELECT ticker, fired_at, setup_score AS score, indicators->>'primary_setup' AS setup, "
                    "return_5d, benchmark_return_5d AS benchmark_5d, outcome_computed_at AS computed_at "
                    f"FROM signal_outcomes WHERE {' AND '.join(where)}", tuple(args))  # nosec B608 - fixed clauses, values bound
        raw = cur.fetchall() or []
        cols = [d[0] for d in cur.description] if cur.description else []
        cur.close()
    finally:
        conn.close()
    return [dict(r) if isinstance(r, dict) else dict(zip(cols, r)) for r in raw]


def build(rows: Sequence[Dict[str, Any]], condition_of=None) -> Dict[str, Any]:
    picks = daily_top(rows)
    if condition_of is not None:
        for p in picks:
            p["condition"] = condition_of(p["day"])
    s = stats(picks)
    v = verdict(s)
    groups = []
    if condition_of is not None:
        from analytics import market_condition as mc

        for key in (mc.TRENDING_UP, mc.CHOPPY, mc.RISK_OFF):
            g = [p for p in picks if p.get("condition") == key]
            gs = stats(g)
            groups.append({"condition": key, "label": mc.LABELS[key], "picks": gs["picks"], "days": gs["days"],
                           "avg_return": gs["avg_return"], "avg_excess": gs["avg_excess"],
                           "median_excess": gs["median_excess"], "beat_rate": gs["beat_rate"]})
    mat = [p for p in picks if p["status"] == "matured"]
    mat.sort(key=lambda p: (p["day"], p["score"]), reverse=True)
    return {"horizon_days": HORIZON, "top_per_day": TOP_PER_DAY, "min_score": RANKED_MIN, "summary": s,
            "verdict": v, "verdict_text": VERDICT_TEXT[v], "min_days": MIN_DAYS, "by_condition": groups,
            "recent": mat[:SHOW], "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}


def public_record() -> Dict[str, Any]:
    def load() -> Dict[str, Any]:
        from api.market_condition import condition_of, labels_by_day

        labels = labels_by_day()
        return build(load_rows(), condition_of(labels) if labels else None)

    return _cache.get("record", load, ttl_s=CACHE_TTL_S, stale_s=6 * 3600)


def week_outcomes(monday: dt.date) -> Optional[Dict[str, Any]]:
    """How one week's daily top picks did over 5 trading days, for the weekly recap:
    matured picks only, best and worst three. None when nothing has matured."""
    tz = _et()
    start = dt.datetime.combine(monday, dt.time(0, 0), tz)
    end = dt.datetime.combine(monday + dt.timedelta(days=5), dt.time(0, 0), tz)
    picks = daily_top(load_rows(start, end))
    s = stats(picks)
    if not s["picks"]:
        return None
    mat = sorted((p for p in picks if p["status"] == "matured"), key=lambda p: p["return_5d"])
    trim = lambda p: {k: p[k] for k in ("ticker", "day", "score", "return_5d", "spy_5d", "excess_5d")}  # noqa: E731
    return {"start": monday.isoformat(), "end": (monday + dt.timedelta(days=4)).isoformat(),
            "picks": s["picks"], "pending": s["pending"], "avg_return": s["avg_return"], "avg_spy": s["avg_spy"],
            "avg_excess": s["avg_excess"], "beat_rate": s["beat_rate"], "up_rate": s["up_rate"],
            "best": [trim(p) for p in reversed(mat[-3:])], "worst": [trim(p) for p in mat[:3]]}
