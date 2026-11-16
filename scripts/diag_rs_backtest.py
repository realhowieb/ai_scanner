"""Read-only: does relative strength vs SPY separate HSF picks that went on to beat SPY?

Rebuilds real relative strength for past HSF picks (signal_outcomes source='opportunity')
from what each pick's own scan saved (Trend10D% / Trend20D%) and SPY's daily bars:

    rs_N = the stock's N-session return - SPY's return over the same sessions

The stock's return ends at its scan-time price. SPY's scan-time price isn't stored for
older scans, so SPY's return ends at that day's OPEN (known at scan time: only regular-
session picks, 09:30-16:00 ET, are used). That keeps the test free of look-ahead; the
cost is SPY's move between the open and the scan, usually well under 1%.

Window: development picks from 2026-09-14 through 2026-11-15 only. The ML v4 training
window (from 2026-11-16) and every reserved forward-test interval are never read.
One pick per stock per day (the first); days are the bootstrap clusters, since picks on
the same day share the market's move. Outcomes: matured excess return vs SPY over 1, 3
and 5 trading days, winsorized at the pooled 1st/99th percentiles.

Pre-declared rule (fixed before the data was read):
  primary  = 5-day excess return, picks beating SPY over 20 sessions (rs_20d > 0) vs not
  HELPS    = difference >= +0.50 points, its 95% day-cluster CI above 0, and the 3-day
             difference positive too
  HURTS    = 95% CI entirely below 0
  NO_EDGE  = anything else
Inside one day, ranking by rs_N is the same as ranking by Trend N% (SPY's return is the
same for every pick that day), and BreakoutScore already uses Trend10D%/Trend20D%. So what
RS can add over the current score is the across-day "beating the market or not" cut; the
within-day numbers are printed only to make that explicit.

Prints counts, rates and aggregates only: no tickers.
"""
from __future__ import annotations

import datetime as dt
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

START = dt.date(2026, 9, 14)       # after the Sep 12-13 bootstrap burst (#317)
END = dt.date(2026, 11, 15)        # inclusive; ML v4 training data starts 2026-11-16
HORIZONS = (1, 3, 5)
PRIMARY = 5
MEANINGFUL = 0.005                 # 0.50 percentage points (returns are fractions)
B = 2000
SEED = 20261116
ET = ZoneInfo("America/New_York")


def _num(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _et(iso: str) -> Optional[dt.datetime]:
    try:
        t = dt.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(ET)


def regular_session(t: dt.datetime) -> bool:
    return t.weekday() < 5 and dt.time(9, 30) <= t.time() < dt.time(16, 0)


def spy_table(bars) -> Dict[str, Any]:
    """{'dates': [date...], 'open': [...], 'close': [...]} from a daily-bar DataFrame."""
    df = bars.copy()
    df.columns = [str(c).capitalize() for c in df.columns]
    df = df.sort_index()
    dates = [(i.tz_convert(ET) if getattr(i, "tzinfo", None) else i).date() for i in df.index]
    return {"dates": dates, "open": [float(x) for x in df["Open"]], "close": [float(x) for x in df["Close"]]}


def rebuild_rs(trend_pct: Optional[float], day: dt.date, n: int, spy: Dict[str, Any]) -> Optional[float]:
    """rs_n in percentage points, or None when SPY lacks that day or n sessions before it."""
    if trend_pct is None:
        return None
    try:
        pos = spy["dates"].index(day)
    except ValueError:
        return None
    if pos - n < 0:
        return None
    base, now = spy["close"][pos - n], spy["open"][pos]
    if not base or not now:
        return None
    return round(trend_pct - (now / base - 1.0) * 100.0, 4)


def picks(records: Sequence[Dict[str, Any]], spy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One rebuilt row per (ticker, ET day), first observation kept; regular session only."""
    from analytics.research_dataset import MATURED

    seen, out = set(), []
    for r in records:
        obs, feats, outc = r["observation"], r["features"].values, r["outcome"]
        t = _et(obs.get("observed_at") or "")
        if t is None or not regular_session(t) or not (START <= t.date() <= END):
            continue
        key = (obs.get("ticker"), t.date())
        if key in seen:
            continue
        seen.add(key)
        row = {"day": t.date().isoformat(),
               "rs_10d": rebuild_rs(_num(feats.get("trend_10d_pct")), t.date(), 10, spy),
               "rs_20d": rebuild_rs(_num(feats.get("trend_20d_pct")), t.date(), 20, spy),
               "trend_20d": _num(feats.get("trend_20d_pct"))}
        for h in HORIZONS:
            ok = outc.maturity.get(f"{h}d") == MATURED
            row[f"x{h}"] = _num(outc.values.get(f"excess_return_{h}d")) if ok else None
        out.append(row)
    return out


def winsorize(rows: List[Dict[str, Any]], q: float = 0.01) -> None:
    for h in HORIZONS:
        vals = sorted(r[f"x{h}"] for r in rows if r[f"x{h}"] is not None)
        if len(vals) < 20:
            continue
        lo, hi = vals[int(q * (len(vals) - 1))], vals[int((1 - q) * (len(vals) - 1))]
        for r in rows:
            if r[f"x{h}"] is not None:
                r[f"x{h}"] = min(hi, max(lo, r[f"x{h}"]))


def _mean(v: Sequence[float]) -> Optional[float]:
    return sum(v) / len(v) if v else None


def _median(v: Sequence[float]) -> Optional[float]:
    s = sorted(v)
    if not s:
        return None
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def cluster_diff(rows: Sequence[Dict[str, Any]], is_a, label: str, *, b: int = B, seed: int = SEED) -> Dict[str, Any]:
    """Mean(label | a) - mean(label | not a), with a 95% bootstrap CI resampling days."""
    import random

    by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r[label] is not None and is_a(r) is not None:
            by_day[r["day"]].append(r)
    days = sorted(by_day)

    def diff(sample_days):
        a = [r[label] for d in sample_days for r in by_day[d] if is_a(r)]
        z = [r[label] for d in sample_days for r in by_day[d] if is_a(r) is False]
        return (_mean(a) - _mean(z)) if a and z else None, a, z

    point, a, z = diff(days)
    rng = random.Random(seed)
    boots = sorted(x for x in (diff([rng.choice(days) for _ in days])[0] for _ in range(b) if days) if x is not None)
    ci = [boots[int(0.025 * (len(boots) - 1))], boots[int(0.975 * (len(boots) - 1))]] if len(boots) >= 100 else [None, None]
    return {"n_a": len(a), "n_not": len(z), "days": len(days), "mean_a": _mean(a), "mean_not": _mean(z),
            "median_a": _median(a), "median_not": _median(z), "diff": point, "ci95": ci}


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 10:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                rk[order[k]] = (i + j) / 2.0
            i = j + 1
        return rk

    rx, ry = ranks(x), ranks(y)
    mx, my = _mean(rx), _mean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return sxy / (sx * sy) if sx and sy else None


def within_day_spearman(rows: Sequence[Dict[str, Any]], feat: str, label: str) -> Optional[float]:
    by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r[feat] is not None and r[label] is not None:
            by_day[r["day"]].append(r)
    vals = [s for s in (_spearman([r[feat] for r in g], [r[label] for r in g]) for g in by_day.values()) if s is not None]
    return _mean(vals)


def terciles(rows: Sequence[Dict[str, Any]], feat: str, label: str) -> Dict[str, Any]:
    xs = sorted((r[feat], r[label]) for r in rows if r[feat] is not None and r[label] is not None)
    if len(xs) < 30:
        return {"n": len(xs)}
    k = len(xs) // 3
    parts = {"bottom": xs[:k], "middle": xs[k:-k], "top": xs[-k:]}
    return {name: {"n": len(p), "rs_range": [round(p[0][0], 2), round(p[-1][0], 2)],
                   "mean_excess": _mean([y for _, y in p]), "median_excess": _median([y for _, y in p])}
            for name, p in parts.items()} | {"n": len(xs)}


def verdict(primary: Dict[str, Any], three_day: Dict[str, Any]) -> str:
    d, (lo, hi) = primary.get("diff"), primary.get("ci95") or [None, None]
    if d is None or lo is None:
        return "NOT_ENOUGH_DATA"
    if hi < 0:
        return "HURTS"
    if d >= MEANINGFUL and lo > 0 and (three_day.get("diff") or 0) > 0:
        return "HELPS"
    return "NO_EDGE"


def analyze(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    winsorize(rows)
    beat20 = lambda r: None if r["rs_20d"] is None else r["rs_20d"] > 0  # noqa: E731
    beat10 = lambda r: None if r["rs_10d"] is None else r["rs_10d"] > 0  # noqa: E731
    out: Dict[str, Any] = {"picks": len(rows), "with_rs_20d": sum(r["rs_20d"] is not None for r in rows),
                           "days": len({r["day"] for r in rows})}
    for h in HORIZONS:
        lab = f"x{h}"
        out[f"{h}d"] = {
            "beating_spy_20d_vs_not": cluster_diff(rows, beat20, lab),
            "beating_spy_10d_vs_not": cluster_diff(rows, beat10, lab),
            "pooled_spearman_rs_20d": _spearman(*zip(*[(r["rs_20d"], r[lab]) for r in rows
                                                       if r["rs_20d"] is not None and r[lab] is not None]))
            if sum(r["rs_20d"] is not None and r[lab] is not None for r in rows) >= 10 else None,
            "within_day_spearman_rs_20d": within_day_spearman(rows, "rs_20d", lab),
            "within_day_spearman_trend_20d": within_day_spearman(rows, "trend_20d", lab),
            "rs_20d_terciles": terciles(rows, "rs_20d", lab),
        }
    out["verdict"] = verdict(out[f"{PRIMARY}d"]["beating_spy_20d_vs_not"], out["3d"]["beating_spy_20d_vs_not"])
    return out


def _pp(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}pp"


def main() -> int:
    import json

    from analytics import research_dataset as rd
    from api.research import _bounds, load_window
    from data.price_alpaca import download_multi_alpaca
    from db.ml_forward_cohorts import development_ids

    lo, hi = _bounds(START, END)
    ids = development_ids(lo, hi)  # identities only; reserved intervals excluded
    window = load_window(START, END, ids=ids)
    records = rd.build_records(window["rows"], window["scans"], feature_schema_version=2)
    bars = download_multi_alpaca(["SPY"], period="200d", interval="1d", prepost=False, timeout_s=30)
    spy = spy_table(bars["SPY"])
    rows = picks(records, spy)
    report = analyze(rows)
    print(json.dumps(report, indent=1, default=str))
    p = report[f"{PRIMARY}d"]["beating_spy_20d_vs_not"]
    print(f"\nVERDICT {report['verdict']}: 5-day excess, beating SPY (20d) {_pp(p['mean_a'])} (n={p['n_a']}) "
          f"vs not {_pp(p['mean_not'])} (n={p['n_not']}); difference {_pp(p['diff'])}, "
          f"95% CI [{_pp(p['ci95'][0])}, {_pp(p['ci95'][1])}] over {p['days']} days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
