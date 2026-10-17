"""How my stocks did: each watchlist ticker's move since it was added, next to SPY.

The starting price is the one saved when the ticker was added, when there is one;
otherwise the last daily close at the moment it was added (that day's close if it
was added after 4 PM New York time, else the previous session's). The latest
price is the latest daily close (today's bar while the session trades). SPY over
the same span is the benchmark. One provider batch per distinct ticker set,
shared for PERF_TTL_S; a ticker the provider has no bars for is listed with
null numbers rather than an estimate.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Sequence, Tuple

from api.today import TTLCache, _iso, _num

PERF_TTL_S = 1800
PERF_MAX_TICKERS = 200
BENCHMARK = "SPY"
_NY_CLOSE_HOUR = 16
perf_cache = TTLCache(max_entries=64)


def _closes(tickers: Sequence[str]) -> Dict[str, List[Tuple[str, float]]]:
    """{ticker: [(ISO day, close), ...] oldest first} for about a year."""
    syms = tuple(sorted({str(t).strip().upper() for t in tickers if str(t).strip()}))
    if not syms:
        return {}

    def load() -> Dict[str, List[Tuple[str, float]]]:
        from data.prices import fetch_price_data_batch

        frames, _skipped = fetch_price_data_batch(list(syms), period="1y", timeout_s=10.0, batch_retries=1)
        out: Dict[str, List[Tuple[str, float]]] = {}
        for t in syms:
            df = frames.get(t)
            if df is None or getattr(df, "empty", True):
                continue
            col = next((c for c in df.columns if str(c).lower() == "close"), None)
            if col is None:
                continue
            series = []
            for idx, v in zip(df.index, df[col].tolist()):
                n = _num(v)
                if n is None or n <= 0:
                    continue
                day = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
                series.append((day, round(n, 4)))
            if series:
                out[t] = series
        return out

    return perf_cache.get(("closes", syms), load, ttl_s=PERF_TTL_S, stale_s=6 * 3600)


def _added_day(added_at: Any) -> Optional[Tuple[str, bool]]:
    """(New York date the ticker was added, True when added after the close)."""
    if added_at is None:
        return None
    if isinstance(added_at, dt.datetime):
        ts = added_at
    else:
        try:
            ts = dt.datetime.fromisoformat(str(added_at).replace("Z", "+00:00"))
        except ValueError:
            return None
    ts = ts if ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        ny = ts.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        ny = ts
    return ny.date().isoformat(), ny.hour >= _NY_CLOSE_HOUR


def _base_close(series: List[Tuple[str, float]], day: str, after_close: bool) -> Tuple[Optional[str], Optional[float], bool]:
    """(bar day, close, capped). The last close known when the ticker was added;
    capped=True when the add is older than the history, so the first bar is used."""
    pick = None
    for d, c in series:
        if d < day or (d == day and after_close):
            pick = (d, c)
        else:
            break
    if pick is None:
        return series[0][0], series[0][1], True
    return pick[0], pick[1], False


def _change(base: Optional[float], last: Optional[float]) -> Optional[float]:
    if not base or last is None:
        return None
    return round((last - base) / base * 100.0, 2)


def _close_on(series: List[Tuple[str, float]], day: str) -> Optional[float]:
    """The benchmark's close on `day`, or the last one before it."""
    best = None
    for d, c in series:
        if d <= day:
            best = c
        else:
            break
    return best


def performance(watchlist: Dict[str, Any]) -> Dict[str, Any]:
    items = list(watchlist.get("items") or [])[:PERF_MAX_TICKERS]
    tickers = [str(i.get("ticker") or "").strip().upper() for i in items if i.get("ticker")]
    out: Dict[str, Any] = {"watchlist_id": watchlist.get("id"), "name": watchlist.get("name"),
                           "benchmark": BENCHMARK, "as_of": None, "items": [], "summary": None}
    if not tickers:
        return out
    try:
        closes = _closes(tickers + [BENCHMARK])
    except Exception:
        closes = {}
    spy = closes.get(BENCHMARK) or []
    last_days = []
    rows = []
    for item in items:
        t = str(item.get("ticker") or "").strip().upper()
        if not t:
            continue
        series = closes.get(t) or []
        row: Dict[str, Any] = {"ticker": t, "added_at": _iso(item.get("added_at")), "base_date": None,
                               "base_price": None, "base_source": None, "since_first_bar": False,
                               "last": None, "last_date": None, "change_pct": None,
                               "benchmark_change_pct": None, "vs_benchmark": None, "sessions": None}
        added = _added_day(item.get("added_at"))
        if series and added:
            saved = _num(item.get("price_when_added"))
            bday, bclose, capped = _base_close(series, added[0], added[1])
            if saved and saved > 0:
                row.update(base_price=round(saved, 4), base_source="saved", base_date=added[0])
                bday, capped = added[0], False
            else:
                row.update(base_price=bclose, base_source="close", base_date=bday)
            last_day, last = series[-1]
            row.update(last=last, last_date=last_day, since_first_bar=capped,
                       change_pct=_change(row["base_price"], last),
                       sessions=sum(1 for d, _c in series if d > (row["base_date"] or "")))
            last_days.append(last_day)
            if spy:
                b0 = _close_on(spy, row["base_date"] or bday)
                row["benchmark_change_pct"] = _change(b0, spy[-1][1])
                if row["change_pct"] is not None and row["benchmark_change_pct"] is not None:
                    row["vs_benchmark"] = round(row["change_pct"] - row["benchmark_change_pct"], 2)
        rows.append(row)
    rows.sort(key=lambda r: (r["change_pct"] is None, -(r["change_pct"] or 0.0), r["ticker"]))
    out["items"] = rows
    out["as_of"] = max(last_days) if last_days else None
    priced = [r for r in rows if r["change_pct"] is not None]
    if priced:
        compared = [r for r in priced if r["vs_benchmark"] is not None]
        ups = sorted(r["change_pct"] for r in priced)
        mid = len(ups) // 2
        out["summary"] = {
            "count": len(priced),
            "up": sum(1 for r in priced if r["change_pct"] > 0),
            "down": sum(1 for r in priced if r["change_pct"] < 0),
            "avg_change_pct": round(sum(ups) / len(ups), 2),
            "median_change_pct": round(ups[mid] if len(ups) % 2 else (ups[mid - 1] + ups[mid]) / 2, 2),
            "beat_benchmark": sum(1 for r in compared if r["vs_benchmark"] > 0),
            "compared": len(compared),
            "best": {"ticker": priced[0]["ticker"], "change_pct": priced[0]["change_pct"]},
            "worst": {"ticker": priced[-1]["ticker"], "change_pct": priced[-1]["change_pct"]},
        }
    return out
