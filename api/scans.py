"""GET /v1/scans/latest and GET /v1/stocks/{ticker} (P1-59 step 5).

Built from the same saved market scans and the same helpers as the Streamlit
Scanner and Stock Intelligence pages, so the app and the web show one answer.
Premium model output is redacted below Premium, as on the web.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any, Dict, List, Optional

from api.today import RUN_TTL_S, TTLCache, _cached, _iso, _num, market_runs, run_df, run_freshness, scan_freshness

SCAN_FIELDS = ("ticker", "score", "primary_setup", "status", "n_signals")
SCAN_NUMBERS = ("last", "chg_pct", "gap_pct", "rvol", "breakout_score", "prob", "prob_rank")
CALIBRATION_TTL_S = 1800  # matured outcomes change once a day; same as the web
CALIBRATION_STALE_S = 6 * 3600  # past the TTL, served while one background rebuild runs
BAR_LIMIT = 260  # about a year of sessions, for the chart's 1Y range
EMA_WARMUP_BARS = 22  # ema_cross_detail needs 23 bars before it reports a cross
# Stock pages get their own cache so a client paging through many tickers can't
# push the scan runs out of the shared one.
stock_cache = TTLCache(max_entries=128)
# Stock pages are built from the latest scan (no live quote), so a page is reused for
# two minutes and, past that, served while one background rebuild runs.
STOCK_TTL_S = 120
STOCK_STALE_S = 1800


def json_safe(value: Any) -> Any:
    """NaN/inf -> None, datetimes -> ISO strings, recursively (strict JSON)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dt.datetime, dt.date)):
        return _iso(value)            # datetimes always with a timezone (UTC when stored without one)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):  # numpy scalars
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            return None
    return value


def run_opportunities(run_id: int) -> List[Dict[str, Any]]:
    """Ranked HSF opportunities of one saved run (one per ticker), cached."""
    def load():
        from ui.market_scans import top_setups

        df = run_df(run_id)
        return top_setups(df, n=100_000) if df is not None else []

    # Ranking a whole run is CPU work: past the minute it's served while one rebuild runs.
    return _cached(("opps", int(run_id)), load, stale_s=RUN_TTL_S)


def max_results_for(tier: str) -> int:
    """Rows a plan sees in the Scanner (config.TIERS_CONFIG max_results)."""
    from config import TIERS_CONFIG

    cfg = TIERS_CONFIG.get(tier) or TIERS_CONFIG["basic"]
    return int(cfg.get("max_results") or 25)


def _scan_row(o: Dict[str, Any]) -> Dict[str, Any]:
    return {**{k: o.get(k) for k in SCAN_FIELDS},
            "signals": [str(s) for s in o.get("signals") or []],
            "fading": bool(o.get("fading")),
            **{k: _num(o.get(k)) for k in SCAN_NUMBERS}}


def latest_scan(entitlements: Dict[str, bool], tier: str, *, limit: int, offset: int,
                min_score: int = 0, signal: Optional[str] = None, sort: str = "score") -> Dict[str, Any]:
    from analytics.data_freshness import describe
    from ui.entitlement_view import redact_prebreakout_rows

    cap = max_results_for(tier)
    runs = market_runs()
    if not runs:
        return {"scan_at": None, "total": 0, "scan_total": 0, "max_results": cap, "limited": False, "setups": [],
                'freshness': describe(None)}
    allowed = bool(entitlements.get("can_early_breakout"))
    opps = redact_prebreakout_rows(run_opportunities(int(runs[0]["id"])), allowed=allowed)
    scan_total = len(opps)
    if min_score:
        opps = [o for o in opps if int(o.get("score") or 0) >= int(min_score)]
    if signal == "prebreakout":
        # The model's strongest names this scan, kept in HSF order (see PREBREAKOUT_TOP_N).
        from ui.results_intelligence import prebreakout_picks

        picked = {id(o) for o in prebreakout_picks(opps)}
        opps = [o for o in opps if id(o) in picked]
    elif signal:
        opps = [o for o in opps if signal in (o.get("signals") or [])]
    visible = opps[:cap]
    if sort != "score":
        # Re-orders only the rows the plan already sees (its top HSF-ranked setups); missing values last.
        def key(o: Dict[str, Any]) -> Any:
            v = _num(o.get(sort))
            return (v is None, -(v or 0.0))
        visible = sorted(visible, key=key)
    return {"scan_at": _iso(runs[0]["created_at"]), "total": len(opps), "scan_total": scan_total,
            "max_results": cap, "limited": len(opps) > cap,
            "stale": scan_freshness(runs[0]["created_at"], dt.datetime.now(dt.timezone.utc))["stale"],
            'freshness': run_freshness(runs[0]),
            "setups": [_scan_row(o) for o in visible[offset:offset + limit]]}


# ---- Earnings scanner ---------------------------------------------------------------------------
# Day offsets from today's ET date, inclusive. Today is free; the others need can_earnings (Pro).
EARNINGS_WINDOWS = {"today": (0, 0), "tomorrow": (1, 1), "week": (0, 7)}
EARNINGS_TIMES = ("bmo", "amc")


def _ticker_key(ticker: Any) -> str:
    """BRK-B, BRK.B and BRKB match: the calendar stores symbols without separators."""
    return "".join(ch for ch in str(ticker or "").upper() if ch.isalnum())


def _et_date(now: Optional[dt.datetime] = None) -> dt.date:
    from zoneinfo import ZoneInfo

    return (now or dt.datetime.now(dt.timezone.utc)).astimezone(ZoneInfo("America/New_York")).date()


def _earnings_time(raw: Any) -> Optional[str]:
    t = str(raw or "").strip().lower()
    return t or None


def earnings_tickers(window: str, now: Optional[dt.datetime] = None) -> List[str]:
    """Every symbol in the earnings calendar that reports in `window` (ET days), soonest first."""
    from db.earnings import fetch_earnings_this_week

    lo, hi = EARNINGS_WINDOWS[window]
    today = _et_date(now)
    start, end = today + dt.timedelta(days=lo), today + dt.timedelta(days=hi)
    rows = []
    for r in fetch_earnings_this_week(days_ahead=hi - lo, start=start) or []:
        d = r.get("earnings_date")
        d = d.date() if isinstance(d, dt.datetime) else d
        sym = str(r.get("symbol") or "").strip().upper()
        if sym and isinstance(d, dt.date) and start <= d <= end:
            rows.append((d, sym))
    out: List[str] = []
    for _, sym in sorted(rows):
        if sym not in out:
            out.append(sym)
    return out


def earnings_scan(entitlements: Dict[str, bool], tier: str, *, window: str = "today",
                  time: Optional[str] = None, include_unranked: bool = False,
                  watched: Optional[set] = None, now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """Setups in the latest market scan that report earnings in `window` (ET days), joined with
    the earnings calendar. Soonest day first, then HSF Score. With include_unranked, names that
    report but did not rank in the scan follow each day's ranked names (no score)."""
    from analytics.data_freshness import describe
    from db.earnings import fetch_earnings_this_week
    from ui.entitlement_view import redact_prebreakout_rows

    cap = max_results_for(tier)
    lo, hi = EARNINGS_WINDOWS[window]
    today = _et_date(now)
    start, end = today + dt.timedelta(days=lo), today + dt.timedelta(days=hi)
    base = {"window": window, "start": start.isoformat(), "end": end.isoformat(), "max_results": cap}

    calendar: Dict[str, Dict[str, Any]] = {}
    for r in fetch_earnings_this_week(days_ahead=hi - lo, start=start) or []:
        d = r.get("earnings_date")
        if isinstance(d, dt.datetime):
            d = d.date()
        key = _ticker_key(r.get("symbol"))
        if not key or not isinstance(d, dt.date) or not (start <= d <= end):
            continue
        t = _earnings_time(r.get("earnings_time"))
        if time and t != time:
            continue
        calendar[key] = {"ticker": str(r.get("symbol")).strip().upper(), "earnings_date": d, "time": t}

    watched_keys = {_ticker_key(t) for t in watched or ()}

    def earnings_fields(e: Dict[str, Any], key: str) -> Dict[str, Any]:
        return {"earnings_date": e["earnings_date"].isoformat(), "days_until": (e["earnings_date"] - today).days,
                "time": e["time"], "on_watchlist": key in watched_keys}

    runs = market_runs()
    opps = (redact_prebreakout_rows(run_opportunities(int(runs[0]["id"])),
                                    allowed=bool(entitlements.get("can_early_breakout"))) if runs else [])
    ranked, seen = [], set()
    for i, o in enumerate(opps):
        key = _ticker_key(o.get("ticker"))
        if key in calendar and key not in seen:
            seen.add(key)
            ranked.append({**_scan_row(o), "rank": i + 1, "ranked": True, **earnings_fields(calendar[key], key)})
    unranked = [{"ticker": e["ticker"], "score": None, "rank": None, "ranked": False, "signals": [],
                 **earnings_fields(e, key)} for key, e in calendar.items() if key not in seen]

    def order(row: Dict[str, Any]) -> Any:
        return (row["earnings_date"], not row["ranked"], -(row.get("score") or 0), row["ticker"])

    rows = sorted(ranked[:cap], key=order)
    if include_unranked:
        rows = sorted(rows + sorted(unranked, key=order)[:cap], key=order)
    out = {**base, "total": len(ranked), "unranked_total": len(unranked), "limited": len(ranked) > cap,
           "setups": rows}
    if not runs:
        return {**out, "scan_at": None, "stale": None, "freshness": describe(None)}
    return {**out, "scan_at": _iso(runs[0]["created_at"]),
            "stale": scan_freshness(runs[0]["created_at"], dt.datetime.now(dt.timezone.utc))["stale"],
            "freshness": run_freshness(runs[0])}


MOVERS_MAX = 10


def score_movers(tier: str, *, limit: int = 5) -> Dict[str, Any]:
    """The setups whose HSF Score moved most between the two latest full-market scans.

    Read-only over the same observation the watchlist and alert-rule code use (no
    provider calls). Only tickers ranked in both scans count, and only within the
    plan's row cap (risers by their rank now, fallers by their rank before), so a
    plan never sees setups below its Scanner cap. Never raises: no data reads as empty."""
    from api import watchlist_intel

    out: Dict[str, Any] = {"scan_at": None, "previous_scan_at": None, "rising": [], "falling": [],
                           "max_results": max_results_for(tier)}
    try:
        obs = watchlist_intel.market_observation()
    except Exception:
        return out
    if not obs or obs.get("previous_run_id") is None:
        if obs:
            out["scan_at"] = _iso(obs.get("scan_at"))
        return out
    cap = out["max_results"]
    rows = []
    for t, cur in obs["ranked"].items():
        prev = obs["previous_ranked"].get(t)
        now_s, was_s = _num(cur.get("score")), _num((prev or {}).get("score"))
        if prev is None or now_s is None or was_s is None or now_s == was_s:
            continue
        rows.append({"ticker": t, "score": int(round(now_s)), "previous_score": int(round(was_s)),
                     "change": int(round(now_s - was_s)), "rank": int(cur["rank"]),
                     "previous_rank": int(prev["rank"]), "setup": cur.get("primary_setup"),
                     "last": _num(cur.get("last")), "chg_pct": _num(cur.get("chg_pct"))})
    n = max(1, min(int(limit), MOVERS_MAX))
    rising = sorted((r for r in rows if r["change"] > 0 and r["rank"] <= cap),
                    key=lambda r: (-r["change"], r["rank"]))[:n]
    falling = sorted((r for r in rows if r["change"] < 0 and r["previous_rank"] <= cap),
                     key=lambda r: (r["change"], r["previous_rank"]))[:n]
    out.update(scan_at=_iso(obs.get("scan_at")), previous_scan_at=_iso(obs.get("previous_scan_at")),
               rising=rising, falling=falling)
    return out


def watchlist_scan_state(tickers: List[str], entitlements: Dict[str, bool]) -> Dict[str, Any]:
    """Each ticker's row in the latest market scan's ranked setups (None when it isn't one),
    with its rank in that scan. The same rows as /v1/scans/latest, without the plan's row
    cap: these are the user's own tickers, as on the stock page. Never raises; an
    unavailable scan reads as no state."""
    from ui.entitlement_view import redact_prebreakout_rows

    try:
        runs = market_runs()
        if not runs:
            return {"scan_at": None, "stale": None, "rows": {}}
        allowed = bool(entitlements.get("can_early_breakout"))
        opps = redact_prebreakout_rows(run_opportunities(int(runs[0]["id"])), allowed=allowed)
        wanted = {str(t).strip().upper() for t in tickers}
        rows: Dict[str, Dict[str, Any]] = {}
        for i, o in enumerate(opps):
            t = str(o.get("ticker") or "").strip().upper()
            if t in wanted and t not in rows:
                rows[t] = {**_scan_row(o), "rank": i + 1}
        return {"scan_at": _iso(runs[0]["created_at"]), "total": len(opps),
                "stale": scan_freshness(runs[0]["created_at"], dt.datetime.now(dt.timezone.utc))["stale"],
                "rows": rows}
    except Exception:
        return {"scan_at": None, "stale": None, "rows": {}}


def _calibration_records() -> List[Dict[str, Any]]:
    def load():
        from analytics.hsf_calibration import build_calibration_dataset

        return build_calibration_dataset(days_back=180).get("records") or []

    return _cached("calibration", load, ttl_s=CALIBRATION_TTL_S, stale_s=CALIBRATION_STALE_S)


def warm_stock_pages() -> None:
    """Load what every Stock Intelligence page shares (the latest scan, its ranked
    opportunities and the calibration records) so the first stock visited after a
    restart doesn't pay for them."""
    runs = market_runs()
    if runs:
        run_opportunities(int(runs[0]["id"]))
    _calibration_records()


def _ticker_rows(df: Any, ticker: str) -> List[Dict[str, Any]]:
    if df is None or "Ticker" not in getattr(df, "columns", []):
        return []
    sub = df[df["Ticker"].astype(str).str.strip().str.upper() == ticker]
    return sub.to_dict(orient="records")


BARS_TTL_S = 900  # a provider top-up is reused for 15 minutes per ticker
BARS_STALE_S = 6 * 3600  # past that, served while one background refetch runs


def _last_session_day(now: Optional[dt.datetime] = None) -> dt.date:
    """The latest weekday whose regular session has closed, in ET (holidays are not
    known here, so on a holiday the previous day's bars simply count as current)."""
    from zoneinfo import ZoneInfo

    et = (now or dt.datetime.now(dt.timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    day = et.date() if et.hour >= 16 else et.date() - dt.timedelta(days=1)
    while day.weekday() >= 5:
        day -= dt.timedelta(days=1)
    return day


def _frame_bars(df: Any, limit: int, through: dt.date) -> List[Dict[str, Any]]:
    """Completed daily bars (through `through`) from an OHLCV frame, oldest first."""
    if df is None or getattr(df, "empty", True):
        return []
    cols = {str(c).lower().replace("adj close", "adj_close"): c for c in df.columns}
    pick = {k: cols.get(k) for k in ("open", "high", "low", "close", "volume")}
    if pick["close"] is None:
        return []
    bars = []
    for idx, row in df.iterrows():
        day = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
        if day > through.isoformat():  # today's session is still trading
            continue
        bar = {"date": day, **{k: _num(row[c]) if c is not None else None for k, c in pick.items()}}
        if bar["close"] is not None:
            bars.append(bar)
    _add_emas(bars)
    return bars[-limit:]


def _add_emas(bars: List[Dict[str, Any]]) -> None:
    """EMA 9/21 per bar, the same lines the scanner's golden cross uses
    (scan.indicators.ema_cross_detail: ewm adjust=False over the whole history),
    so the chart's crossovers match the signal. Left None during the warm-up."""
    fast = slow = None
    for i, b in enumerate(bars):
        c = b["close"]
        fast = c if fast is None else fast + (c - fast) * 2 / 10
        slow = c if slow is None else slow + (c - slow) * 2 / 22
        warm = i >= EMA_WARMUP_BARS
        b["ema9"] = round(fast, 4) if warm else None
        b["ema21"] = round(slow, 4) if warm else None
    for n in (20, 50):
        total = 0.0
        for i, b in enumerate(bars):
            total += b["close"]
            if i >= n:
                total -= bars[i - n]["close"]
            b[f"sma{n}"] = round(total / n, 4) if i >= n - 1 else None


def _provider_bars(ticker: str, limit: int, through: dt.date) -> List[Dict[str, Any]]:
    from data.prices import fetch_price_data_batch

    # Two years, so the moving averages are warmed up across the whole 1Y view.
    frames, _skipped = fetch_price_data_batch([ticker], period="2y", timeout_s=8.0, batch_retries=1)
    return _frame_bars(frames.get(ticker), limit, through)


def daily_bars(ticker: str, limit: int = BAR_LIMIT) -> Dict[str, Any]:
    """Daily OHLCV for the Stock Intelligence chart.

    Starts from the frames the scans cached. Scheduled scans don't write that cache
    (only admin full-universe scans do) and its frames are short, so when its last bar
    is older than the last completed session, or it holds fewer than `limit` bars, the
    ticker's bars are fetched once from the price provider and reused for BARS_TTL_S.
    The cached bars stand if that fetch fails."""
    from db.prices import get_price_data_snapshot

    through = _last_session_day()
    frames, _stale = get_price_data_snapshot([ticker], max_age_minutes=7 * 24 * 60)
    bars = _frame_bars(frames.get(ticker), limit, through)
    # Scan frames hold only a few months; a year of history comes from the provider.
    if len(bars) < limit or bars[-1]["date"] < through.isoformat():
        try:
            fresh = stock_cache.get(("bars", ticker, through.isoformat(), limit),
                                    lambda: _provider_bars(ticker, limit, through),
                                    ttl_s=BARS_TTL_S, stale_s=BARS_STALE_S)
        except Exception:
            fresh = []
        if fresh and (not bars or fresh[-1]["date"] > bars[-1]["date"]
                      or (fresh[-1]["date"] == bars[-1]["date"] and len(fresh) > len(bars))):
            bars = fresh
    return {"bars": bars, "as_of": bars[-1]["date"] if bars else None}


def _stock_core(ticker: str) -> Dict[str, Any]:
    """The Stock Intelligence object for `ticker` before redaction (cached per ticker)."""
    def load():
        from db.signal_outcomes import fetch_ticker_opportunity_history
        from scheduler.morning_digest import _earnings_days_map
        from ui.stock_intelligence import build_stock_intelligence

        runs = market_runs()
        run = runs[0] if runs else None
        current_opp, rows = None, []
        if run is not None:
            current_opp = next((o for o in run_opportunities(int(run["id"])) if o["ticker"] == ticker), None)
            rows = _ticker_rows(run_df(int(run["id"])), ticker)
        try:
            history = list(fetch_ticker_opportunity_history(ticker) or [])
        except Exception:
            history = []
        try:
            calibration = _calibration_records()
        except Exception:
            calibration = None
        intel = build_stock_intelligence(
            ticker, current_opp=current_opp, current_row=rows[0] if rows and current_opp is None else None,
            history=history, regime=None, calibration_records=calibration,
            earnings_days=_earnings_days_map([ticker]).get(ticker))
        quote = {"last": None, "chg_pct": None}
        for r in rows:  # price even when the name doesn't qualify as a setup
            quote["last"] = quote["last"] if quote["last"] is not None else _num(r.get("Last"))
            quote["chg_pct"] = quote["chg_pct"] if quote["chg_pct"] is not None else _num(r.get("PctChange"))
        return {"intel": intel, "scan_at": _iso(run["created_at"]) if run else None,
                "in_latest_scan": bool(rows), "rvol": _num((current_opp or {}).get("rvol")), "quote": quote}

    return stock_cache.get(ticker, load, ttl_s=STOCK_TTL_S, stale_s=STOCK_STALE_S)


def stock_detail(ticker: str, entitlements: Dict[str, bool]) -> Dict[str, Any]:
    from ui.entitlement_view import redact_prebreakout_opportunity

    core = _stock_core(ticker)
    intel = redact_prebreakout_opportunity(core["intel"], allowed=bool(entitlements.get("can_early_breakout")))
    movement = intel.get("movement") or {}
    out = {
        "ticker": ticker,
        "rvol": core.get("rvol"),
        "scan_at": core["scan_at"],
        "in_latest_scan": core["in_latest_scan"],
        "has_setup": bool(intel.get("has_opportunity")),
        "from_history": bool(intel.get("from_history")),
        "price": _num(intel.get("price")) if intel.get("price") is not None else core["quote"]["last"],
        "change_pct": (_num(intel.get("change_pct")) if intel.get("change_pct") is not None
                       else core["quote"]["chg_pct"]),
        "hsf_score": intel.get("hsf_score"),
        "status": intel.get("status"),
        "primary_setup": intel.get("primary_setup"),
        "signals": [str(s) for s in intel.get("signals") or []],
        "score_components": intel.get("score_components"),
        "movement": movement.get("movement_state"),
        "score_change": movement.get("score_delta"),
        "reasons": [str(r) for r in intel.get("reasons") or []],
        "risks": [str(r) for r in intel.get("risks") or []],
        "watch_next": [str(r) for r in intel.get("watch_next") or []],
        "breakout_score": _num((intel.get("model") or {}).get("breakout_score")),
        "prob": _num((intel.get("model") or {}).get("prebreakout_prob")),
        "earnings_days": intel.get("earnings_days"),
        "history_summary": intel.get("history_summary"),
        "historical_context": intel.get("historical_context"),
        "outcome_cohort": intel.get("outcome_cohort"),
        "lifecycle": intel.get("lifecycle") or [],
    }
    # Historical research is Pro (can_track_record), as on the web's stock page (Run 85E).
    out["historical_locked"] = not bool(entitlements.get("can_track_record"))
    if out["historical_locked"]:
        out.update({"history_summary": None, "historical_context": None, "outcome_cohort": None})
    return json_safe(out)


# ---- sparklines and score history (Scanner, Watchlists, Today, Stock page) ----------------------
SPARK_SESSIONS = 21  # 20 sessions of change: the first close is the baseline
SPARK_MAX = 50
SPARK_TTL_S = 900
spark_cache = TTLCache(max_entries=64)


def daily_closes(tickers: List[str]) -> Dict[str, Any]:
    """The last SPARK_SESSIONS daily closes per ticker (the session in progress included),
    for row sparklines. One provider batch per distinct ticker set, shared by every visitor
    for SPARK_TTL_S; a ticker the provider has no bars for is left out."""
    syms = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})[:SPARK_MAX]
    if not syms:
        return {"series": {}}

    def load() -> Dict[str, Any]:
        from data.prices import fetch_price_data_batch

        frames, _skipped = fetch_price_data_batch(syms, period="2mo", timeout_s=8.0, batch_retries=1)
        out: Dict[str, List[float]] = {}
        last_bar: Dict[str, Any] = {}
        for t in syms:
            df = frames.get(t)
            if df is None or getattr(df, "empty", True):
                continue
            col = next((c for c in df.columns if str(c).lower() == "close"), None)
            if col is None:
                continue
            valid = [(stamp, value) for stamp, raw in df[col].items() if (value := _num(raw)) is not None]
            if len(valid) >= 2:
                out[t] = [round(value, 4) for _, value in valid[-SPARK_SESSIONS:]]
            if valid:
                stamp, value = valid[-1]
                # A numeric index is not a session date. Never invent provenance.
                if isinstance(stamp, (dt.datetime, dt.date)):
                    last_bar[t] = {"close": round(value, 4), "session": stamp.strftime("%Y-%m-%d")}
        return {"series": out, "last_bar": last_bar}

    try:
        return spark_cache.get(("dated-closes", tuple(syms)), load, ttl_s=SPARK_TTL_S, stale_s=6 * 3600)
    except Exception:
        return {"series": {}, "last_bar": {}}


QUOTE_MAX = 50
QUOTE_TTL_S = 60
quote_cache = TTLCache(max_entries=64)


def _trade_day(stamp: Any) -> Optional[str]:
    """The New York calendar day of a provider timestamp (UTC ISO), or None."""
    if not isinstance(stamp, str) or not stamp:
        return None
    try:
        from zoneinfo import ZoneInfo

        when = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        return when.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    except ValueError:
        return None


def latest_quotes(tickers: List[str]) -> Dict[str, Any]:
    """Latest trade per ticker from the quote provider (Alpaca snapshots), for the Watchlists
    Refresh. Change is against the close before the trade's session, so a pre-market trade
    is compared with the last regular close. One provider call per distinct ticker set,
    shared for QUOTE_TTL_S; an empty answer (no keys, provider down) is not cached."""
    syms = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})[:QUOTE_MAX]
    if not syms:
        return {"quotes": {}, "feed": None}

    def load() -> Dict[str, Any]:
        from market_data import _get_alpaca_data_feed, fetch_alpaca_snapshots

        snaps = fetch_alpaca_snapshots(syms) or {}
        out: Dict[str, Any] = {}
        for t in syms:
            snap = snaps.get(t)
            if not isinstance(snap, dict):
                continue
            trade = snap.get("latestTrade") or {}
            day = snap.get("dailyBar") or {}
            prev = snap.get("prevDailyBar") or {}
            last = _num(trade.get("p"))
            at = trade.get("t") if isinstance(trade.get("t"), str) else None
            if last is None:
                last, at = _num(day.get("c")), day.get("t") if isinstance(day.get("t"), str) else None
            if last is None:
                continue
            # Before the open, dailyBar is still the last finished session: that is the base.
            trade_day, bar_day = _trade_day(at), _trade_day(day.get("t"))
            base_bar = day if trade_day and bar_day and trade_day > bar_day else prev
            base = _num(base_bar.get("c"))
            out[t] = {"last": round(last, 4), "prev_close": base,
                      "chg_pct": round((last - base) / base * 100.0, 2) if base else None, "at": at}
        feed = None
        try:
            feed = str(_get_alpaca_data_feed()).strip().lower() or None
        except Exception:
            pass
        return {"quotes": out, "feed": feed}

    key = ("quotes", tuple(syms))
    try:
        res = quote_cache.get(key, load, ttl_s=QUOTE_TTL_S)
    except Exception:
        return {"quotes": {}, "feed": None}
    if not res.get("quotes"):
        quote_cache.clear_key(key)
    return res


SCORE_HISTORY_RUNS = 24  # about eight sessions of full-market scans
score_cache = TTLCache(max_entries=SCORE_HISTORY_RUNS * 2)


def _run_ticker_scores(run_id: int) -> Dict[str, Dict[str, Any]]:
    """Ticker -> HSF Score and signals in one saved market run, scored the Scanner's way.
    Read straight from the run (not run_df's small cache, which holds the latest few runs)."""
    def load() -> Dict[str, Dict[str, Any]]:
        from db.runs import load_run_results
        from ui.results_intelligence import consolidate_scanner_results

        raw = load_run_results(int(run_id))
        if not raw:
            return {}
        data = json.loads(raw) if isinstance(raw, str) else raw
        rows = data if isinstance(data, list) else [data]
        return {o["ticker"]: {"score": o.get("score"), "signals": [str(s) for s in o.get("signals") or []]}
                for o in consolidate_scanner_results(rows, top_n=None)}

    return score_cache.get(("scores", int(run_id)), load, ttl_s=RUN_TTL_S, stale_s=RUN_TTL_S)


def score_history(ticker: str, entitlements: Dict[str, bool]) -> Dict[str, Any]:
    """The ticker's HSF Score in each recent full-market scan, oldest first, with the same
    scores as the Scanner. score is None when the ticker wasn't ranked in that scan.
    signals_added names the signals new since the scan before. PreBreakout is Premium."""
    t = ticker.strip().upper()
    early = bool(entitlements.get("can_early_breakout"))
    runs = list(reversed(market_runs()[:SCORE_HISTORY_RUNS]))
    points: List[Dict[str, Any]] = []
    prev: Optional[set] = None
    for run in runs:
        try:
            hit = _run_ticker_scores(int(run["id"])).get(t)
        except Exception:
            continue  # an unreadable run is a gap, not an error
        signals = [s for s in (hit or {}).get("signals", []) if early or s != "prebreakout"]
        added = sorted(set(signals) - prev) if (hit and prev is not None) else []
        points.append({"time": _iso(run["created_at"]), "score": (hit or {}).get("score"),
                       "signals": signals, "signals_added": added})
        prev = set(signals)
    return {"ticker": t, "points": points}
