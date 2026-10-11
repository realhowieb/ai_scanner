"""Market pages for API clients: earnings (P1-68), Market Brief (P1-69) and
the Day Trader snapshot (P1-70). Each reuses the web page's data code; plan
gates are applied by the routes in api.main.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any, Dict, List, Optional, Sequence

MAX_EARNINGS_DAYS = 30


def earnings(days: int, tickers: Optional[Sequence[str]] = None,
             today: Optional[dt.date] = None) -> List[Dict[str, Any]]:
    """Upcoming earnings in the next `days` days (DB only, like the web's
    'Earnings this week' panel), optionally only for `tickers`."""
    from db.earnings import fetch_earnings_this_week

    today = today or dt.datetime.now(dt.timezone.utc).date()
    wanted = {t.strip().upper().replace(".", "-") for t in tickers or [] if t.strip()}
    out = []
    for r in fetch_earnings_this_week(days_ahead=int(days)) or []:
        sym = str(r.get("symbol") or "").strip().upper()
        if not sym or (wanted and sym not in wanted and sym.replace("-", ".") not in wanted):
            continue
        d = r.get("earnings_date")
        if isinstance(d, dt.datetime):
            d = d.date()
        out.append({"ticker": sym, "earnings_date": d.isoformat() if isinstance(d, dt.date) else None,
                    "days_until": (d - today).days if isinstance(d, dt.date) else None,
                    "time": (str(r.get("earnings_time")).strip().lower() or None) if r.get("earnings_time") else None})
    out.sort(key=lambda x: (x["days_until"] if x["days_until"] is not None else 10_000, x["ticker"]))
    return out


# ---- Market Brief (P1-69) ------------------------------------------------------------------------
BRIEF_TTL_S = 300   # the web caches the brief for 5 minutes too
BRIEF_STALE_S = 6 * 3600  # past 5 minutes, serve the last brief and rebuild it in the background


def _brief_core() -> Optional[Dict[str, Any]]:
    """The user-independent brief (ui.market_brief._compute_brief, the web's builder)
    plus opportunities compared with the previous snapshot, read-only: the API never
    writes opportunity snapshots or research rows (the web page does that)."""
    from api.today import _cached

    def load():
        from ui.market_brief import _base_ticker, _compute_brief, _market_phase

        data = _compute_brief()
        if not data:
            return None
        compared: List[Dict[str, Any]] = []
        has_previous = False
        try:
            from ui.opportunities import build_opportunities, compare_opportunities

            opps = build_opportunities(data, base_ticker=_base_ticker, top_n=5)
            previous = None
            try:
                from db.opportunity_snapshots import load_previous_opportunity_snapshot

                prev = load_previous_opportunity_snapshot(data.get("snapshot_time"), context="market_brief")
                previous = prev.get("opportunities") if prev else None
            except Exception:
                previous = None
            has_previous = previous is not None
            compared = compare_opportunities(opps, previous)
        except Exception:
            compared = []
        return {"data": data, "compared": compared, "has_previous": has_previous, "phase": _market_phase()}

    return _cached("brief", load, ttl_s=BRIEF_TTL_S, stale_s=BRIEF_STALE_S)


_EARN_FLAG = re.compile(r"^\s*(\S+)\s*(?:⚠️\s*E(\d+)d)?\s*$")


def _split_flag(label: Any) -> tuple:
    """'MSFT ⚠️E3d' (the digest's earnings flag) -> ('MSFT', 3); 'MSFT' -> ('MSFT', None)."""
    m = _EARN_FLAG.match(str(label or ""))
    if not m:
        return str(label or "").strip(), None
    return m.group(1), int(m.group(2)) if m.group(2) else None


def brief(entitlements: Dict[str, bool]) -> Dict[str, Any]:
    from analytics.data_freshness import describe
    from ui.entitlement_view import redact_prebreakout_rows

    core = _brief_core()
    if not core:
        return {"available": False, "snapshot_time": None, 'freshness': describe(None)}
    d = core["data"]
    early = bool(entitlements.get("can_early_breakout"))
    picks = []
    from analytics.prediction_provenance import customer_opportunity
    for p in (d.get("picks") or []) if early else []:
        p = customer_opportunity(p)
        ticker, edays = _split_flag(p.get("symbol"))
        picks.append({**{k: v for k, v in p.items() if k != "symbol"}, "ticker": ticker, "earnings_days": edays})
    gappers = []
    for g in d.get("gappers") or []:
        ticker, edays = _split_flag(g.get("ticker"))
        gappers.append({"ticker": ticker, "last": g.get("last"), "chg_pct": g.get("chg_pct"),
                        "gap_pct": g.get("gap_pct"), "earnings_days": edays})
    breadth = d.get("breadth")
    return {
        "available": True,
        "snapshot_time": d.get("snapshot_time"),
        'freshness': describe(d.get('snapshot_time'), market_data_at=d.get('market_data_at')),
        "phase": core["phase"],
        "market": [{"label": lbl, "last": last, "chg_pct": chg} for lbl, last, chg in d.get("market_close") or []],
        "breadth": {"advancers": breadth[0], "decliners": breadth[1]} if breadth else None,
        "sectors": [{"sector": s, "chg_pct": c} for s, c in d.get("sectors") or []],
        "opportunities": redact_prebreakout_rows(core["compared"], allowed=early),
        "has_previous_snapshot": core["has_previous"],
        "gappers": gappers,
        "gainers": [{"ticker": t, "chg_pct": c} for t, c in d.get("gainers") or []],
        "losers": [{"ticker": t, "chg_pct": c} for t, c in d.get("losers") or []],
        "golden_crosses": list(d.get("golden") or []),
        "top_breakout_scores": [{"ticker": t, "score": s} for t, s in d.get("top_setups") or []],
        "prebreakout_picks": picks,
        "prebreakout_locked": not early,
        "earnings_today": list(d.get("earnings_today") or []),
    }


# ---- Day Trader (P1-70) --------------------------------------------------------------------------
DT_SOURCES = ("watchlist", "movers", "movers_sp500", "movers_nasdaq", "premarket", "postmarket",
              "scan_picks", "megacaps", "custom")
DT_ROWS_TTL_S = 30          # live quotes: shared across users for 30 s (the web refreshes every 30-60 s)
DT_MOVERS_TTL_S = 120       # the movers screen, as on the web
# Past their TTL, Day Trader lists and quotes are served for this long while one
# background reload runs, so a visitor never waits on the movers screen (~16 s).
DT_MOVERS_STALE_S = 600
DT_ROWS_STALE_S = 90


def _dt_symbols(source: str, custom: Sequence[str], watch: Sequence[str]) -> List[str]:
    from api.today import _cached
    from ui import day_trader as dtm

    if source == "custom":
        return dtm._parse_symbols(",".join(custom), dtm.MAX_SYMBOLS)
    if source == "watchlist":
        return dtm._parse_symbols(",".join(watch), dtm.MAX_SYMBOLS)
    if source == "megacaps":
        return dtm._parse_symbols(dtm.MEGA_CAPS)
    loaders = {
        "movers": lambda: dtm._top_movers_symbols(),
        "movers_sp500": lambda: dtm._top_movers_symbols(universe=dtm._sp500_universe()),
        "movers_nasdaq": lambda: dtm._top_movers_symbols(universe=dtm._nasdaq_universe()),
        "premarket": lambda: dtm._session_scan_symbols("premarket"),
        "postmarket": lambda: dtm._session_scan_symbols("postmarket"),
        "scan_picks": lambda: dtm._scan_pick_symbols(),
    }
    return list(_cached(("dt_source", source), loaders[source], ttl_s=DT_MOVERS_TTL_S,
                        stale_s=DT_MOVERS_STALE_S) or [])


def day_trader(source: str, custom: Sequence[str] = (), watch: Sequence[str] = ()) -> Dict[str, Any]:
    """The web's Day Trader table: live Alpaca snapshot metrics for the source's
    symbols, with the day-trade score; plus the market state."""
    from analytics.day_trade_display import enrich_row, rank_key
    from api.today import _cached
    from ui import day_trader as dtm

    symbols = _dt_symbols(source, custom, watch)
    state = _cached("dt_state", lambda: dtm.market_state(clock_is_open=dtm._fetch_clock_is_open()), ttl_s=60,
                    stale_s=300)
    rows: List[Dict[str, Any]] = []
    if symbols:
        def load():
            from market_data import build_day_trader_metrics

            return build_day_trader_metrics(list(symbols)) or []

        now = dt.datetime.now(dt.timezone.utc)
        rows = sorted((enrich_row(r, state, now)
                       for r in _cached(("dt_rows", tuple(symbols)), load, ttl_s=DT_ROWS_TTL_S,
                                        stale_s=DT_ROWS_STALE_S)), key=rank_key)
    flagged = [r for r in rows if r.get("quote_flags")]
    bullish = [r for r in rows if r.get("dt_direction") == "bullish" and not r.get("quote_flags")]
    bearish = [r for r in rows if r.get("dt_direction") == "bearish" and not r.get("quote_flags")]
    return {"state": state, "source": source, "symbols": symbols, "missing": max(0, len(symbols) - len(rows)),
            "as_of": dt.datetime.now(dt.timezone.utc),
            "summary": {"strong": sum(1 for r in rows if r.get("dt_quality") == "strong"),
                        "developing": sum(1 for r in rows if r.get("dt_quality") == "developing"),
                        "flagged": len(flagged), "missing": max(0, len(symbols) - len(rows)),
                        "best_long": (bullish[0].get("ticker") if bullish else None),
                        "best_short": (bearish[0].get("ticker") if bearish else None)},
            "rows": rows}


def day_trader_sparklines(symbols: Sequence[str]) -> Dict[str, Any]:
    """Latest-session 1-minute closes per symbol for the Day Trader row sparklines.
    Shares the stair-stepper check's minute-bar cache."""
    from analytics.day_trade_display import sparkline
    from api.today import _cached
    from ui.day_trader import _parse_symbols
    from ui.stair_stepper import MAX_CHECK, fetch_recent_minute_bars

    checked = _parse_symbols(",".join(symbols), MAX_CHECK)
    bars = _cached(("dt_minute", tuple(sorted(set(checked)))), lambda: fetch_recent_minute_bars(checked),
                   ttl_s=DT_MOVERS_TTL_S) if checked else {}
    return {"checked": checked, "series": {s: sparkline((bars or {}).get(s) or []) for s in checked}}


def stair_steppers(symbols: Sequence[str], *, window: int, direction: str, r2_min: float,
                   max_pullback_pct: float, min_trend_pct_per_hour: float) -> Dict[str, Any]:
    """The web's Stair-steppers check (P2-26): smooth 1-minute trends."""
    from analytics.stair_step import is_stair_stepper
    from api.today import _cached
    from ui.day_trader import _parse_symbols
    from ui.stair_stepper import MAX_CHECK, build_rows, fetch_recent_minute_bars

    checked = _parse_symbols(",".join(symbols), MAX_CHECK)
    bars = _cached(("dt_minute", tuple(sorted(set(checked)))), lambda: fetch_recent_minute_bars(checked),
                   ttl_s=DT_MOVERS_TTL_S) if checked else {}
    rows = build_rows(bars, checked, window)
    hits = [r for r in rows if is_stair_stepper(r, r2_min=r2_min, direction=direction,
                                                max_pullback_pct=max_pullback_pct,
                                                min_trend_pct_per_hour=min_trend_pct_per_hour)]
    return {"checked": checked, "matches": hits, "all": rows}
