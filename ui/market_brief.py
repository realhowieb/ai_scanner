"""Market brief panel: the morning-digest content, surfaced (and actionable) in-app.

Renders the same Top gappers / Today's setups / PreBreakout picks the morning
email sends — reusing scheduler.morning_digest so email and UI never drift — plus
UI-only value: your watchlist today, earnings-today flags, how yesterday's brief
actually did, a freshness stamp, and per-ticker Chart / Alert / Watch actions.
Never raises into the app.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


# --------------------------- user-independent core ---------------------------

def _snapshot_time():
    try:
        from db.runs import list_snapshot_runs

        runs = list_snapshot_runs(days=3, limit=5) or []
        return runs[0].get("created_at") if runs else None
    except Exception:
        return None


def _yesterday_performance() -> Optional[Dict[str, Any]]:
    """The prior day's top breakout picks and how they've moved since. Or None."""
    try:
        from db.runs import list_snapshot_runs, load_many_run_results
        from market_data import get_latest_quotes
        from scheduler.morning_digest import _symbol_column, _todays_setups
        from ui.app_runtime import normalize_results_to_df

        runs = list_snapshot_runs(days=7, limit=25) or []
        seen, day_runs = set(), []
        for r in runs:                       # newest-first, one per calendar day
            ca = r.get("created_at")
            if not ca or ca.date() in seen:
                continue
            seen.add(ca.date())
            day_runs.append(r)
        if len(day_runs) < 2:
            return None
        prior = day_runs[1]
        raw = load_many_run_results([prior["id"]]).get(int(prior["id"]))
        df = normalize_results_to_df(raw) if raw else None
        if df is None or len(df) == 0:
            return None
        _golden, top = _todays_setups(df)    # top = [(ticker, score)]
        picks = (top or [])[:5]
        if not picks:
            return None
        col = _symbol_column(df)
        entry: Dict[str, float] = {}
        for _i, row in df.iterrows():
            t = str(row.get(col) or "").upper()
            try:
                entry[t] = float(row.get("Last"))
            except (TypeError, ValueError):
                pass
        quotes = get_latest_quotes([t for t, _ in picks]) or {}
        rows = []
        for t, _score in picks:
            e = entry.get(t)
            cur = (quotes.get(t) or {}).get("last")
            if e and cur:
                rows.append((t, (float(cur) - e) / e * 100.0))
        if not rows:
            return None
        return {"date": prior["created_at"].date(), "rows": rows}
    except Exception:
        return None


def _compute_brief() -> Optional[Dict[str, Any]]:
    """Assemble the user-independent brief from the latest snapshot, or None."""
    try:
        from scheduler.morning_digest import (
            _earnings_days_map,
            _earnings_today,
            _flag_earnings_rows,
            _latest_snapshot_df,
            _market_gappers,
            _prebreakout_picks,
            _todays_setups,
        )
    except Exception:
        return None
    try:
        df = _latest_snapshot_df()
    except Exception:
        df = None
    if df is None:
        return None

    gappers = _market_gappers(df) or []
    try:
        _flag_earnings_rows(gappers, _earnings_days_map([g.get("ticker") for g in gappers]))
    except Exception:
        pass
    try:
        golden, top_setups = _todays_setups(df)
    except Exception:
        golden, top_setups = [], []
    picks = _prebreakout_picks(df) or []
    try:
        pe = _earnings_days_map([p["symbol"] for p in picks])
        for p in picks:
            days = pe.get(p["symbol"])
            if days is not None:
                p["symbol"] = f"{p['symbol']} ⚠️E{days}d"
    except Exception:
        pass
    try:
        earnings_today = sorted(_earnings_today())
    except Exception:
        earnings_today = []
    # From the evening wrap: market backdrop + full gainers/losers (the down side
    # the gap-based gappers don't show).
    try:
        from scheduler.evening_wrap import _day_movers, _market_close_context

        market_close = _market_close_context()
        gainers, losers = _day_movers(df)
    except Exception:
        market_close, gainers, losers = [], [], []
    # Breadth from the snapshot's own PctChange, plus sector-ETF leaders.
    breadth = None
    try:
        import pandas as pd

        if "PctChange" in df.columns:
            pc = pd.to_numeric(df["PctChange"], errors="coerce").dropna()
            if len(pc):
                breadth = (int((pc > 0).sum()), int((pc < 0).sum()))
    except Exception:
        breadth = None
    return {
        "gappers": gappers,
        "golden": golden,
        "top_setups": top_setups,
        "picks": picks,
        "earnings_today": earnings_today,
        "market_close": market_close,
        "gainers": gainers,
        "losers": losers,
        "breadth": breadth,
        "sectors": _sector_leaders(),
        "snapshot_time": _snapshot_time(),
        "yesterday": _yesterday_performance(),
    }


_SECTOR_ETFS = {
    "XLK": "Tech", "XLF": "Financials", "XLE": "Energy", "XLV": "Health",
    "XLY": "Cons Disc", "XLP": "Staples", "XLI": "Industrials", "XLU": "Utilities",
    "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Comm",
}


def _sector_leaders() -> list:
    """[(sector, chg_pct)] for the S&P sector ETFs, best-first. Or []."""
    try:
        from market_data import build_day_trader_metrics

        rows = build_day_trader_metrics(list(_SECTOR_ETFS), with_rvol=False) or []
        out = []
        for r in rows:
            t = str(r.get("ticker") or "").upper()
            chg = r.get("chg_pct")
            if t in _SECTOR_ETFS and chg is not None:
                out.append((_SECTOR_ETFS[t], float(chg)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out
    except Exception:
        return []


if st is not None:

    @st.cache_data(ttl=300, show_spinner="Building your market brief…")
    def _brief_cached() -> Optional[Dict[str, Any]]:
        return _compute_brief()

    @st.cache_data(ttl=1800, show_spinner=False)
    def _calibration_records_cached() -> List[Dict[str, Any]]:
        """Matured HSF calibration records for the brief's historical-context
        line. Cached 30 min so it's read once, not per rerun."""
        try:
            from analytics.hsf_calibration import build_calibration_dataset

            return build_calibration_dataset(days_back=180).get("records") or []
        except Exception:
            return []

else:  # pragma: no cover
    _brief_cached = _compute_brief

    def _calibration_records_cached() -> List[Dict[str, Any]]:
        return []


# --------------------------------- helpers -----------------------------------

def _base_ticker(t: str) -> str:
    """Strip any '⚠️E{n}d' earnings flag → the bare symbol."""
    return str(t or "").split(" ")[0].upper()


def _watchlist_rows(user: str) -> List[Dict[str, Any]]:
    """The user's watchlist quotes — session first, else a fresh compute."""
    rows = st.session_state.get("active_watchlist_quote_rows")
    if rows:
        return rows
    try:
        from db.watchlists import get_watchlist_tickers, list_watchlists
        from market_data import build_day_trader_metrics

        tickers: List[str] = []
        for wl in (list_watchlists(user) or []):
            tickers += get_watchlist_tickers(wl.get("id"), user) or []
        tickers = sorted({str(t).upper() for t in tickers if t})
        return build_day_trader_metrics(tickers, with_rvol=False) if tickers else []
    except Exception:
        return []


def _add_to_watchlist(ticker: str) -> None:
    user = (st.session_state.get("username") or "").strip().lower()
    aid = st.session_state.get("active_watchlist_id")
    if not user or not aid:
        st.warning("Open the Watchlists page once to set an active watchlist first.")
        return
    try:
        from db.watchlists import get_watchlist_tickers, set_watchlist_tickers

        cur = get_watchlist_tickers(aid, user) or []
        up = _base_ticker(ticker)
        if up in [str(t).upper() for t in cur]:
            st.toast(f"{up} is already in your watchlist")
            return
        set_watchlist_tickers(aid, user, list(cur) + [up])
        st.session_state["active_watchlist_tickers"] = list(cur) + [up]
        st.toast(f"Added {up} to your watchlist")
    except Exception:
        st.warning("Could not add to watchlist.")


# --------------------------------- render ------------------------------------

def render_market_brief() -> None:
    """Render the actionable market brief. Never raises."""
    if st is None:
        return
    try:
        c1, c2 = st.columns([3, 1])
        if c2.button("🔄 Refresh", key="brief_refresh"):
            _brief_cached.clear()
            st.rerun()
        data = _brief_cached()
    except Exception:
        data = None
    if not data:
        st.info(
            "No recent scan snapshot yet — the market brief appears after the "
            "day's first scan runs (same content as your morning email)."
        )
        return

    user = (st.session_state.get("username") or "").strip().lower()
    ts = data.get("snapshot_time")
    if ts is not None:
        try:
            st.caption(f"📸 As of {ts:%b %d, %I:%M %p} UTC (latest scan snapshot).")
        except Exception:
            pass

    phase = _market_phase()

    # ---- glance layer (always on) ----
    # Compute opportunities + movement once (cached per snapshot).
    compared, previous = compute_compared_opportunities(data)
    # A. Market state — regime + compact metrics + freshness.
    render_market_header(data, phase)
    _render_claude_narrative(data)
    # B. Since last scan — only meaningful changes, only when history exists.
    render_since_last_scan(compared, previous)
    # C / D. Top Opportunities (score movement + status transitions) + detail.
    render_top_opportunities(compared, data)
    # E. What to watch next (~3 deterministic items).
    render_watch_next(compared)
    # F. HSF signal performance (outcome scorecard).
    render_signal_scorecard()
    # G. Sector leadership (compact leaders / laggards).
    render_sector_leadership(data)
    st.markdown("---")
    # H. Secondary market detail (existing sections, demoted below the fold).
    _render_standouts(data)
    _render_market_pulse(data.get("market_close") or [])
    # Admin-only: HSF score calibration evidence (read-only; changes nothing).
    try:
        if (st.session_state.get("entitlements") or {}).get("can_diagnostics"):
            from ui.hsf_calibration_report import render_hsf_calibration_report

            with st.expander("🔬 HSF score calibration (admin)", expanded=False):
                render_hsf_calibration_report()
    except Exception:
        pass

    # ---- detail sections: user-toggleable, time-aware order ----
    keys = [k for k, _ in _TOGGLEABLE]
    labels = {k: lbl for k, lbl in _TOGGLEABLE}
    with st.expander("⚙️ Customize sections", expanded=False):
        chosen = st.multiselect(
            "Sections to show", keys, default=keys,
            format_func=lambda k: labels.get(k, k), key="brief_sections",
        )
    chosen = set(chosen) if chosen else set(keys)

    if phase in ("afterhours", "closed"):
        order = ["movers", "alerts", "gappers", "setups", "picks", "positions",
                 "watchlist", "catalysts", "yesterday"]
    else:
        order = ["gappers", "movers", "setups", "picks", "positions",
                 "watchlist", "alerts", "catalysts", "yesterday"]
    render_map = {
        "gappers": lambda: _render_gappers(data.get("gappers") or []),
        "movers": lambda: _render_day_movers(data.get("gainers") or [], data.get("losers") or []),
        "setups": lambda: _render_setups(data.get("golden") or [], data.get("top_setups") or []),
        "picks": lambda: _render_picks(data.get("picks") or []),
        "positions": lambda: _render_open_positions(user),
        "watchlist": lambda: _render_watchlist(data.get("earnings_today") or []),
        "alerts": lambda: _render_fired_alerts(user),
        "catalysts": lambda: _render_catalysts(data),
        "yesterday": lambda: _render_yesterday(data.get("yesterday")),
    }
    for name in order:
        if name in chosen:
            render_map[name]()

    st.markdown("---")
    _render_email_button(user, data)
    _render_actions(data)


_TOGGLEABLE = [
    ("gappers", "🚀 Gappers"), ("movers", "📊 Movers"), ("setups", "🎯 Setups"),
    ("picks", "🧠 PreBreakout"), ("positions", "💼 Open positions"),
    ("watchlist", "📋 Watchlist"), ("alerts", "🔔 Fired alerts"),
    ("catalysts", "📅 Catalysts"), ("yesterday", "📊 Yesterday"),
]


def _market_summary(data: Dict[str, Any]) -> Optional[str]:
    """A one-line, deterministic synthesis of the brief for the top of the page."""
    bits: List[str] = []
    spy = next((c for (lbl, _last, c) in (data.get("market_close") or [])
                if "SPY" in str(lbl) and c is not None), None)
    if spy is not None:
        tone = "Risk-on" if spy >= 0.15 else "Risk-off" if spy <= -0.15 else "Mixed"
        bits.append(f"{tone} — SPY {spy:+.1f}%")
    b = data.get("breadth")
    if b:
        bits.append(f"breadth {b[0]}/{b[1]}")
    sec = data.get("sectors") or []
    if sec:
        bits.append(f"{sec[0][0]} leading")
    n = len(_standouts(data))
    if n:
        bits.append(f"{n} standout{'s' if n != 1 else ''}")
    et = len(data.get("earnings_today") or [])
    if et:
        bits.append(f"{et} earnings today")
    return " · ".join(bits) if bits else None


def _render_breadth_sectors(data: Dict[str, Any]) -> None:
    b = data.get("breadth")
    sec = data.get("sectors") or []
    if not b and not sec:
        return
    c1, c2 = st.columns([1, 2])
    if b:
        c1.metric("Breadth (adv/dec)", f"{b[0]} / {b[1]}")
    if sec:
        lead, lag = sec[0], sec[-1]
        c2.caption(
            f"🟢 Leading: **{lead[0]}** {lead[1]:+.1f}%   ·   "
            f"🔴 Lagging: **{lag[0]}** {lag[1]:+.1f}%"
        )


def _open_positions(user: str) -> List[tuple]:
    """[(ticker, pnl_pct)] for open journal trades marked to now. Or []."""
    try:
        from db.trades import list_trades
        from market_data import get_latest_quotes

        trades = [t for t in (list_trades(user) or []) if not t.get("closed_at")]
        if not trades:
            return []
        quotes = get_latest_quotes(sorted({t["ticker"] for t in trades})) or {}
        out = []
        for t in trades:
            try:
                entry = float(t.get("entry_price") or 0)
            except (TypeError, ValueError):
                entry = 0.0
            cur = (quotes.get(t["ticker"]) or {}).get("last")
            if entry > 0 and cur:
                out.append((t["ticker"], (float(cur) - entry) / entry * 100.0))
        return out
    except Exception:
        return []


def _render_open_positions(user: str) -> None:
    if not user:
        return
    pos = _open_positions(user)
    if not pos:
        return
    st.markdown("### 💼 Your open positions")
    st.markdown("  ·  ".join(
        f"{'🟢' if r >= 0 else '🔴'} {t} {r:+.1f}%" for t, r in pos
    ))
    st.caption("Open journal positions, marked to the latest quote.")


def _market_phase() -> Optional[str]:
    try:
        import datetime as _dt

        from ui.day_trader import market_state

        return market_state(_dt.datetime.now(_dt.timezone.utc))
    except Exception:
        return None


def _render_phase_banner(phase: Optional[str]) -> None:
    banner = {
        "premarket": "🌅 **Premarket** — focus on today's setups & gappers.",
        "open": "🔔 **Market open** — live gainers/losers below.",
        "afterhours": "🌙 **After the close** — here's how the day went.",
        "closed": "🌙 **Market closed** — recap of the last session.",
    }.get(phase or "", "")
    if banner:
        st.caption(banner)


def _standouts(data: Dict[str, Any]) -> List[tuple]:
    """[(ticker, [tags])] for names appearing across ≥2 brief lists, most first."""
    tags: Dict[str, List[str]] = {}

    def add(raw, tag):
        t = _base_ticker(raw)
        if not t:
            return
        tags.setdefault(t, [])
        if tag not in tags[t]:
            tags[t].append(tag)

    for g in (data.get("gappers") or []):
        add(g.get("ticker"), "gapper")
    for t in (data.get("golden") or []):
        add(t, "golden cross")
    for t, _s in (data.get("top_setups") or []):
        add(t, "breakout")
    for p in (data.get("picks") or []):
        add(p.get("symbol"), "prebreakout")
    for t, _c in (data.get("gainers") or []):
        add(t, "gainer")
    for t, _c in (data.get("losers") or []):
        add(t, "loser")

    out = [(t, tg) for t, tg in tags.items() if len(tg) >= 2]
    out.sort(key=lambda x: len(x[1]), reverse=True)
    return out


def _render_standouts(data: Dict[str, Any]) -> None:
    rows = _standouts(data)
    if not rows:
        return
    st.markdown("### ⭐ Standouts — multiple signals")
    for t, tg in rows[:8]:
        st.markdown(f"- **{t}** — {' + '.join(tg)}")
    st.caption("Names showing up across more than one list — where the day's "
               "action clusters. Not a prediction; just where to look first.")


def render_signal_scorecard() -> None:
    """D. HSF signal performance — how flagged signals actually resolved.

    Uses the immutable signal_outcomes table (real forward returns) only. Uses
    'positive-outcome' language, not 'win rate' — these are 5-day positive-move
    outcomes, not entries/exits of a trading strategy. Shows a 'maturing' note
    while windows complete; silent if the table is empty/unavailable.
    """
    if st is None:
        return
    try:
        from db.signal_outcomes import summarize_outcomes_by_type, summarize_recent_outcomes

        s = summarize_recent_outcomes(days_back=7)
    except Exception:
        return
    completed = int(s.get("completed") or 0)
    pending = int(s.get("pending") or 0)
    flagged = completed + pending
    if flagged == 0:
        return
    st.markdown("### 📈 HSF Signal Performance — 7d")
    if completed == 0:
        st.caption(
            f"{flagged} signal(s) flagged — outcomes still maturing (5-day "
            "window). Positive-outcome rates fill in as signals complete."
        )
        return
    rate = s.get("hit_rate")
    aw = s.get("avg_winner")
    al = s.get("avg_loser")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flagged", flagged, help=f"{completed} matured · {pending} maturing")
    c2.metric("Positive outcomes", f"{rate*100:.0f}%" if rate is not None else "—",
              help="Share that made a +4% move within 5 trading days. Not a "
                   "trading win rate — no defined entries/exits.")
    c3.metric("Avg positive move", f"{aw*100:+.1f}%" if aw is not None else "—")
    c4.metric("Avg negative move", f"{al*100:+.1f}%" if al is not None else "—")

    try:
        by_type = summarize_outcomes_by_type(days_back=7)
    except Exception:
        by_type = []
    if by_type:
        st.caption("By signal type (positive-outcome rate):")
        st.markdown("\n".join(
            f"- **{r['signal_type']}** — {r['positive_rate']*100:.0f}%"
            f" ({r['completed']} matured)"
            for r in by_type if r.get("positive_rate") is not None
        ))
    best_t, best_r = s.get("best_ticker"), s.get("best_return")
    if best_t and best_r is not None:
        st.caption(f"Best 5-day move: **{best_t}** {best_r*100:+.1f}%")
    st.caption("Outcome = 5-day positive move (reached +4% before −2%), the "
               "models' target — educational, not a trading result.")


def _brief_narrative_facts(data: Dict[str, Any]) -> str:
    """Compact, real facts block for Claude to narrate (never invents)."""
    lines: List[str] = []
    for lbl, _last, chg in (data.get("market_close") or []):
        if chg is not None and ("SPY" in str(lbl) or "QQQ" in str(lbl)):
            lines.append(f"- {lbl}: {chg:+.2f}%")
    b = data.get("breadth")
    if b:
        lines.append(f"- Breadth (advancers/decliners): {b[0]}/{b[1]}")
    sec = data.get("sectors") or []
    if sec:
        top = ", ".join(f"{n} {c:+.1f}%" for n, c in sec[:3] if c is not None)
        if top:
            lines.append(f"- Leading sectors: {top}")
    st_names = [str(x) for x in _standouts(data)][:5]
    if st_names:
        lines.append(f"- Confluence standouts: {', '.join(st_names)}")
    counts = []
    for key, label in (("gappers", "gappers"), ("picks", "PreBreakout picks"),
                       ("top_setups", "setups"), ("earnings_today", "earnings today")):
        n = len(data.get(key) or [])
        if n:
            counts.append(f"{n} {label}")
    if counts:
        lines.append(f"- Counts: {', '.join(counts)}")
    return "\n".join(lines)


def _render_claude_narrative(data: Dict[str, Any]) -> None:
    """A 2-3 sentence plain-English brief written by Claude from real data.

    Grounded strictly on the facts block; cached per snapshot so it costs one
    call per scan, not per rerun. Silent when AI is off or facts are thin.
    """
    if st is None:
        return
    try:
        from ui.ai import ask_claude, is_configured

        if not is_configured():
            return
    except Exception:
        return
    facts = _brief_narrative_facts(data)
    if not facts.strip():
        return
    ts = data.get("snapshot_time")
    cache_key = f"brief_narrative_{ts}"
    cached = st.session_state.get(cache_key)
    if cached is None:
        system = (
            "You write a 2-3 sentence morning market brief for an experienced "
            "trader. Use ONLY the facts provided — do NOT invent prices, news, "
            "levels, forecasts, or tickers not listed. Be plain, concise, and "
            "non-promissory (describe conditions and what to watch, don't predict). "
            "No emojis, no bullet points, no preamble."
        )
        text, err = ask_claude(
            system=system,
            user=f"Today's scan facts:\n{facts}",
            max_tokens=200,
            username=(st.session_state.get("username") or "").strip().lower() or None,
            feature="market_brief_narrative",
        )
        cached = text or ""  # cache empty on error so we don't retry every rerun
        st.session_state[cache_key] = cached
    if cached:
        st.markdown(f"> {cached}")


def _breadth_word(adv: Any, dec: Any) -> str:
    total = (adv or 0) + (dec or 0)
    if not total:
        return "—"
    ratio = (adv or 0) / total
    if ratio >= 0.60:
        return "Bullish"
    if ratio <= 0.40:
        return "Bearish"
    return "Neutral"


def _ago(secs: int) -> str:
    secs = max(int(secs), 0)
    if secs < 90:
        return f"{secs} sec ago"
    mins = secs // 60
    if mins < 90:
        return f"{mins} min ago"
    return f"{mins // 60} hr ago"


def _freshness_label(ts: Any, phase: Optional[str]) -> Optional[str]:
    """User-friendly freshness derived from real snapshot time + market phase.

    Never fakes live status: 'Live' only when the market is actually open/
    premarket; otherwise 'Last session'. Shows ET, not raw UTC.
    """
    if ts is None:
        return None
    import datetime as _dt

    ts_utc = ts if getattr(ts, "tzinfo", None) else (ts.replace(tzinfo=_dt.timezone.utc) if hasattr(ts, "replace") else None)
    if ts_utc is None:
        return None
    stamp = None
    try:
        from zoneinfo import ZoneInfo

        et = ts_utc.astimezone(ZoneInfo("America/New_York"))
        stamp = et.strftime("%-I:%M %p ET")
    except Exception:
        stamp = None
    if phase in ("open", "premarket"):
        try:
            secs = int((_dt.datetime.now(_dt.timezone.utc) - ts_utc).total_seconds())
            return f"🟢 Live · updated {_ago(secs)}"
        except Exception:
            return "🟢 Live" + (f" · {stamp}" if stamp else "")
    return "⚪ Last session" + (f" · updated {stamp}" if stamp else "")


def render_market_header(data: Dict[str, Any], phase: Optional[str]) -> None:
    """A. Market state — compact risk headline + 4-metric row + freshness."""
    if st is None:
        return
    mc = data.get("market_close") or []

    def find(sym: str):
        return next(((last, c) for (lbl, last, c) in mc if sym in str(lbl)), (None, None))

    spy_last, spy_chg = find("SPY")
    qqq_last, qqq_chg = find("QQQ")
    b = data.get("breadth")
    sec = data.get("sectors") or []

    tone = "MIXED"
    if spy_chg is not None:
        tone = "RISK-ON" if spy_chg >= 0.15 else "RISK-OFF" if spy_chg <= -0.15 else "MIXED"
    # Deterministic regime (richer than tone); leads the headline when available.
    regime = None
    interp = None
    try:
        from ui.opportunities import classify_market_regime

        r = classify_market_regime(spy_chg=spy_chg, qqq_chg=qqq_chg, breadth=b, sectors=sec)
        regime, interp = r.get("regime"), r.get("interpretation")
    except Exception:
        pass
    bits = [f"**{regime or tone}**"]
    if spy_chg is not None:
        bits.append(f"SPY {spy_chg:+.2f}%")
    if qqq_chg is not None:
        bits.append(f"QQQ {qqq_chg:+.2f}%")
    if b:
        bits.append(f"Breadth {b[0]}/{b[1]}")
    if sec and sec[0][1] is not None:
        bits.append(f"{sec[0][0]} leading")
    st.markdown(" &nbsp;•&nbsp; ".join(bits))
    if interp:
        st.caption(interp)
    fresh = _freshness_label(data.get("snapshot_time"), phase)
    if fresh:
        st.caption(fresh)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SPY", f"{spy_last:,.2f}" if spy_last is not None else "—",
              f"{spy_chg:+.2f}%" if spy_chg is not None else None)
    c2.metric("QQQ", f"{qqq_last:,.2f}" if qqq_last is not None else "—",
              f"{qqq_chg:+.2f}%" if qqq_chg is not None else None)
    if b:
        c3.metric("Breadth", f"{b[0]}/{b[1]}", _breadth_word(b[0], b[1]), delta_color="off")
    else:
        c3.metric("Breadth", "—")
    if sec:
        name, chg = sec[0]
        c4.metric("Leading sector", str(name), f"{chg:+.1f}%" if chg is not None else None)
    else:
        c4.metric("Leading sector", "—")


_STATUS_ICON = {"STRONG": "🟢", "WATCH": "🟡", "CAUTION": "🟠"}


def compute_compared_opportunities(data: Dict[str, Any]) -> tuple:
    """Build current opportunities, compare to the previous scan snapshot, and
    persist the current set for next time. Cached per snapshot in session_state
    so DB reads/writes happen once per scan, not once per rerun.

    Returns (compared_opportunities, previous_rows). Degrades to (current, None)
    when history/DB is unavailable — never fakes previous scores.
    """
    try:
        from ui.opportunities import build_opportunities, compare_opportunities
    except Exception:
        return [], None
    opps = build_opportunities(data, base_ticker=_base_ticker, top_n=5)
    ts = data.get("snapshot_time")

    if st is not None:
        cache = st.session_state.get("_opp_compare_cache")
        if cache and cache.get("ts") == ts:
            return cache["compared"], cache["previous"]

    previous_rows = None
    try:
        from db.opportunity_snapshots import (
            load_previous_opportunity_snapshot,
            save_opportunity_snapshot,
        )
        from ui.opportunities import to_snapshot_rows

        prev = load_previous_opportunity_snapshot(ts)
        previous_rows = prev.get("opportunities") if prev else None
        save_opportunity_snapshot(ts, to_snapshot_rows(opps))
    except Exception:
        previous_rows = None
    # Freeze full opportunities (score + components + signal-time features) into
    # signal_outcomes for leakage-safe calibration; the existing cron backfill
    # fills their forward outcomes. Idempotent per (snapshot, ticker).
    try:
        from db.signal_outcomes import freeze_opportunities

        freeze_opportunities(ts, opps)
    except Exception:
        pass

    compared = compare_opportunities(opps, previous_rows)
    if st is not None:
        st.session_state["_opp_compare_cache"] = {
            "ts": ts, "compared": compared, "previous": previous_rows}
    return compared, previous_rows


def render_top_opportunities(compared: List[Dict[str, Any]], data: Dict[str, Any]) -> None:
    """C/D. Ranked Top Opportunities with score movement + status transitions,
    plus a 'why it ranked' / chart detail."""
    if st is None:
        return
    from ui.opportunities import movement_badge

    st.markdown("### 🎯 Top Opportunities")
    if not compared:
        st.caption("No multi-signal opportunities in the latest snapshot — "
                   "see the full scanner for individual signals.")
        return
    st.caption("Ranked by HSF Opportunity Score (0-100) — confluence of existing "
               "signals; not a price target. Movement vs the previous scan.")
    rows = []
    for i, o in enumerate(compared, 1):
        tr = o.get("status_transition")
        status = f"{_STATUS_ICON.get(o['status'], '')} {o['status']}"
        if tr:
            status = f"{tr[0]} → {tr[1]}"
        rows.append({
            "#": i,
            "Ticker": o["ticker"],
            "HSF": o["score"],
            "Δ": movement_badge(o),
            "Primary setup": o["primary_setup"],
            "Signals": o["n_signals"],
            "Status": status,
        })
    try:
        cc = st.column_config
        st.dataframe(
            rows, hide_index=True, width="stretch",
            column_config={
                "#": cc.NumberColumn(width="small"),
                "HSF": cc.ProgressColumn(min_value=0, max_value=100, format="%d"),
                "Δ": cc.TextColumn(width="small"),
                "Signals": cc.NumberColumn(width="small"),
            },
        )
    except Exception:
        st.dataframe(rows, hide_index=True, width="stretch")

    tickers = [o["ticker"] for o in compared]
    pick = st.selectbox("Explain / chart", tickers, key="opp_pick", label_visibility="collapsed")
    o = next((x for x in compared if x["ticker"] == pick), None)
    if o:
        _render_opportunity_detail(o, data)


def render_since_last_scan(compared: List[Dict[str, Any]], previous: Optional[List[Dict[str, Any]]]) -> None:
    """B. Compact 'since last scan' — only meaningful changes, only when a
    previous snapshot exists."""
    if st is None:
        return
    from ui.opportunities import summarize_changes

    s = summarize_changes(compared, previous)
    if not s or not s.get("any"):
        return
    st.markdown("#### 🔄 Since last scan")
    bits = []
    if s["new"]:
        bits.append(f"{len(s['new'])} new")
    if s["strengthened"]:
        bits.append(f"{len(s['strengthened'])} strengthened")
    if s["weakened"]:
        bits.append(f"{len(s['weakened'])} weakened")
    if s["upgrades"]:
        bits.append(f"{len(s['upgrades'])} status upgrade{'s' if len(s['upgrades']) != 1 else ''}")
    if s["downgrades"]:
        bits.append(f"{len(s['downgrades'])} status downgrade{'s' if len(s['downgrades']) != 1 else ''}")
    if s["dropped"]:
        bits.append(f"{len(s['dropped'])} dropped")
    if bits:
        st.markdown(" · ".join(bits))
    line = []
    for u in s["upgrades"][:2]:
        tr = u["status_transition"]
        line.append(f"**{u['ticker']}** {tr[0]} → {tr[1]}")
    big = s.get("biggest_mover")
    if big and big.get("score_delta"):
        from ui.opportunities import movement_badge
        line.append(f"Biggest mover: **{big['ticker']}** {movement_badge(big)}")
    if s["dropped"]:
        line.append(f"Dropped from ranking: {', '.join(s['dropped'][:3])}")
    if line:
        st.caption(" · ".join(line))


def render_watch_next(compared: List[Dict[str, Any]]) -> None:
    """E. 'What to watch next' — up to ~3 deterministic developing situations."""
    if st is None or not compared:
        return
    from ui.opportunities import select_watch_next

    items = select_watch_next(compared, limit=3)
    if not items:
        return
    st.markdown("### 👀 What to watch next")
    for it in items:
        st.markdown(f"**{it['ticker']}** — {it['headline']}  \n{it['detail']}")


def render_sector_leadership(data: Dict[str, Any]) -> None:
    """G. Compact sector leaders / laggards from existing ETF sector data."""
    if st is None:
        return
    sec = data.get("sectors") or []
    rated = [(n, c) for (n, c) in sec if c is not None]
    if len(rated) < 3:
        return
    st.markdown("### 🧭 Sector leadership")
    leaders = rated[:3]
    laggards = rated[-2:] if len(rated) >= 5 else []
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Leaders**")
        st.markdown("\n".join(f"- {n} &nbsp; {c:+.1f}%" for n, c in leaders))
    if laggards:
        with c2:
            st.markdown("**Laggards**")
            st.markdown("\n".join(f"- {n} &nbsp; {c:+.1f}%" for n, c in laggards))


def _render_opportunity_detail(o: Dict[str, Any], data: Dict[str, Any]) -> None:
    from ui.opportunities import build_opportunity_explanation, movement_badge

    ex = build_opportunity_explanation(o, earnings_today=data.get("earnings_today") or [])
    sub = []
    badge = movement_badge(o)
    if badge and badge != "—":
        sub.append(badge)
    tr = o.get("status_transition")
    if tr:
        sub.append(f"{tr[0]} → {tr[1]}")
    head = f"**{o['ticker']} — HSF {o['score']}/100 · {o['status']}**"
    st.markdown(head + ("  ·  " + "  ·  ".join(sub) if sub else ""))
    if ex["reasons"]:
        st.markdown("**Why it ranked**")
        st.markdown("\n".join(f"- ✓ {r}" for r in ex["reasons"]))
    if ex["risks"]:
        st.markdown("**Risk flags**")
        st.markdown("\n".join(f"- ⚠ {r}" for r in ex["risks"]))
    # Historical context (real forward outcomes only; shows 'still building'
    # until a bucket has enough matured samples). Never fabricated.
    try:
        from analytics.hsf_calibration import historical_context

        ctx = historical_context(_calibration_records_cached(), o.get("score"))
        if ctx and ctx.get("sufficient"):
            st.caption(
                f"📊 Historical context · {ctx['bucket']} range: "
                f"{ctx['positive_rate']*100:.0f}% positive outcome (reached +4% in 5D) · "
                f"n={ctx['n']} · {ctx['confidence'].title()}"
            )
        elif ctx and ctx.get("n"):
            st.caption(f"📊 Historical context · Still building history · n={ctx['n']}")
    except Exception:
        pass
    _render_opp_ai_take(o, ex, data)

    a1, a2, a3 = st.columns(3)
    if a1.button("📈 Chart", key=f"opp_chart_{o['ticker']}"):
        st.session_state["brief_show_chart"] = o["ticker"]
    if a2.button("👁 Watch", key=f"opp_watch_{o['ticker']}"):
        _add_to_watchlist(o["ticker"])
    if a3.button("🔔 Alert", key=f"opp_alert_{o['ticker']}"):
        st.session_state["alert_price_tk"] = o["ticker"]
        try:
            st.switch_page("pages/alerts.py")
        except Exception:
            st.caption("Open Alerts from the sidebar — it's pre-filled.")
    if st.session_state.get("brief_show_chart") == o["ticker"]:
        try:
            from ui.charts import render_chart_for_ticker

            render_chart_for_ticker(o["ticker"], key=f"opp_chartimg_{o['ticker']}")
        except Exception:
            st.caption("Chart unavailable.")


def _render_opp_ai_take(o: Dict[str, Any], ex: Dict[str, List[str]], data: Dict[str, Any]) -> None:
    """Optional Claude one-liner, grounded ONLY on the deterministic reasons.

    Secondary to the deterministic explanation above; the section works fine if
    AI is off or the call fails.
    """
    if st is None or not ex.get("reasons"):
        return
    try:
        from ui.ai import ask_claude, is_configured

        if not is_configured():
            return
    except Exception:
        return
    cache_key = f"opp_ai_{o['ticker']}_{data.get('snapshot_time')}"
    cached = st.session_state.get(cache_key)
    if cached is None:
        facts = "; ".join(ex["reasons"])
        risks = "; ".join(ex["risks"]) if ex["risks"] else "none noted"
        system = (
            "You summarize a stock setup for a trader in ONE sentence. Use ONLY "
            "the confirming signals and risks provided — do NOT invent prices, "
            "levels, news, or numbers. Be plain and non-promissory. No emojis."
        )
        text, _err = ask_claude(
            system=system,
            user=f"Ticker {o['ticker']} (HSF {o['score']}/100). Confirming: {facts}. Risks: {risks}.",
            max_tokens=90,
            username=(st.session_state.get("username") or "").strip().lower() or None,
            feature="opportunity_ai_take",
        )
        cached = text or ""
        st.session_state[cache_key] = cached
    if cached:
        st.markdown(f"**AI take:** {cached}")


def _render_catalysts(data: Dict[str, Any]) -> None:
    et = set(data.get("earnings_today") or [])
    if not et:
        return
    brief: set = set()
    for g in (data.get("gappers") or []):
        brief.add(_base_ticker(g.get("ticker")))
    for t, _s in (data.get("top_setups") or []):
        brief.add(_base_ticker(t))
    for p in (data.get("picks") or []):
        brief.add(_base_ticker(p.get("symbol")))
    for t, _c in (data.get("gainers") or []) + (data.get("losers") or []):
        brief.add(_base_ticker(t))
    hits = sorted(brief & et)
    st.markdown("### 📅 Today's catalysts")
    if hits:
        st.markdown(f"**Reporting earnings today (in this brief):** {', '.join(hits)}")
    st.caption(f"{len(et)} companies report earnings today market-wide — mind the gaps.")


def _render_email_button(user: str, data: Dict[str, Any]) -> None:
    if not user:
        return
    ent = st.session_state.get("entitlements") or {}
    if not ent.get("can_email_alerts"):
        st.caption("📧 Emailing this brief on demand is a Pro feature.")
        return
    if not st.button("📧 Email me this brief", key="brief_email"):
        return
    try:
        from scheduler.morning_digest import _compose
        from ui.email_utils import send_digest_email

        watch_rows = _watchlist_rows(user)
        et = set(data.get("earnings_today") or [])
        earnings_hits = [
            tk for r in watch_rows
            if (tk := str(r.get("ticker") or r.get("Ticker") or "").upper()) in et
        ]
        html_inner, text_inner = _compose(
            user, watch_rows, data.get("gappers") or [], earnings_hits,
            data.get("picks") or [], golden=data.get("golden") or [],
            top_setups=data.get("top_setups") or [],
        )
        ok = send_digest_email(user, "Your market brief", html_inner, text_inner)
        st.toast("📧 Brief sent to your inbox" if ok else "Couldn't send the email.")
    except Exception:
        st.warning("Could not send the brief right now.")


def _render_market_pulse(market_close: List[tuple]) -> None:
    if not market_close:
        return
    cols = st.columns(len(market_close))
    for col, (label, last, chg) in zip(cols, market_close):
        try:
            col.metric(label, f"{float(last):,.2f}",
                       f"{float(chg):+.2f}%" if chg is not None else None)
        except Exception:
            pass


def _render_day_movers(gainers: List[tuple], losers: List[tuple]) -> None:
    if not gainers and not losers:
        return
    st.markdown("### 📊 Today's movers")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📈 Gainers**")
        if gainers:
            for t, chg in gainers:
                st.markdown(f"- 🟢 {t} {chg:+.1f}%")
        else:
            st.caption("—")
    with c2:
        st.markdown("**📉 Losers**")
        if losers:
            for t, chg in losers:
                st.markdown(f"- 🔴 {t} {chg:+.1f}%")
        else:
            st.caption("—")


def _render_fired_alerts(user: str) -> None:
    if not user:
        return
    try:
        from scheduler.evening_wrap import _todays_events

        events = _todays_events(user)
    except Exception:
        events = []
    if not events:
        return
    st.markdown("### 🔔 Alerts that fired today")
    for msg in events:
        st.markdown(f"- {msg}")


def _render_gappers(gappers: List[Dict[str, Any]]) -> None:
    st.markdown("### 🚀 Top market gappers")
    if not gappers:
        st.caption("No gappers in the latest snapshot.")
        return
    rows = []
    for g in gappers:
        chg, gap = g.get("chg_pct"), g.get("gap_pct")
        rows.append({
            "Ticker": g.get("ticker"),
            "Last": g.get("last"),
            "Chg %": f"{chg:+.2f}%" if chg is not None else "—",
            "Gap %": f"{gap:+.2f}%" if gap is not None else "—",
        })
    st.dataframe(rows, hide_index=True, width="stretch")


def _render_setups(golden: List[str], top_setups: List[tuple]) -> None:
    st.markdown("### 🎯 Today's setups")
    if golden:
        st.markdown(f"📈 **Fresh EMA 9/21 golden crosses:** {', '.join(golden)}")
    if top_setups:
        ts = ", ".join(f"{t} ({s:g})" for t, s in top_setups)
        st.markdown(f"🚀 **Top breakout scores:** {ts}")
    if not golden and not top_setups:
        st.caption("No fresh setups in the latest snapshot.")
    st.caption("Educational only — not financial advice; confirm setups yourself at the open.")


def _render_picks(picks: List[Dict[str, Any]]) -> None:
    if not picks:
        return
    st.markdown("### 🧠 PreBreakout picks")
    for p in picks:
        st.markdown(f"- **{p['symbol']}** — {p['prob']}% model confidence")


def _render_watchlist(earnings_today: List[str]) -> None:
    user = (st.session_state.get("username") or "").strip().lower()
    if not user:
        return
    rows = _watchlist_rows(user)
    if not rows:
        return
    st.markdown("### 📋 Your watchlist today")

    def _chg(r):
        return r.get("chg_pct") if r.get("chg_pct") is not None else r.get("Chg %")

    table, movers = [], []
    for r in rows:
        tk = str(r.get("ticker") or r.get("Ticker") or "").upper()
        chg = _chg(r)
        if not tk:
            continue
        if isinstance(chg, (int, float)):
            movers.append((tk, float(chg)))
        table.append({
            "Ticker": tk + (" 📅" if tk in earnings_today else ""),
            "Last": r.get("last") or r.get("Last"),
            "Chg %": f"{chg:+.2f}%" if isinstance(chg, (int, float)) else "—",
        })
    st.dataframe(table, hide_index=True, width="stretch")
    if movers:
        best = max(movers, key=lambda x: x[1])
        worst = min(movers, key=lambda x: x[1])
        up = sum(1 for _t, c in movers if c >= 0)
        st.caption(
            f"📌 Best {best[0]} {best[1]:+.1f}% · Worst {worst[0]} {worst[1]:+.1f}% · "
            f"{up} up / {len(movers) - up} down"
        )
    hits = [t for t in earnings_today if t in {m[0] for m in movers}]
    if hits:
        st.caption(f"📅 Reporting earnings today: {', '.join(hits)}")


def _render_yesterday(y: Optional[Dict[str, Any]]) -> None:
    if not y or not y.get("rows"):
        return
    st.markdown("### 📊 Yesterday's brief — how it did")
    parts = []
    for t, ret in y["rows"]:
        icon = "🟢" if ret >= 0 else "🔴"
        parts.append(f"{icon} {t} {ret:+.1f}%")
    avg = sum(r for _t, r in y["rows"]) / len(y["rows"])
    st.markdown("  ·  ".join(parts))
    st.caption(
        f"Top breakout picks from {y['date']:%b %d}, marked to now — avg "
        f"{avg:+.1f}%. Honest scoreboard; past performance isn't predictive."
    )


def _render_actions(data: Dict[str, Any]) -> None:
    tickers: List[str] = []
    seen: set = set()
    for g in (data.get("gappers") or []):
        t = _base_ticker(g.get("ticker"))
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)
    for t, _s in (data.get("top_setups") or []):
        tt = _base_ticker(t)
        if tt and tt not in seen:
            seen.add(tt)
            tickers.append(tt)
    for t, _c in (data.get("gainers") or []) + (data.get("losers") or []):
        tt = _base_ticker(t)
        if tt and tt not in seen:
            seen.add(tt)
            tickers.append(tt)
    for p in (data.get("picks") or []):
        t = _base_ticker(p.get("symbol"))
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)
    if not tickers:
        return

    st.markdown("### ⚡ Act on a ticker")
    pick = st.selectbox("Ticker", tickers, key="brief_action_ticker",
                        label_visibility="collapsed")
    last = None
    for g in (data.get("gappers") or []):
        if _base_ticker(g.get("ticker")) == pick:
            last = g.get("last")
            break

    a1, a2, a3 = st.columns(3)
    if a1.button("📈 Chart", key="brief_act_chart"):
        st.session_state["brief_show_chart"] = pick
    if a2.button("👁 Watch", key="brief_act_watch"):
        _add_to_watchlist(pick)
    if a3.button(f"🔔 Alert on {pick}", key="brief_act_alert"):
        st.session_state["alert_price_tk"] = pick
        if last:
            try:
                st.session_state["alert_price_val"] = round(float(last), 2)
            except (TypeError, ValueError):
                pass
        try:
            st.switch_page("pages/alerts.py")
        except Exception:
            st.caption("Open the Alerts page from the sidebar — it's pre-filled.")

    if st.session_state.get("brief_show_chart") == pick:
        try:
            from ui.charts import render_chart_for_ticker

            render_chart_for_ticker(pick, key=f"brief_chart_{pick}")
        except Exception:
            st.caption("Chart unavailable.")
