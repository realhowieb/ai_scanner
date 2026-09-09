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
    return {
        "gappers": gappers,
        "golden": golden,
        "top_setups": top_setups,
        "picks": picks,
        "earnings_today": earnings_today,
        "market_close": market_close,
        "gainers": gainers,
        "losers": losers,
        "snapshot_time": _snapshot_time(),
        "yesterday": _yesterday_performance(),
    }


if st is not None:

    @st.cache_data(ttl=300, show_spinner="Building your market brief…")
    def _brief_cached() -> Optional[Dict[str, Any]]:
        return _compute_brief()

else:  # pragma: no cover
    _brief_cached = _compute_brief


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

    ts = data.get("snapshot_time")
    if ts is not None:
        try:
            st.caption(f"📸 As of {ts:%b %d, %I:%M %p} UTC (latest scan snapshot).")
        except Exception:
            pass

    _render_market_pulse(data.get("market_close") or [])
    _render_gappers(data.get("gappers") or [])
    _render_day_movers(data.get("gainers") or [], data.get("losers") or [])
    _render_setups(data.get("golden") or [], data.get("top_setups") or [])
    _render_picks(data.get("picks") or [])
    _render_watchlist(data.get("earnings_today") or [])
    _render_fired_alerts((st.session_state.get("username") or "").strip().lower())
    _render_yesterday(data.get("yesterday"))
    _render_actions(data)


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
