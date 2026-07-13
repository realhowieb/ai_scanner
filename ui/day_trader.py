"""⚡ Day Trader live monitor: gappers, VWAP, RVOL, today's move.

Built on chunked Alpaca snapshots (cached ~30s). Beyond the table it now:
- shows the market state (premarket / open / after-hours / closed) honestly,
- adds an AH % column after the close (last vs today's close),
- highlights rows that are "in play" (big move or elevated RVOL),
- watches with you: optional toasts when a symbol moves ±X% while open,
- offers row actions (chart / trade plan / alert-me) without leaving the page,
- sources symbols from presets (watchlist / today's scan picks / mega-caps / custom),
- refreshes via st.fragment so only the table re-renders on each tick.

Pure helpers (market_state, detect_moves) are streamlit-free and unit-tested.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

try:
    import pandas as pd
    import streamlit as st
except Exception:  # pragma: no cover - headless envs; pure helpers still work
    pd = None  # type: ignore[assignment]
    st = None  # type: ignore[assignment]

MAX_SYMBOLS = 150
MEGA_CAPS = "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO, SPY, QQQ, IWM, DIA"
HIGHLIGHT_CHG_PCT = 3.0
HIGHLIGHT_RVOL = 2.0


# ------------------------------ pure helpers -------------------------------


def market_state(now_utc: Optional[dt.datetime] = None) -> str:
    """'premarket' | 'open' | 'afterhours' | 'closed' (US equities, ET)."""
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    et = now.astimezone(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return "closed"
    minutes = et.hour * 60 + et.minute
    if 4 * 60 <= minutes < 9 * 60 + 30:
        return "premarket"
    if 9 * 60 + 30 <= minutes < 16 * 60:
        return "open"
    if 16 * 60 <= minutes < 20 * 60:
        return "afterhours"
    return "closed"


def detect_moves(
    baseline: Dict[str, float], current: Dict[str, float], threshold_pct: float
) -> List[tuple]:
    """[(sym, pct_move_since_baseline)] for symbols past the threshold."""
    out: List[tuple] = []
    for sym, last in current.items():
        base = baseline.get(sym)
        if not base or not last:
            continue
        move = (last - base) / base * 100.0
        if abs(move) >= float(threshold_pct):
            out.append((sym, round(move, 2)))
    return out


def _parse_symbols(raw: str) -> List[str]:
    parts = [p.strip().upper() for p in (raw or "").replace("\n", ",").split(",")]
    return [p for p in dict.fromkeys(parts) if p]


# ----------------------------- symbol sources ------------------------------


def _default_symbols(watch_tickers: List[str] | None) -> str:
    if watch_tickers:
        return ", ".join(dict.fromkeys(t.upper() for t in watch_tickers if str(t).strip()))
    return MEGA_CAPS


def _scan_pick_symbols(limit: int = 40) -> List[str]:
    """Tickers from the latest daily snapshot (already ranked model-first)."""
    try:
        from db.runs import list_snapshot_runs, load_many_run_results
        from ui.app_runtime import normalize_results_to_df

        runs = list_snapshot_runs(days=5, limit=5) or []
        if not runs:
            return []
        payloads = load_many_run_results([runs[0]["id"]])
        raw = payloads.get(int(runs[0]["id"]))
        df = normalize_results_to_df(raw) if raw else None
        if df is None or len(df) == 0:
            return []
        col = "Ticker" if "Ticker" in df.columns else ("Symbol" if "Symbol" in df.columns else None)
        if not col:
            return []
        return [str(t).upper() for t in df[col].head(limit).tolist() if str(t).strip()]
    except Exception:
        return []


if st is not None:

    @st.cache_data(ttl=900, show_spinner=False)
    def _scan_picks_cached() -> List[str]:
        return _scan_pick_symbols()

else:  # pragma: no cover - headless fallback
    _scan_picks_cached = _scan_pick_symbols


def _resolve_symbols(source: str, watch_tickers: List[str] | None) -> List[str]:
    if source == "Today's scan picks":
        picks = _scan_picks_cached()
        if picks:
            return picks
        st.caption("No recent scan snapshot found — using your watchlist.")
        return _parse_symbols(_default_symbols(watch_tickers))
    if source == "Mega-caps":
        return _parse_symbols(MEGA_CAPS)
    if source == "Custom":
        raw = st.text_input(
            "Symbols",
            value=st.session_state.get("dt_symbols") or _default_symbols(watch_tickers),
            key="dt_symbols",
            help="Comma-separated tickers.",
        )
        return _parse_symbols(raw)
    # "My watchlist"
    return _parse_symbols(_default_symbols(watch_tickers))


# --------------------------------- panel -----------------------------------


def render_day_trader_panel(
    watch_tickers: List[str] | None = None,
    *,
    max_symbols: int = MAX_SYMBOLS,
) -> None:
    """Render the live day-trader monitor."""
    if st is None:
        return
    try:
        from config import DAY_TRADER_ENABLED
    except Exception:
        return
    if not DAY_TRADER_ENABLED:
        return
    try:
        from market_data import build_day_trader_metrics  # noqa: F401 - availability check
    except Exception:
        return

    st.markdown("## ⚡ Day Trader — live")
    _render_state_banner()

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        source = st.selectbox(
            "Symbols source",
            ["My watchlist", "Today's scan picks", "Mega-caps", "Custom"],
            index=0 if watch_tickers else 2,
            key="dt_source",
        )
    with c2:
        refresh_label = st.selectbox(
            "Auto-refresh", ["Off", "15s", "30s", "60s"], index=0, key="dt_refresh"
        )
    with c3:
        st.write("")
        if st.button("🔄 Refresh now", key="dt_refresh_btn"):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()

    symbols = _resolve_symbols(source, watch_tickers)
    if not symbols:
        st.info("No symbols to monitor for this source.")
        return
    if len(symbols) > max_symbols:
        st.caption(f"Showing the first {max_symbols} of {len(symbols)} symbols.")
        symbols = symbols[:max_symbols]

    # Watch-mode movement notifications (session-local; compares each refresh
    # to the price when you started watching, resets per symbol after firing).
    w1, w2 = st.columns([2, 1])
    notify = w1.checkbox("🔔 Notify me on big moves while watching", key="dt_notify")
    move_thr = w2.number_input(
        "Move ≥ %", min_value=0.5, value=2.0, step=0.5, key="dt_notify_thr",
        disabled=not notify,
    )

    interval_s = {"Off": 0, "15s": 15, "30s": 30, "60s": 60}.get(refresh_label, 0)

    def _body() -> None:
        _render_table(symbols, notify=notify, move_thr=float(move_thr))

    if interval_s and hasattr(st, "fragment"):
        # Fragment re-renders only the table on each tick — the rest of the
        # page (and the wider app) doesn't rerun.
        st.fragment(run_every=f"{interval_s}s")(_body)()
    else:
        if interval_s:
            try:
                from streamlit_autorefresh import st_autorefresh

                st_autorefresh(interval=interval_s * 1000, key="dt_autorefresh")
            except Exception:
                st.caption("Auto-refresh unavailable; use 🔄 Refresh now.")
        _body()

    _render_row_actions()


def _render_state_banner() -> None:
    state = market_state()
    banner = {
        "premarket": "🌅 **Premarket** — extended-hours trades shown; volume is thin.",
        "open": "🔔 **Market open** — live regular-session data.",
        "afterhours": "🌙 **After-hours** — extended-hours trades shown; see the AH % column.",
        "closed": "💤 **Market closed** — showing the last session's data.",
    }.get(state, "")
    if banner:
        st.caption(banner)


def _render_table(symbols: List[str], *, notify: bool, move_thr: float) -> None:
    from market_data import build_day_trader_metrics

    try:
        rows = build_day_trader_metrics(symbols)
    except Exception as e:
        st.caption("Live data is temporarily unavailable.")
        with st.expander("Details", expanded=False):
            st.code(f"{type(e).__name__}: {e}")
        return
    if not rows:
        st.caption("No live data (market closed, Alpaca not configured, or symbols not found).")
        return

    state = market_state()

    # Movement watch: toast symbols that moved past the threshold since the
    # baseline (price at watch start / last notification).
    moved_now: set = set()
    if notify:
        current = {r["ticker"]: r["last"] for r in rows if r.get("last")}
        baseline = st.session_state.get("dt_watch_baseline") or {}
        if baseline:
            for sym, move in detect_moves(baseline, current, move_thr):
                moved_now.add(sym)
                try:
                    st.toast(f"⚡ {sym} {move:+.1f}% since you started watching")
                except Exception:
                    pass
                baseline[sym] = current[sym]  # reset so it doesn't re-toast every tick
        # First render (or newly appeared symbols): set baselines silently.
        for sym, px in current.items():
            baseline.setdefault(sym, px)
        st.session_state["dt_watch_baseline"] = baseline
    else:
        st.session_state.pop("dt_watch_baseline", None)

    df = pd.DataFrame(rows).rename(
        columns={
            "ticker": "Ticker", "last": "Last", "chg_pct": "Chg %", "gap_pct": "Gap %",
            "vwap": "VWAP", "vs_vwap_pct": "vs VWAP %", "volume": "Volume", "rvol": "RVOL",
        }
    )
    df["vs VWAP"] = df["vs VWAP %"].apply(
        lambda v: "▲ above" if (v is not None and v >= 0) else ("▼ below" if v is not None else "—")
    )
    # After-hours change: last trade vs today's official close.
    if state in ("afterhours", "closed") and "close_today" in df.columns:
        df["AH %"] = [
            round((r["Last"] / r["close_today"] - 1) * 100.0, 2)
            if (r.get("close_today") and r.get("Last") and r["Last"] != r["close_today"])
            else None
            for r in df.to_dict(orient="records")
        ]

    compact = st.checkbox("📱 Compact view", value=False, key="dt_compact")
    if compact:
        ordered = ["Ticker", "Last", "Chg %", "AH %", "vs VWAP", "RVOL"]
    else:
        ordered = [
            "Ticker", "Last", "Chg %", "AH %", "Gap %", "VWAP", "vs VWAP",
            "vs VWAP %", "RVOL", "Volume",
        ]
    df = df[[c for c in ordered if c in df.columns]]

    st.dataframe(_styled(df, moved_now), hide_index=True, width="stretch")

    # Stash for the row-action picker rendered outside the fragment.
    st.session_state["dt_rows"] = rows

    now = pd.Timestamp.utcnow().strftime("%H:%M:%S UTC")
    st.caption(
        f"As of {now} · Chg %/Gap % vs prior close · RVOL = today's volume ÷ 20-day avg · "
        "Highlight = ±3% move or 2× RVOL · IEX feed — verify before trading."
    )


def _styled(df, moved_now: set):
    def color_pct(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        return "color: #16a34a" if val >= 0 else "color: #dc2626"

    def _pct(v):
        return "—" if pd.isna(v) else f"{v:+.2f}%"

    def _price(v):
        return "—" if pd.isna(v) else f"{v:,.2f}"

    def _rvol(v):
        return "—" if pd.isna(v) else f"{v:.2f}×"

    def _vol(v):
        return "—" if pd.isna(v) else f"{int(v):,}"

    def in_play(row):
        chg = row.get("Chg %")
        rvol = row.get("RVOL")
        hot = (chg is not None and not pd.isna(chg) and abs(chg) >= HIGHLIGHT_CHG_PCT) or (
            rvol is not None and not pd.isna(rvol) and rvol >= HIGHLIGHT_RVOL
        )
        moved = row.get("Ticker") in moved_now
        style = ""
        if moved:
            style = "background-color: rgba(250, 204, 21, 0.18)"
        elif hot:
            style = "background-color: rgba(59, 130, 246, 0.12)"
        return [style] * len(row)

    fmt = {}
    for col in ("Chg %", "Gap %", "vs VWAP %", "AH %"):
        if col in df.columns:
            fmt[col] = _pct
    for col in ("Last", "VWAP"):
        if col in df.columns:
            fmt[col] = _price
    if "RVOL" in df.columns:
        fmt["RVOL"] = _rvol
    if "Volume" in df.columns:
        fmt["Volume"] = _vol

    try:
        styler = df.style.format(fmt).apply(in_play, axis=1)
        for col in ("Chg %", "Gap %", "vs VWAP %", "AH %"):
            if col in df.columns:
                styler = styler.map(color_pct, subset=[col])
        return styler
    except Exception:
        return df


def _render_row_actions() -> None:
    """Chart / trade plan / alert-me for a picked ticker (outside the fragment)."""
    rows = st.session_state.get("dt_rows") or []
    if not rows:
        return
    tickers = [r["ticker"] for r in rows if r.get("ticker")]
    if not tickers:
        return
    st.markdown("**Act on a symbol**")
    pick = st.selectbox("Ticker", tickers, key="dt_action_ticker", label_visibility="collapsed")
    row = next((r for r in rows if r.get("ticker") == pick), None)
    if not row:
        return
    a1, a2, a3 = st.columns(3)
    if a1.button("📈 Chart", key="dt_act_chart"):
        st.session_state["dt_show_chart"] = pick
    if a2.button("🎯 Trade plan", key="dt_act_plan"):
        st.session_state["dt_show_plan"] = pick
    if a3.button(f"🔔 Alert me on {pick}", key="dt_act_alert"):
        st.session_state["alert_price_tk"] = pick
        if row.get("last"):
            st.session_state["alert_price_val"] = round(float(row["last"]), 2)
        try:
            st.switch_page("pages/alerts.py")
        except Exception:
            st.caption("Open the Alerts page from the sidebar — the form is pre-filled.")

    if st.session_state.get("dt_show_chart") == pick:
        try:
            from ui.charts import render_chart_for_ticker

            render_chart_for_ticker(pick, key=f"dt_chart_{pick}")
        except Exception:
            st.caption("Chart unavailable.")
    if st.session_state.get("dt_show_plan") == pick:
        try:
            from ui.trade_plan import render_trade_plan

            plan_row: Dict[str, Any] = {"Ticker": pick, "Last": row.get("last")}
            render_trade_plan(plan_row, locked=False)
        except Exception:
            st.caption("Trade plan unavailable.")
