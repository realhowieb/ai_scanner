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


def market_state(
    now_utc: Optional[dt.datetime] = None,
    clock_is_open: Optional[bool] = None,
) -> str:
    """'premarket' | 'open' | 'afterhours' | 'closed' (US equities, ET).

    `clock_is_open` is the exchange's own answer (Alpaca /v2/clock) and, when
    provided, overrides the time-based guess for the regular session — the
    time-only logic is wrong on holidays and early-close days. Time buckets
    still classify the extended-hours windows.
    """
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    et = now.astimezone(ZoneInfo("America/New_York"))
    if clock_is_open is True:
        return "open"
    if et.weekday() >= 5:
        return "closed"
    minutes = et.hour * 60 + et.minute
    if 4 * 60 <= minutes < 9 * 60 + 30:
        return "premarket"
    if 9 * 60 + 30 <= minutes < 16 * 60:
        # Would be regular hours by the calendar — but the exchange says
        # closed (holiday) or already closed (early-close day).
        return "closed" if clock_is_open is False else "open"
    if 16 * 60 <= minutes < 20 * 60:
        return "afterhours"
    return "closed"


def _fetch_clock_is_open() -> Optional[bool]:
    """Ask Alpaca's trading clock whether the regular session is open."""
    try:
        import requests

        from data.alpaca_config import get_alpaca_config, get_alpaca_headers

        cfg = get_alpaca_config()
        headers = get_alpaca_headers()
        if not cfg or not headers:
            return None
        resp = requests.get(f"{cfg['base_url']}/v2/clock", headers=headers, timeout=5)
        if resp.status_code != 200:
            return None
        return bool((resp.json() or {}).get("is_open"))
    except Exception:
        return None


if st is not None:  # cached wrapper (the clock changes at minute granularity)

    @st.cache_data(ttl=60, show_spinner=False)
    def _clock_is_open() -> Optional[bool]:
        return _fetch_clock_is_open()

else:  # pragma: no cover
    _clock_is_open = _fetch_clock_is_open


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


_SYMBOL_RE = None


def after_hours_pct(last: Optional[float], close_today: Optional[float]) -> Optional[float]:
    """Extended-hours move: last trade vs the completed regular-session close.

    On Alpaca's IEX feed the current-day bar stops updating at the regular
    close, so after hours `dailyBar.c` is the official close and latestTrade
    carries extended-hours prints. Returns None unless both prices exist and
    genuinely differ (equal prices mean no AH trade yet — show nothing rather
    than a fake 0.00%).
    """
    try:
        if not last or not close_today:
            return None
        if float(last) == float(close_today):
            return None
        return round((float(last) / float(close_today) - 1.0) * 100.0, 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _ema_cross_display(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == "golden":
        return "Golden Cross"
    if text == "death":
        return "Death Cross"
    return "—"


def _parse_symbols(raw: str, max_symbols: int = 200) -> List[str]:
    """Parse, validate, and dedupe a comma-separated ticker list.

    Accepts 1-8 chars of A-Z / digits / dot / dash (covers class shares like
    BRK.B and BRK-B); silently drops anything else rather than sending junk to
    the quotes API.
    """
    global _SYMBOL_RE
    if _SYMBOL_RE is None:
        import re

        _SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,7}$")
    parts = [p.strip().upper() for p in (raw or "").replace("\n", ",").split(",")]
    out = [p for p in dict.fromkeys(parts) if p and _SYMBOL_RE.match(p)]
    return out[:max_symbols]


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
    except Exception as e:
        print(f"[day_trader] config import failed — panel hidden: {type(e).__name__}: {e}")
        return
    if not DAY_TRADER_ENABLED:
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
            # Clear only this page's cached fetches — st.cache_data.clear()
            # would nuke every app cache (models, history, quotes) and recreate
            # the slowness the caching work eliminated.
            for fn in ("fetch_alpaca_snapshots", "fetch_avg_daily_volume"):
                try:
                    import market_data

                    getattr(market_data, fn).clear()
                except Exception:
                    pass
            try:
                _scan_picks_cached.clear()  # type: ignore[attr-defined]
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
    notify = w1.checkbox(
        "🔔 Notify me on big moves while watching",
        key="dt_notify",
        help=(
            "This browser tab only, while it stays open — baselines reset on "
            "refresh/logout. For alerts that persist and email you, use the "
            "Alerts page."
        ),
    )
    if notify:
        w1.caption("⏱️ Session-only — for persistent alerts use the 🔔 Alerts page.")
    # Reset baselines when the watched symbol set changes, so stale entries
    # from a previous source can't produce confusing move calculations.
    _sym_sig = ",".join(sorted(symbols))
    if st.session_state.get("dt_watch_symbols") != _sym_sig:
        st.session_state["dt_watch_symbols"] = _sym_sig
        st.session_state.pop("dt_watch_baseline", None)
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
    state = market_state(clock_is_open=_clock_is_open())
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
    if len(rows) < len(symbols):
        missing = len(symbols) - len(rows)
        st.caption(
            f"⚠️ Showing {len(rows)} of {len(symbols)} symbols — {missing} "
            "returned no quote (unknown ticker or a partial data fetch)."
        )

    state = market_state(clock_is_open=_clock_is_open())

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
            "ema_cross": "EMA Cross",
        }
    )
    df["vs VWAP"] = df["vs VWAP %"].apply(
        lambda v: "▲ above" if (v is not None and v >= 0) else ("▼ below" if v is not None else "—")
    )
    if "EMA Cross" in df.columns:
        df["EMA Cross"] = df["EMA Cross"].apply(_ema_cross_display)
    # After-hours change: last trade vs today's official close. Rendered as a
    # pre-formatted STRING column ("—" when missing). Streamlit's grid shows a
    # NaN/None cell as the literal "None" regardless of the styler's formatter,
    # so the only reliable way to get an em-dash is to put the display string in
    # the data itself. Colored below by parsing the sign.
    if state in ("afterhours", "closed") and "close_today" in df.columns:
        df["AH %"] = [
            "—" if (v := after_hours_pct(r.get("Last"), r.get("close_today"))) is None
            else f"{v:+.2f}%"
            for r in df.to_dict(orient="records")
        ]

    compact = st.checkbox("📱 Compact view", value=False, key="dt_compact")
    if compact:
        ordered = ["Ticker", "Last", "Chg %", "AH %", "vs VWAP", "RVOL"]
    else:
        ordered = [
            "Ticker", "Last", "Chg %", "AH %", "Gap %", "VWAP", "vs VWAP",
            "vs VWAP %", "RVOL", "EMA Cross", "Volume",
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

    def color_pct_str(val):
        # AH % is a pre-formatted string ("+1.2%" / "-0.5%" / "—"); color by sign.
        if not isinstance(val, str):
            return ""
        if val.startswith("-"):
            return "color: #dc2626"
        if val.startswith("+"):
            return "color: #16a34a"
        return ""

    def _pct(v):
        return "—" if pd.isna(v) else f"{v:+.2f}%"

    def _price(v):
        return "—" if pd.isna(v) else f"{v:,.2f}"

    def _rvol(v):
        return "—" if pd.isna(v) else f"{v:.2f}×"

    def _vol(v):
        return "—" if pd.isna(v) else f"{int(v):,}"

    def vwap_heat(val):
        # Diverging heat: green above VWAP, red below, intensity by magnitude,
        # neutral at 0 (midpoint stays uncolored per the diverging rule).
        if val is None or (isinstance(val, float) and pd.isna(val)) or val == 0:
            return ""
        alpha = min(abs(float(val)) / 3.0, 1.0) * 0.35
        return (
            f"background-color: rgba(22, 163, 74, {alpha:.2f})"
            if val > 0
            else f"background-color: rgba(220, 38, 38, {alpha:.2f})"
        )

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
    for col in ("Chg %", "Gap %", "vs VWAP %"):  # AH % is pre-formatted to strings
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
        for col in ("Chg %", "Gap %", "vs VWAP %"):
            if col in df.columns:
                styler = styler.map(color_pct, subset=[col])
        if "AH %" in df.columns:
            styler = styler.map(color_pct_str, subset=["AH %"])
        if "vs VWAP %" in df.columns:
            styler = styler.map(vwap_heat, subset=["vs VWAP %"])
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
