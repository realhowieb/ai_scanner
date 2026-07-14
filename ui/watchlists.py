from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import streamlit as st

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

from db.watchlists import (
    create_watchlist,
    delete_watchlist,
    get_watchlist_tickers,
    list_watchlists,
    set_watchlist_tickers,
)

BannerFn = Callable[[str, str], None]
ScanFn = Callable[[List[str], str], None]


def render_watchlists_panel(user_id: str) -> Tuple[Optional[int], List[str]]:
    """Render the 'My Watchlists' block: live card wall + tucked-away management.

    Returns:
        (active_watchlist_id, active_watchlist_tickers)
    """
    try:
        watchlists = list_watchlists(user_id)
    except Exception as e:
        st.markdown("### 📋 My Watchlists")
        st.caption("Watchlists require Neon DB (cloud) and may be unavailable.")
        with st.expander("Watchlist error details", expanded=False):
            st.code(f"{type(e)}\n{str(e)}\n{repr(e)}")
        return None, []

    # Header row: title · list selector · new-list popover
    h1, h2, h3 = st.columns([2, 2, 1])
    h1.markdown("### 📋 My Watchlists")
    active_id: Optional[int] = None
    active_tickers: List[str] = []
    if watchlists:
        options = {wl["name"]: wl for wl in watchlists}
        with h2:
            selected_label = st.selectbox(
                "Active watchlist", list(options.keys()), index=0,
                label_visibility="collapsed",
            )
        active = options[selected_label]
        active_id = active["id"]
        active_tickers = get_watchlist_tickers(active_id, user_id)
    with h3:
        try:
            pop = st.popover("＋ New")
        except Exception:
            pop = st.expander("＋ New", expanded=False)
        with pop:
            new_name = st.text_input("Watchlist name", key="wl_new_name")
            if st.button("Create", key="wl_create_btn"):
                if new_name.strip():
                    create_watchlist(user_id, new_name.strip())
                    st.rerun()
                else:
                    st.warning("Please enter a name.")
    if not watchlists:
        st.caption("No watchlists yet — create one with ＋ New.")

    # The watchlist IS the visualization: live stat tiles (price + day move).
    if active_tickers:
        _render_watchlist_tiles(active_tickers)
        _render_watchlist_intelligence(active_tickers)

    # Seed session state for the tools/action handler, then render the tools.
    st.session_state["active_watchlist_id"] = active_id
    st.session_state["active_watchlist_tickers"] = active_tickers
    st.session_state["_wl_tools_state"] = render_active_watchlist_tools()

    return active_id, active_tickers


def _render_watchlist_tiles(tickers: List[str], per_row: int = 5, max_tiles: int = 20) -> None:
    """Card wall: each name as a stat tile (price headline, signed day-%)."""
    try:
        from market_data import build_day_trader_metrics

        rows = build_day_trader_metrics(
            [str(t).upper() for t in tickers][:max_tiles], with_rvol=False
        )
    except Exception:
        rows = []
    if not rows:
        st.caption("Live quotes unavailable right now — tickers: " + ", ".join(tickers[:20]))
        return
    # Preserve the user's list order rather than the movers sort.
    by_sym = {r["ticker"]: r for r in rows}
    ordered = [by_sym[t] for t in [str(x).upper() for x in tickers] if t in by_sym]
    for start in range(0, len(ordered), per_row):
        chunk = ordered[start : start + per_row]
        cols = st.columns(per_row)
        for col, r in zip(cols, chunk):
            chg = r.get("chg_pct")
            col.metric(
                r["ticker"],
                f"{r['last']:,.2f}" if r.get("last") is not None else "—",
                delta=f"{chg:+.2f}%" if chg is not None else None,
            )
    if len(tickers) > max_tiles:
        st.caption(f"Showing the first {max_tiles} of {len(tickers)} names.")


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _symbol_key(row: Any) -> str:
    for col in ("Ticker", "Symbol", "ticker", "symbol"):
        try:
            value = row.get(col)
        except AttributeError:
            value = None
        if str(value or "").strip():
            return str(value).strip().upper()
    return ""


def classify_watchlist_signal(row: Any) -> tuple[str, str]:
    """Return a compact signal label and reason for a watchlist row."""
    pre = _to_float(row.get("PreBreakoutProb%") if hasattr(row, "get") else None)
    ai_conf = _to_float(row.get("AI Confidence") if hasattr(row, "get") else None)
    score = _to_float(row.get("BreakoutScore") if hasattr(row, "get") else None)
    is_breakout = _to_bool(row.get("IsBreakout")) if hasattr(row, "get") else False

    if is_breakout or (ai_conf is not None and ai_conf >= 70):
        return "Active breakout", "AI confidence or breakout flag is high."
    if pre is not None and pre >= 60:
        return "Heating up", "Pre-breakout probability is elevated."
    if score is not None and score >= 20:
        return "Strong setup", "Breakout score is elevated."
    if pre is not None and pre <= 5 and ai_conf is not None and ai_conf <= 5:
        return "Cooling down", "Both model signals are quiet."
    return "Watching", "No strong signal in the latest scan yet."


def summarize_watchlist_intelligence(tickers: List[str], results_df: Any) -> list[dict[str, Any]]:
    """Match watchlist symbols against the latest scan results."""
    if results_df is None or not hasattr(results_df, "iterrows"):
        return []
    watch = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not watch:
        return []
    rows_by_symbol: dict[str, Any] = {}
    try:
        iterator = results_df.iterrows()
    except Exception:
        return []
    for _, row in iterator:
        sym = _symbol_key(row)
        if sym and sym not in rows_by_symbol:
            rows_by_symbol[sym] = row

    out: list[dict[str, Any]] = []
    for sym in watch:
        row = rows_by_symbol.get(sym)
        if row is None:
            out.append(
                {
                    "Ticker": sym,
                    "Signal": "Not in latest scan",
                    "Reason": "Run a scan including this ticker.",
                    "PreBreakout": None,
                    "AI Confidence": None,
                    "Score": None,
                }
            )
            continue
        signal, reason = classify_watchlist_signal(row)
        out.append(
            {
                "Ticker": sym,
                "Signal": signal,
                "Reason": reason,
                "PreBreakout": _to_float(row.get("PreBreakoutProb%")),
                "AI Confidence": _to_float(row.get("AI Confidence")),
                "Score": _to_float(row.get("BreakoutScore")),
            }
        )
    priority = {"Active breakout": 0, "Heating up": 1, "Strong setup": 2, "Watching": 3, "Cooling down": 4, "Not in latest scan": 5}
    return sorted(out, key=lambda item: (priority.get(str(item["Signal"]), 9), str(item["Ticker"])))


def _render_watchlist_intelligence(tickers: List[str]) -> None:
    df = st.session_state.get("results_df")
    rows = summarize_watchlist_intelligence(tickers, df)
    if not rows:
        return

    with st.expander("Watchlist intelligence", expanded=False):
        hot = sum(1 for row in rows if row["Signal"] in {"Active breakout", "Heating up", "Strong setup"})
        cold = sum(1 for row in rows if row["Signal"] == "Cooling down")
        missing = sum(1 for row in rows if row["Signal"] == "Not in latest scan")
        c1, c2, c3 = st.columns(3)
        c1.metric("Actionable", hot)
        c2.metric("Cooling", cold)
        c3.metric("Not scanned", missing)
        try:
            frame_mod = pd
            if frame_mod is None:
                import pandas as frame_mod  # type: ignore
            table = frame_mod.DataFrame(rows)
            st.dataframe(
                table,
                width="stretch",
                hide_index=True,
                column_config={
                    "PreBreakout": st.column_config.NumberColumn("PreBreakout", format="%.1f%%"),
                    "AI Confidence": st.column_config.NumberColumn(format="%.1f%%"),
                    "Score": st.column_config.NumberColumn(format="%.1f"),
                },
            )
        except Exception:
            st.write(rows)


def build_watchlist_df(tickers: List[str]):
    """Build a rich watchlist DataFrame for the center 'View Watchlist' table."""
    frame_mod = pd
    if frame_mod is None:
        import pandas as frame_mod  # type: ignore

    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return frame_mod.DataFrame(
            [
                {
                    "Symbol": str(sym).strip().upper(),
                    "Name": None,
                    "Last": None,
                    "Change": None,
                    "% Change": None,
                    "Prev Close": None,
                    "Open": None,
                    "High": None,
                    "Low": None,
                }
                for sym in tickers
            ]
        )

    rows = []
    for sym in tickers:
        sym = str(sym).strip().upper()
        last = prev_close = open_ = high = low = None
        name = None

        try:
            t = yf.Ticker(sym)
            fast = getattr(t, "fast_info", None)

            if fast is not None:
                last = (
                    getattr(fast, "last_price", None)
                    or getattr(fast, "regular_market_price", None)
                )
                prev_close = getattr(fast, "previous_close", None)
                open_ = getattr(fast, "open", None)
                high = getattr(fast, "day_high", None)
                low = getattr(fast, "day_low", None)

            if last is None or prev_close is None:
                hist = t.history(period="2d")
                if not hist.empty and "Close" in hist.columns:
                    closes = hist["Close"].tolist()
                    if len(closes) >= 1 and last is None:
                        last = float(closes[-1])
                    if len(closes) >= 2 and prev_close is None:
                        prev_close = float(closes[-2])

            try:
                info = t.get_info() if hasattr(t, "get_info") else getattr(t, "info", {})
            except Exception:
                info = {}
            name = info.get("shortName") or info.get("longName") or ""
        except Exception:
            pass

        change = None
        change_pct = None
        if last is not None and prev_close not in (None, 0):
            change = float(last) - float(prev_close)
            change_pct = (change / float(prev_close)) * 100.0

        rows.append(
            {
                "Symbol": sym,
                "Name": name,
                "Last": last,
                "Change": change,
                "% Change": change_pct,
                "Prev Close": prev_close,
                "Open": open_,
                "High": high,
                "Low": low,
            }
        )

    return frame_mod.DataFrame(rows)


def _active_watchlist_tickers() -> List[str]:
    tickers = st.session_state.get("active_watchlist_tickers", []) or []
    return [str(t).strip().upper() for t in tickers if str(t).strip()]


def render_active_watchlist_tools() -> tuple[bool, bool, bool, bool, bool, str]:
    """Watchlist actions: one add path up front, destructive ops inside Manage.

    Returns the same (view, run, clear, add, remove, symbol) tuple the scan
    page's action handler consumes; `symbol` carries the add-input text or the
    remove-picker choice depending on which button fired.
    """
    watchlist_tickers = st.session_state.get("active_watchlist_tickers", []) or []
    has_watchlist = isinstance(watchlist_tickers, list) and len(watchlist_tickers) > 0
    active_id = st.session_state.get("active_watchlist_id")

    # Primary action row: add · scan · view · export
    a1, a2, a3, a4, a5 = st.columns([2, 1, 2, 1.4, 1])
    with a1:
        add_text = st.text_input(
            "Add ticker", key="watchlist_add_symbol", placeholder="＋ Add ticker (e.g. AAPL)",
            label_visibility="collapsed",
        )
    with a2:
        add_watchlist_btn = st.button(
            "Add", key="btn_add_watchlist_symbol", width="stretch",
            disabled=active_id is None,
        )
    with a3:
        run_watchlist_btn = st.button(
            "🔎 Run Watchlist Scan", key="btn_scan_watchlist", width="stretch",
            disabled=not has_watchlist,
        )
    with a4:
        view_watchlist_btn = st.button(
            "View as table", key="btn_view_watchlist", width="stretch",
            disabled=not has_watchlist,
        )
    with a5:
        export_csv_data = ""
        if has_watchlist:
            frame_mod = pd
            if frame_mod is None:
                import pandas as frame_mod  # type: ignore
            export_csv_data = frame_mod.DataFrame({"Symbol": watchlist_tickers}).to_csv(index=False)
        st.download_button(
            "CSV", data=export_csv_data,
            file_name=f"watchlist_{len(watchlist_tickers) or 0}.csv", mime="text/csv",
            key="btn_export_watchlist", disabled=not has_watchlist, width="stretch",
        )

    # Rare/destructive operations live out of the way.
    remove_watchlist_btn = False
    clear_watchlist_btn = False
    remove_pick = ""
    with st.expander("⚙️ Manage", expanded=False):
        if has_watchlist:
            m1, m2 = st.columns([3, 1])
            with m1:
                remove_pick = st.selectbox(
                    "Remove a ticker",
                    [""] + [str(t).upper() for t in watchlist_tickers],
                    key="wl_remove_pick",
                )
            with m2:
                st.write("")
                remove_watchlist_btn = st.button(
                    "Remove", key="btn_remove_watchlist_symbol", width="stretch",
                    disabled=not remove_pick,
                )
            edited = st.text_area(
                "Bulk edit (comma-separated)",
                value=",".join(watchlist_tickers),
                key="wl_tickers_edit",
                help="Paste a full list to replace the watchlist contents.",
            )
            if st.button("Save list", key="wl_save_tickers"):
                tickers = [t.strip().upper() for t in edited.split(",") if t.strip()]
                set_watchlist_tickers(active_id, st.session_state.get("username", ""), tickers)
                st.success("Watchlist updated.")
                st.rerun()
            clear_watchlist_btn = st.button(
                "Clear all tickers", key="btn_clear_watchlist", disabled=not has_watchlist
            )
        if active_id is not None:
            if st.button("🗑️ Delete this watchlist", key="wl_delete_btn"):
                delete_watchlist(active_id, st.session_state.get("username", ""))
                st.rerun()
        if not has_watchlist and active_id is None:
            st.caption("Create a watchlist to manage tickers.")

    # 🧠 Premium: AI watchlist digest + alert preview.
    ent = st.session_state.get("entitlements") or {}
    if has_watchlist and ent.get("can_ai_notes"):
        try:
            from ui.ai import is_configured
            if is_configured():
                df = st.session_state.get("results_df")
                with st.expander("🧠 AI watchlist insights", expanded=False):
                    from ui.ai_insights import (
                        render_watchlist_alert_preview,
                        render_watchlist_digest,
                    )
                    render_watchlist_digest(watchlist_tickers, df)
                    st.divider()
                    render_watchlist_alert_preview(watchlist_tickers, df)
        except Exception:
            pass

    # One `symbol` slot serves both actions: the remove picker wins when its
    # button fired, otherwise the add input.
    symbol = remove_pick if remove_watchlist_btn else (add_text or "")
    return (
        bool(view_watchlist_btn),
        bool(run_watchlist_btn),
        bool(clear_watchlist_btn),
        bool(add_watchlist_btn),
        bool(remove_watchlist_btn),
        str(symbol or ""),
    )


def handle_active_watchlist_actions(
    *,
    view_watchlist: bool,
    run_watchlist: bool,
    clear_watchlist: bool,
    add_symbol: bool,
    remove_symbol: bool,
    symbol: str,
    username: str,
    do_scan: ScanFn,
    banner: BannerFn,
) -> None:
    """Handle center-panel watchlist actions used by the scan page."""
    if view_watchlist:
        tickers = _active_watchlist_tickers()
        if not tickers:
            banner("Active watchlist has no tickers to view.", "warning")
        else:
            st.session_state.results_df = build_watchlist_df(tickers)
            st.session_state["force_results_refresh"] = True
            banner(
                f"Showing active watchlist with {len(tickers)} tickers (with prices & daily change).",
                "info",
            )

    if run_watchlist:
        tickers = _active_watchlist_tickers()
        if not tickers:
            banner("Active watchlist has no tickers to scan.", "warning")
        else:
            do_scan(tickers, f"Watchlist ({len(tickers)} tickers)")

    if clear_watchlist:
        active_watchlist_id = st.session_state.get("active_watchlist_id")
        if active_watchlist_id is None:
            banner("No active watchlist selected to clear.", "warning")
        else:
            try:
                set_watchlist_tickers(active_watchlist_id, username, [])
                st.session_state["active_watchlist_tickers"] = []
                banner("Cleared all tickers from the active watchlist.", "success")
            except Exception:
                banner("Failed to clear active watchlist (database unavailable).", "error")

    sym = str(symbol or "").strip().upper()
    if add_symbol:
        if not sym:
            banner("Please enter a ticker to add.", "warning")
        else:
            active_watchlist_id = st.session_state.get("active_watchlist_id")
            if active_watchlist_id is None:
                banner("No active watchlist selected to add to.", "warning")
            else:
                try:
                    existing = get_watchlist_tickers(active_watchlist_id, username) or []
                    norm_existing = {str(t).strip().upper() for t in existing}
                    if sym not in norm_existing:
                        updated = sorted(norm_existing | {sym})
                        set_watchlist_tickers(active_watchlist_id, username, list(updated))
                        st.session_state["active_watchlist_tickers"] = list(updated)
                        banner(f"Added {sym} to the active watchlist.", "success")
                    else:
                        banner(f"{sym} is already in the active watchlist.", "info")
                except Exception:
                    banner("Failed to update active watchlist (database unavailable).", "error")

    if remove_symbol:
        if not sym:
            banner("Please enter a ticker to remove.", "warning")
        else:
            active_watchlist_id = st.session_state.get("active_watchlist_id")
            if active_watchlist_id is None:
                banner("No active watchlist selected to remove from.", "warning")
            else:
                try:
                    existing = get_watchlist_tickers(active_watchlist_id, username) or []
                    norm_existing = {str(t).strip().upper() for t in existing}
                    if sym in norm_existing:
                        updated = sorted(norm_existing - {sym})
                        set_watchlist_tickers(active_watchlist_id, username, list(updated))
                        st.session_state["active_watchlist_tickers"] = list(updated)
                        banner(f"Removed {sym} from the active watchlist.", "success")
                    else:
                        banner(f"{sym} is not in the active watchlist.", "info")
                except Exception:
                    banner("Failed to update active watchlist (database unavailable).", "error")
