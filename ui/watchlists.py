from __future__ import annotations

from typing import Callable, List, Optional, Tuple

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
    """Render the 'My Watchlists' block in the sidebar.

    Returns:
        (active_watchlist_id, active_watchlist_tickers)
    """
    st.sidebar.markdown("### 📋 My Watchlists")

    try:
        watchlists = list_watchlists(user_id)
    except Exception as e:
        st.sidebar.caption("Watchlists require Neon DB (cloud) and may be unavailable.")
        # Show the underlying error in dev so we can diagnose connection/config issues.
        with st.sidebar.expander("Watchlist error details", expanded=False):
            st.code(f"{type(e)}\n{str(e)}\n{repr(e)}")
        # For safety, don't crash the app; just return empty.
        return None, []

    active_id: Optional[int] = None
    active_tickers: List[str] = []

    if watchlists:
        options = {f"{wl['name']} (#{wl['id']})": wl for wl in watchlists}
        selected_label = st.sidebar.selectbox(
            "Active watchlist",
            list(options.keys()),
            index=0,
        )
        active = options[selected_label]
        active_id = active["id"]
        active_tickers = get_watchlist_tickers(active_id, user_id)
    else:
        st.sidebar.caption("No watchlists yet. Create one below.")

    # If there is an active watchlist with tickers, show a simple table with prices & changes
    if active_id is not None and active_tickers:
        with st.sidebar.expander("View active watchlist (with prices)", expanded=False):
            rows = []
            try:
                import yfinance as yf  # type: ignore
                for t in active_tickers:
                    sym = str(t).strip().upper()
                    price = None
                    prev_close = None
                    try:
                        ticker_obj = yf.Ticker(sym)
                        fast_info = getattr(ticker_obj, "fast_info", None)

                        # Try to get last price
                        if fast_info is not None:
                            price = getattr(fast_info, "last_price", None)
                            prev_close = getattr(fast_info, "previous_close", None)

                        # Fallbacks if fast_info is missing or incomplete
                        if price is None or prev_close is None:
                            hist = ticker_obj.history(period="2d")
                            if not hist.empty and "Close" in hist.columns:
                                closes = hist["Close"].tolist()
                                # Last close is current, prior close is previous bar if it exists
                                if len(closes) >= 1 and price is None:
                                    price = float(closes[-1])
                                if len(closes) >= 2 and prev_close is None:
                                    prev_close = float(closes[-2])

                    except Exception:
                        price = None
                        prev_close = None

                    change = None
                    change_pct = None
                    if price is not None and prev_close not in (None, 0):
                        change = float(price) - float(prev_close)
                        change_pct = (change / float(prev_close)) * 100.0

                    rows.append(
                        {
                            "Ticker": sym,
                            "Price": price,
                            "$ Change": change,
                            "% Change": change_pct,
                        }
                    )
            except Exception:
                # If yfinance or network is unavailable, still show tickers without prices
                rows = [
                    {
                        "Ticker": str(t).strip().upper(),
                        "Price": None,
                        "$ Change": None,
                        "% Change": None,
                    }
                    for t in active_tickers
                ]

            if pd is not None:
                df = pd.DataFrame(rows)
                st.dataframe(df, hide_index=True, width="stretch")
            else:
                # Fallback: simple text listing
                for row in rows:
                    price = row.get("Price")
                    change = row.get("$ Change")
                    change_pct = row.get("% Change")
                    if price is not None:
                        txt = f"{row['Ticker']}: {price:.2f}"
                        if change is not None and change_pct is not None:
                            txt += f" ({change:+.2f}, {change_pct:+.2f}%)"
                    else:
                        txt = f"{row['Ticker']}: —"
                    st.write(txt)

    with st.sidebar.expander("Manage watchlists", expanded=False):
        new_name = st.text_input("New watchlist name", key="wl_new_name")
        if st.button("Create watchlist", key="wl_create_btn"):
            if new_name.strip():
                create_watchlist(user_id, new_name.strip())
                st.rerun()
            else:
                st.warning("Please enter a name for your watchlist.")

        if active_id is not None:
            if st.button("Delete active watchlist", key="wl_delete_btn"):
                delete_watchlist(active_id, user_id)
                st.rerun()

            tickers_str = ",".join(active_tickers)
            edited = st.text_area(
                "Tickers (comma-separated)",
                value=tickers_str,
                key="wl_tickers_edit",
                help="Example: AAPL, TSLA, NVDA",
            )
            if st.button("Save tickers", key="wl_save_tickers"):
                tickers = [
                    t.strip().upper()
                    for t in edited.split(",")
                    if t.strip()
                ]
                set_watchlist_tickers(active_id, user_id, tickers)
                st.success("Watchlist updated.")
                st.rerun()
        else:
            st.caption("Create a watchlist to add tickers.")

    return active_id, active_tickers


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
    """Render center-panel watchlist controls used by the scan page."""
    watchlist_tickers = st.session_state.get("active_watchlist_tickers", []) or []
    has_watchlist = isinstance(watchlist_tickers, list) and len(watchlist_tickers) > 0

    cw1, cw2, cw3 = st.columns([1, 1, 1])
    with cw1:
        view_watchlist_btn = st.button(
            "View Watchlist",
            key="btn_view_watchlist",
            width="stretch",
            disabled=not has_watchlist,
        )
    with cw2:
        run_watchlist_btn = st.button(
            "Run Watchlist Scan",
            key="btn_scan_watchlist",
            width="stretch",
            disabled=not has_watchlist,
        )
    with cw3:
        export_csv_data = None
        if has_watchlist:
            frame_mod = pd
            if frame_mod is None:
                import pandas as frame_mod  # type: ignore
            export_csv_data = frame_mod.DataFrame({"Symbol": watchlist_tickers}).to_csv(index=False)
        st.download_button(
            "Export CSV",
            data=export_csv_data if export_csv_data is not None else "",
            file_name=f"watchlist_{len(watchlist_tickers) or 0}.csv",
            mime="text/csv",
            key="btn_export_watchlist",
            disabled=not has_watchlist,
        )

    aw1, aw2, aw3 = st.columns([3, 1, 1])
    with aw1:
        watchlist_symbol = st.text_input(
            "Add or remove ticker",
            key="watchlist_add_symbol",
            placeholder="AAPL",
            label_visibility="collapsed",
        )
    with aw2:
        add_watchlist_btn = st.button("Add", key="btn_add_watchlist_symbol", width="stretch")
    with aw3:
        remove_watchlist_btn = st.button(
            "Remove",
            key="btn_remove_watchlist_symbol",
            width="stretch",
        )

    clear_watchlist_btn = st.button(
        "Clear Watchlist",
        key="btn_clear_watchlist",
        width="stretch",
        disabled=not has_watchlist,
    )

    st.caption("Use your active watchlist for viewing, scanning, and managing symbols.")

    # 🤖 Premium: AI watchlist digest + alert preview.
    ent = st.session_state.get("entitlements") or {}
    if has_watchlist and ent.get("can_ai_notes"):
        try:
            from ui.ai import is_configured
            if is_configured():
                df = st.session_state.get("results_df")
                with st.expander("🤖 AI watchlist insights", expanded=False):
                    from ui.ai_insights import (
                        render_watchlist_alert_preview,
                        render_watchlist_digest,
                    )
                    render_watchlist_digest(watchlist_tickers, df)
                    st.divider()
                    render_watchlist_alert_preview(watchlist_tickers, df)
        except Exception:
            pass

    return (
        bool(view_watchlist_btn),
        bool(run_watchlist_btn),
        bool(clear_watchlist_btn),
        bool(add_watchlist_btn),
        bool(remove_watchlist_btn),
        str(watchlist_symbol or ""),
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
