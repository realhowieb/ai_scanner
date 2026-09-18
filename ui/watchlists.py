from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

import streamlit as st

# One or more tickers from free text: split on commas/whitespace, validate the
# shape, upper-case, and de-dupe while preserving order. Lets the Add field take
# either "AAPL" or "VLO, SKHY, KLAC" without storing the whole string as one row.
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,7}$")


def _parse_symbols(text: object, *, cap: int = 200) -> List[str]:
    seen: List[str] = []
    for tok in re.split(r"[,\s]+", str(text or "").strip().upper()):
        tok = tok.strip()
        if tok and _TICKER_RE.match(tok) and tok not in seen:
            seen.append(tok)
            if len(seen) >= cap:
                break
    return seen


def _normalize_stored_tickers(
    stored: Optional[List[str]], watchlist_id: object, username: str
) -> List[str]:
    """Flatten legacy malformed entries (e.g. a whole 'A,B,C' string saved as one
    ticker before the Add-field fix) into individual symbols, de-duped in order.

    Self-heals: when normalization actually changes the list, the cleaned
    version is written back so the garbage row disappears for good.
    """
    stored = stored or []
    cleaned: List[str] = []
    for entry in stored:
        for sym in _parse_symbols(entry):
            if sym not in cleaned:
                cleaned.append(sym)
    orig = [str(t).strip().upper() for t in stored]
    if cleaned != orig:
        try:
            set_watchlist_tickers(watchlist_id, username, cleaned)
        except Exception:
            pass  # display still uses the cleaned list even if the write fails
    return cleaned

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

from db.watchlists import (
    add_tickers_to_watchlist,
    create_watchlist,
    delete_watchlist,
    duplicate_watchlist,
    get_default_watchlist_id,
    get_watchlist_items,
    get_watchlist_tickers,
    list_watchlists,
    move_tickers_between_watchlists,
    remove_tickers_from_watchlist,
    rename_watchlist,
    set_default_watchlist,
    set_watchlist_tickers,
    update_watchlist_item_note,
)
from ui.watchlist_intelligence import render_watchlist_intelligence

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
    h1.markdown("### My Watchlists")
    active_id: Optional[int] = None
    active_tickers: List[str] = []
    if watchlists:
        default_id = next((int(wl["id"]) for wl in watchlists if wl.get("is_default")), None)
        session_id = st.session_state.get("active_watchlist_id")
        ids = [int(wl["id"]) for wl in watchlists]
        id_to_wl = {int(wl["id"]): wl for wl in watchlists}
        initial_id = session_id if session_id in ids else (default_id if default_id in ids else ids[0])
        index = ids.index(int(initial_id)) if initial_id in ids else 0

        def _label(wid: int) -> str:
            wl = id_to_wl[int(wid)]
            star = "★ " if wl.get("is_default") else ""
            suffix = " — Default" if wl.get("is_default") else ""
            return f"{star}{wl['name']} ({wl.get('symbol_count', 0)}){suffix}"

        with h2:
            selected_id = st.selectbox(
                "Active watchlist", ids, index=index, format_func=_label,
                label_visibility="collapsed",
                key="active_watchlist_selector",
            )
        active = id_to_wl[int(selected_id)]
        active_id = int(active["id"])
        active_tickers = _normalize_stored_tickers(
            get_watchlist_tickers(active_id, user_id), active_id, user_id
        )
    with h3:
        try:
            pop = st.popover("＋ New")
        except Exception:
            pop = st.expander("＋ New", expanded=False)
        with pop:
            new_name = st.text_input("Watchlist name", key="wl_new_name")
            make_default = st.checkbox("Make this my default watchlist", key="wl_new_default")
            if st.button("Create", key="wl_create_btn"):
                if new_name.strip():
                    try:
                        new_id = create_watchlist(user_id, new_name.strip(), make_default=make_default)
                        st.session_state["active_watchlist_id"] = new_id
                        st.rerun()
                    except ValueError as exc:
                        st.warning(str(exc))
                    except Exception:
                        st.error("Could not create watchlist right now.")
                else:
                    st.warning("Please enter a name.")
    if not watchlists:
        st.info("Your watchlist is empty. Create a list, then add symbols from Scanner or Stock Intelligence.")

    # The watchlist IS the visualization: live stat tiles (price + day move).
    if active_tickers:
        _render_watchlist_tiles(active_tickers)
        render_watchlist_intelligence(active_tickers)

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
        st.session_state["active_watchlist_quote_rows"] = []
        st.caption("Live quotes unavailable right now — tickers: " + ", ".join(tickers[:20]))
        return
    # Preserve the user's list order rather than the movers sort.
    by_sym = {r["ticker"]: r for r in rows}
    ordered = [by_sym[t] for t in [str(x).upper() for x in tickers] if t in by_sym]
    st.session_state["active_watchlist_quote_rows"] = ordered
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

_WATCHLIST_DF_COLUMNS = (
    "Symbol", "Name", "Last", "Change", "% Change", "Prev Close", "Open", "High", "Low",
)


def build_watchlist_df(tickers: List[str]):
    """Watchlist quote table for the center 'View Watchlist' view.

    Alpaca-first (one batched snapshot call), consistent with the rest of the
    app's data layer and fast. Falls back to the per-ticker yfinance path only
    when Alpaca returns nothing — yfinance is flaky, rate-limited, and disabled
    in some sessions, and its get_info() name lookup is slow per ticker.
    """
    frame_mod = pd
    if frame_mod is None:
        import pandas as frame_mod  # type: ignore

    syms = [str(s).strip().upper() for s in tickers if str(s).strip()]

    # --- Alpaca batched quotes (preferred) ---
    try:
        from market_data import get_latest_quotes

        quotes = get_latest_quotes(syms) or {}
    except Exception:
        quotes = {}
    if quotes:
        rows = []
        for sym in syms:
            q = quotes.get(sym) or {}
            last = q.get("last")
            prev = q.get("prev_close")
            change = change_pct = None
            if last is not None and prev not in (None, 0):
                change = float(last) - float(prev)
                change_pct = (change / float(prev)) * 100.0
            rows.append({
                "Symbol": sym, "Name": None, "Last": last, "Change": change,
                "% Change": change_pct, "Prev Close": prev,
                "Open": q.get("open"), "High": q.get("high"), "Low": q.get("low"),
            })
        return frame_mod.DataFrame(rows, columns=list(_WATCHLIST_DF_COLUMNS))

    # --- yfinance fallback (only when Alpaca returned nothing) ---
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
        scan_all = st.checkbox(
            "Score every ticker (ignore filters)",
            key="wl_scan_all",
            help="Run the models on all watchlist names regardless of the sidebar "
            "Min Gap / price / Unusual-Volume filters, so none silently drop out.",
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
            try:
                items = get_watchlist_items(active_id, st.session_state.get("username", ""))
            except Exception:
                items = [{"ticker": t, "date_added": None, "price_when_added": None, "note": None} for t in watchlist_tickers]
            export_csv_data = frame_mod.DataFrame(
                [
                    {
                        "ticker": item.get("ticker"),
                        "date_added": item.get("date_added"),
                        "price_when_added": item.get("price_when_added"),
                        "note": item.get("note"),
                    }
                    for item in items
                ]
            ).to_csv(index=False)
        st.download_button(
            "CSV", data=export_csv_data,
            file_name=f"watchlist_{len(watchlist_tickers) or 0}.csv", mime="text/csv",
            key="btn_export_watchlist", disabled=not has_watchlist, width="stretch",
        )

    # Rare/destructive operations live out of the way.
    remove_watchlist_btn = False
    clear_watchlist_btn = False
    remove_pick = ""
    with st.expander("Manage Watchlist", expanded=False):
        username = st.session_state.get("username", "")
        all_watchlists = []
        try:
            all_watchlists = list_watchlists(username)
        except Exception:
            all_watchlists = []
        id_to_name = {int(wl["id"]): str(wl["name"]) for wl in all_watchlists}
        active_name = id_to_name.get(int(active_id), "this watchlist") if active_id is not None else "this watchlist"
        default_id = None
        try:
            default_id = get_default_watchlist_id(username)
        except Exception:
            default_id = None
        if active_id is not None:
            st.caption("★ Default" if int(active_id) == int(default_id or -1) else "Not default")
            c_name, c_save = st.columns([3, 1])
            with c_name:
                renamed = st.text_input(
                    "Watchlist name",
                    value=active_name,
                    key=f"wl_rename_{active_id}_{active_name}",
                )
            with c_save:
                st.write("")
                if st.button("Rename", key=f"wl_rename_btn_{active_id}", width="stretch"):
                    try:
                        rename_watchlist(int(active_id), username, renamed)
                        st.success("Watchlist renamed.")
                        st.rerun()
                    except ValueError as exc:
                        st.warning(str(exc))
                    except Exception:
                        st.error("Could not rename watchlist right now.")
            m_default, m_duplicate = st.columns(2)
            with m_default:
                if st.button(
                    "Set as Default",
                    key=f"wl_set_default_{active_id}",
                    disabled=int(active_id) == int(default_id or -1),
                    width="stretch",
                ):
                    try:
                        set_default_watchlist(int(active_id), username)
                        st.success("Default watchlist updated.")
                        st.rerun()
                    except Exception:
                        st.error("Could not update the default watchlist.")
            with m_duplicate:
                if st.button("Duplicate", key=f"wl_duplicate_{active_id}", width="stretch"):
                    try:
                        new_id = duplicate_watchlist(int(active_id), username)
                        if new_id is not None:
                            st.session_state["active_watchlist_id"] = new_id
                        st.success("Watchlist duplicated.")
                        st.rerun()
                    except Exception:
                        st.error("Could not duplicate watchlist right now.")
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
            # Key the box by a signature of the current list so it re-seeds when
            # the list changes (e.g. after a self-heal) instead of a sticky
            # widget value pinning stale/garbage text.
            edit_sig = "-".join(watchlist_tickers)
            edited = st.text_area(
                "Bulk edit (comma-separated)",
                value=",".join(watchlist_tickers),
                key=f"wl_tickers_edit_{edit_sig}",
                help="Paste a full list to replace the watchlist contents.",
            )
            if st.button("Save list", key="wl_save_tickers"):
                # Parse splits on commas/whitespace, validates, upper-cases, and
                # de-dupes — so a pasted list with dupes can't recreate garbage.
                tickers = _parse_symbols(edited)
                set_watchlist_tickers(active_id, st.session_state.get("username", ""), tickers)
                st.success("Watchlist updated.")
                st.rerun()
            st.markdown("#### Move or copy tickers")
            selected_bulk = st.multiselect(
                "Selected tickers",
                [str(t).upper() for t in watchlist_tickers],
                key=f"wl_bulk_symbols_{active_id}",
            )
            dest_options = [
                int(wl["id"])
                for wl in all_watchlists
                if active_id is not None and int(wl["id"]) != int(active_id)
            ]
            dest_id = st.selectbox(
                "Destination watchlist",
                dest_options,
                format_func=lambda wid: id_to_name.get(int(wid), str(wid)),
                key=f"wl_bulk_dest_{active_id}",
                disabled=not dest_options,
            ) if dest_options else None
            b1, b2, b3 = st.columns(3)
            if b1.button(
                "Copy Selected",
                key=f"wl_copy_selected_{active_id}",
                disabled=not selected_bulk or dest_id is None,
                width="stretch",
            ):
                try:
                    result = add_tickers_to_watchlist(username, selected_bulk, int(dest_id))
                    st.success(
                        f"{len(result.get('added', []))} copied. "
                        f"{len(result.get('already_present', []))} already present."
                    )
                except Exception:
                    st.error("Could not copy selected symbols.")
            if b2.button(
                "Move Selected",
                key=f"wl_move_selected_{active_id}",
                disabled=not selected_bulk or dest_id is None,
                width="stretch",
            ):
                try:
                    result = move_tickers_between_watchlists(
                        username, int(active_id), int(dest_id), selected_bulk
                    )
                    st.success(f"{len(result.get('moved', []))} moved.")
                    st.rerun()
                except Exception:
                    st.error("Could not move selected symbols.")
            if b3.button(
                "Remove Selected",
                key=f"wl_remove_selected_{active_id}",
                disabled=not selected_bulk,
                width="stretch",
            ):
                try:
                    result = remove_tickers_from_watchlist(username, selected_bulk, int(active_id))
                    st.success(f"{len(result.get('removed', []))} removed.")
                    st.rerun()
                except Exception:
                    st.error("Could not remove selected symbols.")
            with st.expander("Ticker notes", expanded=False):
                try:
                    items = get_watchlist_items(int(active_id), username)
                except Exception:
                    items = []
                for item in items:
                    ticker = item["ticker"]
                    note_value = st.text_input(
                        ticker,
                        value=str(item.get("note") or ""),
                        key=f"wl_note_{active_id}_{ticker}",
                        placeholder="Optional note",
                    )
                    if note_value != str(item.get("note") or ""):
                        update_watchlist_item_note(int(active_id), username, ticker, note_value)
            clear_watchlist_btn = st.button(
                "Clear all tickers", key="btn_clear_watchlist", disabled=not has_watchlist
            )
        if active_id is not None:
            confirm_key = f"wl_delete_confirm_{active_id}"
            st.checkbox(f"Confirm delete {active_name}", key=confirm_key)
            if st.button(
                "Delete Watchlist",
                key="wl_delete_btn",
                disabled=not st.session_state.get(confirm_key),
            ):
                delete_watchlist(active_id, username)
                st.session_state["active_watchlist_id"] = None
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
        bool(scan_all),
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
    scan_all: bool = False,
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
            label = f"Watchlist ({len(tickers)} tickers)"
            if scan_all:
                label += " · all"
            do_scan(tickers, label, bypass_filters=scan_all)

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
        # Accept a single ticker or a comma/space-separated list in one go.
        syms = _parse_symbols(symbol)
        if not syms:
            banner("Please enter a ticker to add.", "warning")
        else:
            active_watchlist_id = st.session_state.get("active_watchlist_id")
            if active_watchlist_id is None:
                banner("No active watchlist selected to add to.", "warning")
            else:
                try:
                    result = add_tickers_to_watchlist(username, syms, int(active_watchlist_id))
                    added = result.get("added", [])
                    if added:
                        updated = get_watchlist_tickers(active_watchlist_id, username) or []
                        st.session_state["active_watchlist_tickers"] = list(updated)
                        if len(added) == 1:
                            banner(f"Added {added[0]} to the active watchlist.", "success")
                        else:
                            banner(f"Added {len(added)} tickers: {', '.join(added)}.", "success")
                    else:
                        banner("Those tickers are already in the active watchlist.", "info")
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
                    result = remove_tickers_from_watchlist(username, [sym], int(active_watchlist_id))
                    if result.get("removed"):
                        updated = get_watchlist_tickers(active_watchlist_id, username) or []
                        st.session_state["active_watchlist_tickers"] = list(updated)
                        banner(f"Removed {sym} from the active watchlist.", "success")
                    else:
                        banner(f"{sym} is not in the active watchlist.", "info")
                except Exception:
                    banner("Failed to update active watchlist (database unavailable).", "error")
