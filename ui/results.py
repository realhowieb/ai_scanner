"""Results, charts, and AI notes UI module."""

from typing import Callable, Optional

import logging
import re


import pandas as pd
import streamlit as st


# --- yfinance / Yahoo hard-fail guard ---
# Yahoo sometimes returns 401 "Invalid Crumb" which can spam logs and stall UI.
# When detected, we disable further yfinance calls for the current session.
_YF_DISABLED_KEY = "yf_disabled"
_YF_DISABLED_REASON_KEY = "yf_disabled_reason"
_YF_WARNED_KEY = "yf_disabled_warned"

# Reduce noisy library logging (won't hide exceptions, just prevents log spam).
for _name in ("yfinance", "urllib3", "requests"):
    try:
        logging.getLogger(_name).setLevel(logging.ERROR)
    except Exception:
        pass


def _is_yahoo_crumb_error(exc: Exception) -> bool:
    msg = str(exc) or ""
    msg_l = msg.lower()
    # Common yfinance/Yahoo failure strings
    if "invalid crumb" in msg_l:
        return True
    if "http error 401" in msg_l or "401" in msg_l and "unauthorized" in msg_l:
        return True
    return False


def _disable_yfinance_for_session(reason: str) -> None:
    try:
        st.session_state[_YF_DISABLED_KEY] = True
        st.session_state[_YF_DISABLED_REASON_KEY] = reason
    except Exception:
        pass


def _warn_yfinance_disabled_once() -> None:
    try:
        if st.session_state.get(_YF_WARNED_KEY):
            return
        if st.session_state.get(_YF_DISABLED_KEY):
            st.session_state[_YF_WARNED_KEY] = True
            reason = st.session_state.get(_YF_DISABLED_REASON_KEY) or "Yahoo Finance blocked the request (401)."
            st.caption(f"⚠️ Live quotes temporarily disabled this session: {reason}")
    except Exception:
        pass


def get_results_df() -> Optional[pd.DataFrame]:
    """Return the current results DataFrame from session_state.

    If none exists yet, return None.
    """
    return st.session_state.get("results_df")


# Row selection helper for interactive tables
def _sync_selected_ticker_from_table(selection_obj: object, df: pd.DataFrame, picker_key: str) -> None:
    """If Streamlit dataframe selection is available, sync selected ticker into session_state.

    This keeps row-click selection consistent with the existing chart picker keys.
    """
    try:
        rows = getattr(getattr(selection_obj, "selection", None), "rows", None)
        if not rows:
            return
        idx = int(rows[0])
        if idx < 0 or idx >= len(df):
            return
        if "Ticker" not in df.columns:
            return
        t = str(df.iloc[idx]["Ticker"]).strip().upper()
        if not t:
            return
        st.session_state["results_selected_ticker"] = t
        # Also sync chart picker so existing chart/AI notes follow the click
        st.session_state[picker_key] = t
    except Exception:
        return


def render_results(
    df: Optional[pd.DataFrame],
    can_export_csv: bool,
    can_ai_notes: bool,
    render_chart_for_ticker: Callable[[str], None],
    generate_ai_note: Callable[[pd.Series], str],
) -> None:
    """Render the results table, chart picker, and AI Notes section.

    Args:
        df: The results DataFrame (or None/empty).
        can_export_csv: Whether the current tier can export CSV.
        can_ai_notes: Whether the current tier can use AI Notes.
        render_chart_for_ticker: Callback to render a chart for a single ticker.
        generate_ai_note: Callback to generate an AI note for a single row.
    """
    if df is None or df.empty:
        st.caption("Run a scan to see results.")
        return

    # Centralized entitlements (preferred). If present, they override passed flags.
    ent = st.session_state.get("entitlements") or {}
    if ent:
        can_export_csv = bool(ent.get("can_export_csv", can_export_csv))
        # Treat AI Notes as Premium-only; fall back to passed flag if not present.
        can_ai_notes = bool(ent.get("can_ai_notes", can_ai_notes))

    # Option A: Basic = auto-details only, no selection
    is_basic = not can_export_csv

    def _auto_details_ticker(_df: pd.DataFrame) -> str | None:
        try:
            if _df is None or _df.empty or "Ticker" not in _df.columns:
                return None
            if "BreakoutScore" in _df.columns:
                s = pd.to_numeric(_df["BreakoutScore"], errors="coerce")
                if s.notna().any():
                    i = int(s.fillna(-1e18).idxmax())
                    t = str(_df.loc[i, "Ticker"]).strip().upper()
                    return t or None
            # fallback: first row
            t = str(_df.iloc[0]["Ticker"]).strip().upper()
            return t or None
        except Exception:
            return None

    st.subheader("Results")
    st.caption(
        f"Showing {len(df)} results. Increase 'Top N Results' in the sidebar to see more, "
        "or relax filters (Min Gap %, price range, Unusual Volume Filter). "
        "If you see 0 results, try lowering Min Gap or turning off the Unusual Volume Filter."
    )

    # ─────────────────────────────
    # 📅 Earnings Filters (fast)
    # Requires a precomputed column from app.py: "📅 Earnings in X days"
    # ─────────────────────────────
    earn_col = "📅 Earnings in X days"
    if earn_col in df.columns:
        with st.expander("📅 Earnings Filters", expanded=False):
            excl_3 = st.checkbox(
                "Exclude earnings in next 3 days",
                value=False,
                key="earn_excl_3_results",
                help="Hides stocks with earnings 0–3 days away (keeps unknown earnings).",
            )
            within_7 = st.checkbox(
                "Only earnings within 7 days",
                value=False,
                key="earn_within_7_results",
                help="Shows only stocks with earnings 0–7 days away.",
            )

            s = pd.to_numeric(df[earn_col], errors="coerce")
            before = len(df)

            if excl_3:
                df = df[s.isna() | (s > 3)]
                s = pd.to_numeric(df[earn_col], errors="coerce")

            if within_7:
                df = df[(s >= 0) & (s <= 7)]

            after = len(df)
            if before != after:
                st.caption(f"Filtered by earnings: {before} → {after} rows")

    # --- Watchlist action (used in ticker details panel) ---
    def _render_watchlist_action(ticker: str) -> None:
        t = (ticker or "").strip().upper()
        if not t:
            return

        active_id = st.session_state.get("active_watchlist_id")
        username = st.session_state.get("username") or st.session_state.get("user") or "anonymous"

        # Session copy (best UX), but DB is source of truth if available
        current = st.session_state.get("active_watchlist_tickers", []) or []
        current_norm = {str(x).strip().upper() for x in current if str(x).strip()}

        if active_id is None:
            st.caption("📋 Watchlist: select an active watchlist to add tickers.")
            return

        already = t in current_norm

        cA, cB = st.columns([1, 2])
        with cA:
            clicked = st.button(
                "⭐ Add to Watchlist" if not already else "✅ In Watchlist",
                key=f"btn_details_add_watchlist_{t}",
                disabled=already,
                use_container_width=True,
            )
        with cB:
            st.caption("Adds this ticker to your active watchlist.")

        if not clicked:
            return

        # Resolve DB functions (app may expose them via different module paths)
        def _resolve_watchlist_fns():
            # 1) If app.py injected them into globals() at import time
            g_get = globals().get("get_watchlist_tickers")
            g_set = globals().get("set_watchlist_tickers")
            if callable(g_get) and callable(g_set):
                return g_get, g_set

            # 2) Try common import locations
            candidates = [
                ("db.watchlists", "get_watchlist_tickers", "set_watchlist_tickers"),
                ("db.watchlist", "get_watchlist_tickers", "set_watchlist_tickers"),
                ("watchlists", "get_watchlist_tickers", "set_watchlist_tickers"),
                ("watchlist", "get_watchlist_tickers", "set_watchlist_tickers"),
            ]
            for mod_name, get_name, set_name in candidates:
                try:
                    mod = __import__(mod_name, fromlist=[get_name, set_name])
                    get_fn = getattr(mod, get_name, None)
                    set_fn = getattr(mod, set_name, None)
                    if callable(get_fn) and callable(set_fn):
                        return get_fn, set_fn
                except Exception:
                    continue

            return None, None

        get_fn, set_fn = _resolve_watchlist_fns()

        # Always update session_state immediately for responsive UX
        updated_norm = set(current_norm)
        updated_norm.add(t)
        st.session_state["active_watchlist_tickers"] = sorted(updated_norm)

        # If DB functions exist, persist + verify
        if callable(get_fn) and callable(set_fn):
            try:
                existing_db = get_fn(active_id, username) or []
                existing_db_norm = {str(x).strip().upper() for x in existing_db if str(x).strip()}
                new_db = sorted(existing_db_norm | {t})

                set_fn(active_id, username, list(new_db))

                # Verify (read-after-write) so we don't show a false success
                verify_db = get_fn(active_id, username) or []
                verify_norm = {str(x).strip().upper() for x in verify_db if str(x).strip()}
                st.session_state["active_watchlist_tickers"] = sorted(verify_norm)

                if t in verify_norm:
                    st.success(f"Added **{t}** to your active watchlist.")
                else:
                    st.warning(
                        f"Tried to add **{t}**, but it did not appear in the DB on verification. "
                        "Your DB write may be failing or the watchlist is keyed differently."
                    )
            except Exception as e:
                # Keep the session_state add (UX), but make DB failure visible
                st.warning(
                    f"Added **{t}** locally, but DB save failed: {e}. "
                    "It may disappear after refresh if DB cannot be reached."
                )
        else:
            # No DB integration available in this runtime
            st.info(
                "Added locally (no DB watchlist functions found). "
                "If it disappears after refresh, wire get_watchlist_tickers/set_watchlist_tickers into this module."
            )

    # --- Performance guard: Pandas Styler becomes very slow on large tables ---
    MAX_STYLED_ROWS = 1500
    MAX_STYLED_COLS = 25

    # "Fast mode" disables ALL pandas Styler work (df.style / applymap / gradients / to_html).
    # Even medium-sized tables can feel slow on Streamlit Cloud, so we also provide a manual toggle.
    auto_fast = (len(df) > MAX_STYLED_ROWS) or (df.shape[1] > MAX_STYLED_COLS)

    # Styling must be OPT-IN. Even ~50–200 rows can feel slow with multiple Styler passes.
    STYLE_ROW_LIMIT = 40

    default_enable_style = False

    enable_styling = st.checkbox(
        "🎨 Enable table styling (slower)",
        value=default_enable_style and (not auto_fast),
        help="Styling can be slow even on medium tables. Leave off for the fastest results.",
        key="results_enable_styling",
    )

    # If user enables styling but the table is beyond the safe limit, force fast mode.
    if enable_styling and len(df) > STYLE_ROW_LIMIT:
        st.caption(
            f"⚡ Styling auto-disabled for {len(df):,} rows (limit={STYLE_ROW_LIMIT}). "
            "Lower Top N Results to re-enable styling."
        )
        enable_styling = False

    fast_mode = auto_fast or (not enable_styling)

    if fast_mode:
        if auto_fast:
            st.caption(
                f"⚡ Fast mode enabled (styling disabled) — {len(df):,} rows × {df.shape[1]} cols. "
                f"Refine filters / lower Top N Results to re-enable styling."
            )
        else:
            st.caption("⚡ Fast mode enabled (styling disabled) for faster rendering.")

        # Render without Styler for speed
        if can_export_csv:
            _tbl = st.dataframe(
                df,
                use_container_width=True,
                height=420,
                selection_mode="single-row",
                on_select="rerun",
                key="results_table_fast",
            )
            _sync_selected_ticker_from_table(_tbl, df, picker_key="results_chart_picker_fast")
        else:
            # Basic: keep non-interactive rendering. Use plain HTML (much faster than styled.to_html).
            try:
                table_html = df.to_html(index=False)

                st.markdown(
                    """
<style>
.basic-results-wrap {
  max-height: 420px;
  overflow-x: auto;
  overflow-y: auto;
  border: 1px solid rgba(49, 51, 63, 0.25);
  border-radius: 10px;
  padding: 6px;
}

/* Prevent vertical letter stacking on mobile */
.basic-results-wrap table {
  width: max-content;
  min-width: 100%;
  border-collapse: collapse;
}

.basic-results-wrap th,
.basic-results-wrap td {
  white-space: nowrap;
  padding: 6px 10px;
}

/* Sticky header */
.basic-results-wrap th {
  position: sticky;
  top: 0;
  background: rgba(15, 17, 22, 0.98);
  z-index: 2;
}
</style>
""",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='basic-results-wrap'>{table_html}</div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                # Fallback: still non-interactive
                try:
                    st.table(df)
                except Exception:
                    st.markdown(df.to_html(index=False), unsafe_allow_html=True)

        # Export (tier-gated) still available even in fast mode
        if can_export_csv:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=csv,
                file_name="breakout_results.csv",
                mime="text/csv",
                use_container_width=False,
            )
        else:
            st.info("🔒 Pro feature — export scan results to CSV")

        # Continue with charts / details / AI notes
        if is_basic:
            # Basic: no interactive selection; show one auto-selected ticker details.
            auto_t = _auto_details_ticker(df)
            if auto_t:
                with st.expander(f"📌 {auto_t} details", expanded=False):
                    st.caption("📌 Top breakout candidate (auto-selected). 🔒 Upgrade to Pro to explore other tickers.")
                    try:
                        row_df = df[df["Ticker"].astype(str).str.upper() == str(auto_t).upper()]
                        r0 = row_df.iloc[0] if len(row_df) else None
                    except Exception:
                        r0 = None

                    if r0 is not None:
                        c1, c2, c3, c4 = st.columns(4)

                        def _as_float(v):
                            try:
                                if pd.isna(v):
                                    return None
                                return float(v)
                            except Exception:
                                return None

                        bs = _as_float(r0.get("BreakoutScore"))
                        last = _as_float(r0.get("Last"))
                        gap = _as_float(r0.get("GapPct"))
                        dv = _as_float(r0.get("DollarVol20"))

                        c1.metric("BreakoutScore", "—" if bs is None else f"{bs:.2f}")
                        c2.metric("Last", "—" if last is None else f"{last:.2f}")
                        c3.metric("Gap%", "—" if gap is None else f"{gap:.2f}%")
                        c4.metric("$Vol20", "—" if dv is None else f"{dv:,.0f}")

                        earn_days = _as_float(r0.get("📅 Earnings in X days")) if "📅 Earnings in X days" in df.columns else None
                        if earn_days is None:
                            st.caption("📅 Earnings: —")
                        else:
                            st.caption(f"📅 Earnings in {int(earn_days)} days")

                        with st.expander("Show row fields", expanded=False):
                            st.json({k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in r0.to_dict().items()})
                    else:
                        st.caption("No row details available for this ticker.")
            else:
                st.caption("No ticker details available.")

            # Basic: keep AI notes locked as before
            if can_ai_notes:
                st.subheader("AI Notes")
                st.caption("⭐ Premium feature")
                st.caption("AI notes require a selectable ticker; upgrade to Pro/Premium.")
            else:
                st.info("🔒 Premium feature — AI-powered notes for the selected ticker")

            return

        # Pro/Premium: keep charts + interactive details
        with st.expander("📈 Charts", expanded=False):
            tickers = df["Ticker"].tolist() if "Ticker" in df.columns else []
            if not tickers:
                st.caption("No tickers available to chart.")
                return

            pick = st.selectbox("Select ticker to chart", tickers, key="results_chart_picker_fast")

            # Render chart ONLY when expander is opened
            render_chart_for_ticker(pick)

        pick = st.session_state.get("results_chart_picker_fast")

        # Detail panel (row-click driven)
        selected_ticker = st.session_state.get("results_selected_ticker") or pick
        if selected_ticker:
            with st.expander(f"📌 {selected_ticker} details", expanded=False):
                # Show a compact stats + earnings card (no extra network calls)
                try:
                    row_df = df[df["Ticker"].astype(str).str.upper() == str(selected_ticker).upper()]
                    r0 = row_df.iloc[0] if len(row_df) else None
                except Exception:
                    r0 = None

                if r0 is not None:
                    c1, c2, c3, c4 = st.columns(4)

                    def _as_float(v):
                        try:
                            if pd.isna(v):
                                return None
                            return float(v)
                        except Exception:
                            return None

                    # Core stats
                    bs = _as_float(r0.get("BreakoutScore"))
                    last = _as_float(r0.get("Last"))
                    gap = _as_float(r0.get("GapPct"))
                    dv = _as_float(r0.get("DollarVol20"))

                    c1.metric("BreakoutScore", "—" if bs is None else f"{bs:.2f}")
                    c2.metric("Last", "—" if last is None else f"{last:.2f}")
                    c3.metric("Gap%", "—" if gap is None else f"{gap:.2f}%")
                    c4.metric("$Vol20", "—" if dv is None else f"{dv:,.0f}")

                    # Earnings context (if present)
                    earn_days = _as_float(r0.get("📅 Earnings in X days")) if "📅 Earnings in X days" in df.columns else None
                    if earn_days is None:
                        st.caption("📅 Earnings: —")
                    else:
                        st.caption(f"📅 Earnings in {int(earn_days)} days")

                    # ⭐ Add to watchlist action
                    _render_watchlist_action(str(selected_ticker))

                    # Optional: show a tiny raw row preview for debugging (collapsed)
                    with st.expander("Show row fields", expanded=False):
                        st.json({k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in r0.to_dict().items()})
                else:
                    st.caption("No row details available for this ticker.")

        if can_ai_notes:
            st.subheader("AI Notes")
            st.caption("⭐ Premium feature")
            try:
                row = df[df["Ticker"] == pick].iloc[0]
                auto_note = generate_ai_note(row)
                st.markdown(auto_note)
                st.text_area(
                    "Edit or copy these notes (Premium only):",
                    value=auto_note,
                    height=220,
                )
            except Exception:
                st.caption("AI notes are unavailable for the selected row.")
        else:
            st.info("🔒 Premium feature — AI-powered notes for the selected ticker")

        return

    # --- UI polish: Earnings column (display-only) ---
    earn_col = "📅 Earnings in X days"

    # Move earnings column next to Ticker (if present)
    if earn_col in df.columns and "Ticker" in df.columns:
        try:
            cols = list(df.columns)
            cols.remove(earn_col)
            ticker_idx = cols.index("Ticker")
            cols.insert(ticker_idx + 1, earn_col)
            df = df[cols]
        except Exception:
            pass

    # --- Pro styling for results table ---
    styled = df.style

    # Format earnings column: None/NaN -> — ; ints shown as whole numbers
    if "📅 Earnings in X days" in df.columns:
        styled = styled.format(
            {"📅 Earnings in X days": (lambda v: "—" if pd.isna(v) else int(float(v)))}
        )

        def _earnings_style(v):
            try:
                if pd.isna(v):
                    return ""
                d = int(float(v))
            except Exception:
                return ""

            # Warning for earnings very soon
            if 0 <= d <= 3:
                return "background-color: #FFF3CD; color: #856404; font-weight: 700;"
            # Soft info for 4-7 days
            if 4 <= d <= 7:
                return "background-color: #E8F4FD; color: #0C5460;"
            return ""

        styled = styled.applymap(_earnings_style, subset=["📅 Earnings in X days"])

    # Heatmap for BreakoutScore
    if "BreakoutScore" in df.columns:
        styled = styled.background_gradient(axis=None, cmap="RdYlGn", subset=["BreakoutScore"])

    # Conditional formatting for RS_Rank (0-100)
    if "RS_Rank" in df.columns:
        styled = styled.background_gradient(axis=None, cmap="Greens", subset=["RS_Rank"])

    # Bold / color trend markers
    def _trend_style(series: pd.Series):
        styles = []
        for v in series:
            try:
                val = float(v)
            except Exception:
                styles.append("")
                continue
            if val >= 20:
                styles.append("font-weight: bold; color: #006400;")  # strong uptrend
            elif val <= -10:
                styles.append("font-weight: bold; color: #8B0000;")  # strong downtrend
            else:
                styles.append("")
        return styles

    if "Trend20D%" in df.columns:
        styled = styled.apply(_trend_style, subset=["Trend20D%"])
    if "Trend10D%" in df.columns:
        styled = styled.apply(_trend_style, subset=["Trend10D%"])

    # Watchlist-style numeric formatting (Symbol/Last/Change/% Change/etc.)
    watchlist_cols = {"Last", "Change", "% Change", "Prev Close", "Open", "High", "Low"}
    if watchlist_cols.intersection(df.columns):
        # Define per-column formatters
        def _fmt_price(x):
            try:
                return f"{float(x):.2f}"
            except Exception:
                return x

        def _fmt_change(x):
            try:
                return f"{float(x):+,.2f}"
            except Exception:
                return x

        def _fmt_pct(x):
            try:
                return f"{float(x):+,.2f}%"
            except Exception:
                return x

        fmt: dict[str, object] = {}
        for col in df.columns:
            if col in ["Last", "Prev Close", "Open", "High", "Low"]:
                fmt[col] = _fmt_price
            elif col == "Change":
                fmt[col] = _fmt_change
            elif col == "% Change":
                fmt[col] = _fmt_pct

        styled = styled.format(fmt)

        # Color Change / % Change: green for up, red for down
        def _change_style(v):
            try:
                val = float(v)
            except Exception:
                return ""
            if val > 0:
                return "color: #00C853; font-weight: 600;"  # green
            if val < 0:
                return "color: #FF5252; font-weight: 600;"  # red
            return ""

        for col in ["Change", "% Change"]:
            if col in df.columns:
                styled = styled.applymap(_change_style, subset=[col])

        # Color Last, Prev Close, Open, High, Low relative to Prev Close (green if above, red if below)
        if "Prev Close" in df.columns:
            def _price_relative_style(v, prev_close):
                try:
                    val = float(v)
                    pc = float(prev_close)
                except Exception:
                    return ""
                if val > pc:
                    return "color: #00C853; font-weight: 500;"
                if val < pc:
                    return "color: #FF5252; font-weight: 500;"
                return ""

            # Apply per-row
            def _apply_price_row(row):
                styles = {}
                pc = row.get("Prev Close", None)
                for c in ["Last", "Prev Close", "Open", "High", "Low"]:
                    if c in row:
                        styles[c] = _price_relative_style(row[c], pc)
                return pd.Series(styles)

            styled = styled.apply(_apply_price_row, axis=1)

    # Results table: Basic users must not see the interactive dataframe toolbar (includes CSV download).
    # Pro/Premium can keep the interactive dataframe.
    if can_export_csv:
        try:
            _tbl = st.dataframe(
                styled,
                use_container_width=True,
                height=420,
                selection_mode="single-row",
                on_select="rerun",
                key="results_table_styled",
            )
            _sync_selected_ticker_from_table(_tbl, df, picker_key="results_chart_picker")
        except Exception:
            # Fallback: keep styled rendering without selection
            st.dataframe(styled, use_container_width=True, height=420)
    else:
        # Basic: keep the pro styling but render as static HTML (no Streamlit dataframe toolbar/download).
        # Mobile-safe: enable horizontal scroll + prevent vertical letter stacking.
        try:
            styled_basic = styled
            try:
                # pandas>=1.4
                styled_basic = styled_basic.hide(axis="index")
            except Exception:
                pass

            table_html = styled_basic.to_html()

            st.markdown(
                """
<style>
.basic-results-wrap {
  max-height: 420px;
  overflow-x: auto;
  overflow-y: auto;
  border: 1px solid rgba(49, 51, 63, 0.25);
  border-radius: 10px;
  padding: 6px;
}

/* Prevent vertical letter stacking on mobile */
.basic-results-wrap table {
  width: max-content;
  min-width: 100%;
  border-collapse: collapse;
}

.basic-results-wrap th,
.basic-results-wrap td {
  white-space: nowrap;
  padding: 6px 10px;
}

/* Sticky header */
.basic-results-wrap th {
  position: sticky;
  top: 0;
  background: rgba(15, 17, 22, 0.98);
  z-index: 2;
}
</style>
""",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div class='basic-results-wrap'>{table_html}</div>",
                unsafe_allow_html=True,
            )
        except Exception:
            # Fallback: still non-interactive
            try:
                st.table(df)
            except Exception:
                st.markdown(df.to_html(index=False), unsafe_allow_html=True)

    # Export (tier-gated)
    if can_export_csv:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name="breakout_results.csv",
            mime="text/csv",
            use_container_width=False,
        )
    else:
        st.info("🔒 Pro feature — export scan results to CSV")

    # Chart picker/details/AI notes: Option A logic for non-fast (styled) branch
    if is_basic:
        auto_t = _auto_details_ticker(df)
        if auto_t:
            with st.expander(f"📌 {auto_t} details", expanded=False):
                st.caption("📌 Top breakout candidate (auto-selected). 🔒 Upgrade to Pro to explore other tickers.")
                try:
                    row_df = df[df["Ticker"].astype(str).str.upper() == str(auto_t).upper()]
                    r0 = row_df.iloc[0] if len(row_df) else None
                except Exception:
                    r0 = None

                if r0 is not None:
                    c1, c2, c3, c4 = st.columns(4)

                    def _as_float(v):
                        try:
                            if pd.isna(v):
                                return None
                            return float(v)
                        except Exception:
                            return None

                    bs = _as_float(r0.get("BreakoutScore"))
                    last = _as_float(r0.get("Last"))
                    gap = _as_float(r0.get("GapPct"))
                    dv = _as_float(r0.get("DollarVol20"))

                    c1.metric("BreakoutScore", "—" if bs is None else f"{bs:.2f}")
                    c2.metric("Last", "—" if last is None else f"{last:.2f}")
                    c3.metric("Gap%", "—" if gap is None else f"{gap:.2f}%")
                    c4.metric("$Vol20", "—" if dv is None else f"{dv:,.0f}")

                    earn_days = _as_float(r0.get("📅 Earnings in X days")) if "📅 Earnings in X days" in df.columns else None
                    if earn_days is None:
                        st.caption("📅 Earnings: —")
                    else:
                        st.caption(f"📅 Earnings in {int(earn_days)} days")

                    with st.expander("Show row fields", expanded=False):
                        st.json({k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in r0.to_dict().items()})
                else:
                    st.caption("No row details available for this ticker.")
        else:
            st.caption("No ticker details available.")
        # Do NOT render charts or watchlist action in Basic mode
    else:
        # Pro/Premium: keep existing Charts expander + details + watchlist action
        with st.expander("📈 Charts", expanded=False):
            tickers = df["Ticker"].tolist() if "Ticker" in df.columns else []
            if not tickers:
                st.caption("No tickers available to chart.")
                return

            pick = st.selectbox("Select ticker to chart", tickers, key="results_chart_picker")
            render_chart_for_ticker(pick)

        pick = st.session_state.get("results_chart_picker")

        # Detail panel (row-click driven)
        selected_ticker = st.session_state.get("results_selected_ticker") or pick
        if selected_ticker:
            with st.expander(f"📌 {selected_ticker} details", expanded=False):
                # Show a compact stats + earnings card (no extra network calls)
                try:
                    row_df = df[df["Ticker"].astype(str).str.upper() == str(selected_ticker).upper()]
                    r0 = row_df.iloc[0] if len(row_df) else None
                except Exception:
                    r0 = None

                if r0 is not None:
                    c1, c2, c3, c4 = st.columns(4)

                    def _as_float(v):
                        try:
                            if pd.isna(v):
                                return None
                            return float(v)
                        except Exception:
                            return None

                    bs = _as_float(r0.get("BreakoutScore"))
                    last = _as_float(r0.get("Last"))
                    gap = _as_float(r0.get("GapPct"))
                    dv = _as_float(r0.get("DollarVol20"))

                    c1.metric("BreakoutScore", "—" if bs is None else f"{bs:.2f}")
                    c2.metric("Last", "—" if last is None else f"{last:.2f}")
                    c3.metric("Gap%", "—" if gap is None else f"{gap:.2f}%")
                    c4.metric("$Vol20", "—" if dv is None else f"{dv:,.0f}")

                    earn_days = _as_float(r0.get("📅 Earnings in X days")) if "📅 Earnings in X days" in df.columns else None
                    if earn_days is None:
                        st.caption("📅 Earnings: —")
                    else:
                        st.caption(f"📅 Earnings in {int(earn_days)} days")

                    # ⭐ Add to watchlist action
                    _render_watchlist_action(str(selected_ticker))

                    with st.expander("Show row fields", expanded=False):
                        st.json({k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in r0.to_dict().items()})
                else:
                    st.caption("No row details available for this ticker.")

    # AI notes (tier-gated)
    if can_ai_notes:
        st.subheader("AI Notes")
        st.caption("⭐ Premium feature")
        try:
            # Use the same ticker the user selected for the chart
            pick = st.session_state.get("results_chart_picker")
            row = df[df["Ticker"] == pick].iloc[0]
            auto_note = generate_ai_note(row)
            st.markdown(auto_note)
            st.text_area(
                "Edit or copy these notes (Premium only):",
                value=auto_note,
                height=220,
            )
        except Exception:
            st.caption("AI notes are unavailable for the selected row.")
    else:
        st.info("🔒 Premium feature — AI-powered notes for the selected ticker")