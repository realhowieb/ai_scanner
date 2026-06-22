"""Results, charts, and AI notes UI module."""
from __future__ import annotations

from typing import Callable, Optional
import re


import pandas as pd
import streamlit as st
from ui.result_helpers import (
    as_optional_float,
    auto_details_ticker,
    disable_yfinance_for_session as _disable_yfinance_for_session,
    find_row_for_ticker,
    get_results_df,
    is_yahoo_crumb_error as _is_yahoo_crumb_error,
    quiet_provider_loggers,
    row_to_jsonable_dict,
    sync_selected_ticker_from_table,
    warn_yfinance_disabled_once as _warn_yfinance_disabled_once,
)
from ui.result_tables import render_static_results_table
from ui.result_watchlist import render_watchlist_action


quiet_provider_loggers()


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

    # Earnings in-card display should only appear when enrichment is enabled (and never for Basic)
    earn_col = "📅 Earnings in X days"
    earn_enabled = bool(st.session_state.get("enable_earnings_enrichment", False))
    show_earnings_in_cards = (not is_basic) and earn_enabled and (earn_col in df.columns)

    # 🔒 Basic hard-lock: clear any selection state so Basic cannot "inherit" Pro clicks
    if is_basic:
        for k in (
            "results_selected_ticker",
            "results_chart_picker",
            "results_chart_picker_fast",
            "results_table_fast",
            "results_table_styled",
            "results_enable_styling",
        ):
            st.session_state.pop(k, None)

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
    if earn_col in df.columns and (not is_basic):
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

    # Show Basic upsell message for earnings filters, if earnings column exists and user is Basic
    if earn_col in df.columns and is_basic:
        st.info("🔒 Pro feature — earnings filters (exclude earnings soon / only within X days)")

    # --- Performance guard: Pandas Styler becomes very slow on large tables ---
    MAX_STYLED_ROWS = 1500
    MAX_STYLED_COLS = 25

    # "Fast mode" disables ALL pandas Styler work (df.style / applymap / gradients / to_html).
    # Even medium-sized tables can feel slow on Streamlit Cloud, so we also provide a manual toggle.
    auto_fast = (len(df) > MAX_STYLED_ROWS) or (df.shape[1] > MAX_STYLED_COLS)

    # Styling must be OPT-IN. Even ~50–200 rows can feel slow with multiple Styler passes.
    STYLE_ROW_LIMIT = 40

    default_enable_style = False

    # 🔒 Basic: hide styling toggle entirely (keeps Basic fast + avoids extra UI options)
    if is_basic:
        enable_styling = False
    else:
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
                width="stretch",
                height=420,
                selection_mode="single-row",
                on_select="rerun",
                key="results_table_fast",
            )
            sync_selected_ticker_from_table(_tbl, df, picker_key="results_chart_picker_fast")
        else:
            # Basic: keep non-interactive rendering. Use plain HTML (much faster than styled.to_html).
            render_static_results_table(df, fallback_df=df)

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
            auto_t = auto_details_ticker(df)
            if auto_t:
                with st.expander(f"📌 {auto_t} details", expanded=False):
                    st.caption(
                        "📌 **Top breakout candidate (auto-selected)**  \n"
                        "🔒 Upgrade to Pro to select tickers, view charts, and export CSV."
                    )
                    r0 = find_row_for_ticker(df, auto_t)
                    if r0 is not None:
                        c1, c2, c3, c4 = st.columns(4)

                        bs = as_optional_float(r0.get("BreakoutScore"))
                        last = as_optional_float(r0.get("Last"))
                        gap = as_optional_float(r0.get("GapPct"))
                        dv = as_optional_float(r0.get("DollarVol20"))

                        c1.metric("BreakoutScore", "—" if bs is None else f"{bs:.2f}")
                        c2.metric("Last", "—" if last is None else f"{last:.2f}")
                        c3.metric("Gap%", "—" if gap is None else f"{gap:.2f}%")
                        c4.metric("$Vol20", "—" if dv is None else f"{dv:,.0f}")
                        # Basic: do not show earnings in the details card

                        with st.expander("Show row fields", expanded=False):
                            st.json(row_to_jsonable_dict(r0))
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
            # Chart follows the clicked row (preferred). Fallback to last chart pick or first ticker.
            ticker_col = "Ticker" if "Ticker" in df.columns else ("Symbol" if "Symbol" in df.columns else None)
            if not ticker_col:
                st.caption("No tickers available to chart.")
            else:
                tickers = (
                    df[ticker_col]
                    .astype(str)
                    .map(lambda x: x.strip().upper())
                    .tolist()
                )
                tickers = [t for t in tickers if t]
                tickers = list(dict.fromkeys(tickers))

                if not tickers:
                    st.caption("No tickers available to chart.")
                else:
                    picker_key = "results_chart_picker_fast"
                    auto_pick = (st.session_state.get("results_selected_ticker") or st.session_state.get(picker_key) or tickers[0])
                    auto_pick = str(auto_pick).strip().upper()
                    if auto_pick not in tickers:
                        auto_pick = tickers[0]

                    # Persist the current chart ticker so it survives reruns.
                    st.session_state[picker_key] = auto_pick

                    st.caption(f"Charting: **{auto_pick}** — click a row in the table to change the chart.")

                    # Optional manual override (kept but hidden by default)
                    with st.expander("Change chart ticker", expanded=False):
                        pick = st.selectbox("Ticker", tickers, index=tickers.index(auto_pick), key=f"{picker_key}_manual")
                        st.session_state[picker_key] = str(pick).strip().upper()

                    render_chart_for_ticker(st.session_state[picker_key])

        pick = st.session_state.get("results_chart_picker_fast")

        # Detail panel (row-click driven)
        selected_ticker = st.session_state.get("results_selected_ticker") or pick
        if selected_ticker:
            with st.expander(f"📌 {selected_ticker} details", expanded=False):
                # Show a compact stats + earnings card (no extra network calls)
                r0 = find_row_for_ticker(df, selected_ticker)
                if r0 is not None:
                    c1, c2, c3, c4 = st.columns(4)

                    # Core stats
                    bs = as_optional_float(r0.get("BreakoutScore"))
                    last = as_optional_float(r0.get("Last"))
                    gap = as_optional_float(r0.get("GapPct"))
                    dv = as_optional_float(r0.get("DollarVol20"))

                    c1.metric("BreakoutScore", "—" if bs is None else f"{bs:.2f}")
                    c2.metric("Last", "—" if last is None else f"{last:.2f}")
                    c3.metric("Gap%", "—" if gap is None else f"{gap:.2f}%")
                    c4.metric("$Vol20", "—" if dv is None else f"{dv:,.0f}")

                    # Earnings (only when enrichment is enabled)
                    if show_earnings_in_cards:
                        earn_days = as_optional_float(r0.get(earn_col))
                        if earn_days is None:
                            st.caption("📅 Earnings: TBA")
                        else:
                            st.caption(f"📅 Earnings in {int(earn_days)} days")

                    # ⭐ Add to watchlist action
                    render_watchlist_action(str(selected_ticker))

                    # Optional: show a tiny raw row preview for debugging (collapsed)
                    with st.expander("Show row fields", expanded=False):
                        st.json(row_to_jsonable_dict(r0))
                else:
                    st.caption("No row details available for this ticker.")

        if can_ai_notes:
            st.subheader("AI Notes")
            st.caption("⭐ Premium feature")
            try:
                row = find_row_for_ticker(df, pick)
                if row is None:
                    raise ValueError("No matching row for selected ticker")
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
            except (TypeError, ValueError, KeyError):
                return ""

            # Warning for earnings very soon
            if 0 <= d <= 3:
                return "background-color: #FFF3CD; color: #856404; font-weight: 700;"
            # Soft info for 4-7 days
            if 4 <= d <= 7:
                return "background-color: #E8F4FD; color: #0C5460;"
            return ""

        styled = styled.map(_earnings_style, subset=["📅 Earnings in X days"])

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
            except (TypeError, ValueError):
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
            except (TypeError, ValueError):
                return x

        def _fmt_change(x):
            try:
                return f"{float(x):+,.2f}"
            except (TypeError, ValueError):
                return x

        def _fmt_pct(x):
            try:
                return f"{float(x):+,.2f}%"
            except (TypeError, ValueError):
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
            except (TypeError, ValueError):
                return ""
            if val > 0:
                return "color: #00C853; font-weight: 600;"  # green
            if val < 0:
                return "color: #FF5252; font-weight: 600;"  # red
            return ""

        for col in ["Change", "% Change"]:
            if col in df.columns:
                styled = styled.map(_change_style, subset=[col])

        # Color Last, Prev Close, Open, High, Low relative to Prev Close (green if above, red if below)
        if "Prev Close" in df.columns:
            def _price_relative_style(v, prev_close):
                try:
                    val = float(v)
                    pc = float(prev_close)
                except (TypeError, ValueError):
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
                width="stretch",
                height=420,
                selection_mode="single-row",
                on_select="rerun",
                key="results_table_styled",
            )
            sync_selected_ticker_from_table(_tbl, df, picker_key="results_chart_picker")
        except Exception:
            # Fallback: keep styled rendering without selection
            st.dataframe(styled, width="stretch", height=420)
    else:
        # Basic: keep the pro styling but render as static HTML (no Streamlit dataframe toolbar/download).
        # Mobile-safe: enable horizontal scroll + prevent vertical letter stacking.
        render_static_results_table(styled, fallback_df=df)

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
        auto_t = auto_details_ticker(df)
        if auto_t:
            with st.expander(f"📌 {auto_t} details", expanded=False):
                st.caption(
                    "📌 **Top breakout candidate (auto-selected)**  \n"
                    "🔒 Upgrade to Pro to select tickers, view charts, and export CSV."
                )
                r0 = find_row_for_ticker(df, auto_t)
                if r0 is not None:
                    c1, c2, c3, c4 = st.columns(4)

                    bs = as_optional_float(r0.get("BreakoutScore"))
                    last = as_optional_float(r0.get("Last"))
                    gap = as_optional_float(r0.get("GapPct"))
                    dv = as_optional_float(r0.get("DollarVol20"))

                    c1.metric("BreakoutScore", "—" if bs is None else f"{bs:.2f}")
                    c2.metric("Last", "—" if last is None else f"{last:.2f}")
                    c3.metric("Gap%", "—" if gap is None else f"{gap:.2f}%")
                    c4.metric("$Vol20", "—" if dv is None else f"{dv:,.0f}")

                    # Basic: do not show earnings in the details card

                    with st.expander("Show row fields", expanded=False):
                        st.json(row_to_jsonable_dict(r0))
                else:
                    st.caption("No row details available for this ticker.")
        else:
            st.caption("No ticker details available.")
        # Basic: keep AI notes locked as before (and avoid any selection/charts paths)
        if can_ai_notes:
            st.subheader("AI Notes")
            st.caption("⭐ Premium feature")
            st.caption("AI notes require a selectable ticker; upgrade to Pro/Premium.")
        else:
            st.info("🔒 Premium feature — AI-powered notes for the selected ticker")

        return
    else:
        # Pro/Premium: keep existing Charts expander + details + watchlist action
        with st.expander("📈 Charts", expanded=False):
            # Chart follows the clicked row (preferred). Fallback to last chart pick or first ticker.
            ticker_col = "Ticker" if "Ticker" in df.columns else ("Symbol" if "Symbol" in df.columns else None)
            if not ticker_col:
                st.caption("No tickers available to chart.")
            else:
                tickers = (
                    df[ticker_col]
                    .astype(str)
                    .map(lambda x: x.strip().upper())
                    .tolist()
                )
                tickers = [t for t in tickers if t]
                tickers = list(dict.fromkeys(tickers))

                if not tickers:
                    st.caption("No tickers available to chart.")
                else:
                    picker_key = "results_chart_picker"
                    auto_pick = (st.session_state.get("results_selected_ticker") or st.session_state.get(picker_key) or tickers[0])
                    auto_pick = str(auto_pick).strip().upper()
                    if auto_pick not in tickers:
                        auto_pick = tickers[0]

                    st.session_state[picker_key] = auto_pick

                    st.caption(f"Charting: **{auto_pick}** — click a row in the table to change the chart.")

                    # Optional manual override (kept but hidden by default)
                    with st.expander("Change chart ticker", expanded=False):
                        pick = st.selectbox("Ticker", tickers, index=tickers.index(auto_pick), key=f"{picker_key}_manual")
                        st.session_state[picker_key] = str(pick).strip().upper()

                    render_chart_for_ticker(st.session_state[picker_key])

        pick = st.session_state.get("results_chart_picker")

        # Detail panel (row-click driven)
        selected_ticker = st.session_state.get("results_selected_ticker") or pick
        if selected_ticker:
            with st.expander(f"📌 {selected_ticker} details", expanded=False):
                # Show a compact stats + earnings card (no extra network calls)
                r0 = find_row_for_ticker(df, selected_ticker)
                if r0 is not None:
                    c1, c2, c3, c4 = st.columns(4)

                    bs = as_optional_float(r0.get("BreakoutScore"))
                    last = as_optional_float(r0.get("Last"))
                    gap = as_optional_float(r0.get("GapPct"))
                    dv = as_optional_float(r0.get("DollarVol20"))

                    c1.metric("BreakoutScore", "—" if bs is None else f"{bs:.2f}")
                    c2.metric("Last", "—" if last is None else f"{last:.2f}")
                    c3.metric("Gap%", "—" if gap is None else f"{gap:.2f}%")
                    c4.metric("$Vol20", "—" if dv is None else f"{dv:,.0f}")

                    # Earnings (only when enrichment is enabled)
                    if show_earnings_in_cards:
                        earn_days = as_optional_float(r0.get(earn_col))
                        if earn_days is None:
                            st.caption("📅 Earnings: TBA")
                        else:
                            st.caption(f"📅 Earnings in {int(earn_days)} days")

                    # ⭐ Add to watchlist action
                    render_watchlist_action(str(selected_ticker))

                    with st.expander("Show row fields", expanded=False):
                        st.json(row_to_jsonable_dict(r0))
                else:
                    st.caption("No row details available for this ticker.")

    # AI notes (tier-gated)
    if can_ai_notes:
        st.subheader("AI Notes")
        st.caption("⭐ Premium feature")
        try:
            # Use the same ticker the user selected for the chart
            pick = st.session_state.get("results_chart_picker")
            row = find_row_for_ticker(df, pick)
            if row is None:
                raise ValueError("No matching row for selected ticker")
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
