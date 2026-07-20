"""Results, charts, and AI notes UI module."""
from __future__ import annotations

import re
from typing import Callable, Optional

import pandas as pd
import streamlit as st

from scan.ai_confidence import CONFIDENCE_COL, SOURCE_ATTR, TRAINED_AT_ATTR, WARNING_ATTR
from ui.result_helpers import (
    as_optional_float,
    auto_details_ticker,
    find_row_for_ticker,
    get_results_df,
    move_column_after,
    quiet_provider_loggers,
    render_track_record_badge,
    row_to_jsonable_dict,
    sync_selected_ticker_from_table,
)
from ui.result_helpers import (
    disable_yfinance_for_session as _disable_yfinance_for_session,
)
from ui.result_helpers import (
    is_yahoo_crumb_error as _is_yahoo_crumb_error,
)
from ui.result_helpers import (
    warn_yfinance_disabled_once as _warn_yfinance_disabled_once,
)
from ui.result_tables import render_static_results_table
from ui.result_watchlist import render_watchlist_action

quiet_provider_loggers()

EARNINGS_TABLE_COLUMNS = ("earnings_in_days", "Earnings", "📅 Earnings in X days")


def _can_use_background_gradient() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_chart_pick(tickers, selected_key: str, picker_key: str) -> str:
    """Resolve which ticker to chart, reconciling row-clicks and the manual override.

    The row-click writes ``selected_key`` and is authoritative. The hidden
    "Change chart ticker" selectbox keeps its own widget state, which otherwise
    goes stale and clobbers the row-click; we sync it to the resolved pick
    before rendering and let its on_change write back to ``selected_key`` so a
    manual choice also sticks. Returns the resolved, persisted ticker.
    """
    auto_pick = str(
        st.session_state.get(selected_key)
        or st.session_state.get(picker_key)
        or tickers[0]
    ).strip().upper()
    if auto_pick not in tickers:
        auto_pick = tickers[0]
    st.session_state[picker_key] = auto_pick
    st.session_state[selected_key] = auto_pick

    manual_key = f"{picker_key}_manual"

    def _on_manual() -> None:
        choice = st.session_state.get(manual_key)
        if choice:
            st.session_state[selected_key] = str(choice).strip().upper()

    with st.expander("Change chart ticker", expanded=False):
        # Seed the widget to the resolved pick BEFORE it renders, so a row-click
        # is reflected here instead of stale widget state winning.
        st.session_state[manual_key] = auto_pick
        st.selectbox("Ticker", tickers, key=manual_key, on_change=_on_manual)

    return auto_pick


def render_results(
    df: Optional[pd.DataFrame],
    can_export_csv: bool,
    can_ai_notes: bool,
    render_chart_for_ticker: Callable[[str], None],
    generate_ai_note: Callable[[pd.Series], str],
    *,
    key_prefix: str = "results",
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

    # AI Confidence is an admin-only evaluation feature for now: it predicts the
    # present breakout label from features that include BreakoutScore, so it adds
    # little user-facing signal beyond BreakoutScore/PreBreakout. Keep it visible
    # to admins for A/B evaluation, but hide the column (and its caption) from
    # everyone else so users see one clean signal.
    is_admin_view = bool(ent.get("can_diagnostics"))
    if not is_admin_view and CONFIDENCE_COL in df.columns:
        df = df.drop(columns=[CONFIDENCE_COL])

    # Option A: Basic = auto-details only, no selection
    is_basic = not can_export_csv

    # Earnings in-card display should only appear when enrichment is enabled (and never for Basic)
    earn_col = "📅 Earnings in X days"
    earn_enabled = bool(st.session_state.get("enable_earnings_enrichment", False))
    show_earnings_in_cards = (not is_basic) and earn_enabled and (earn_col in df.columns)

    # 🔒 Basic hard-lock: clear any selection state so Basic cannot "inherit" Pro clicks
    if is_basic:
        for k in (
            f"{key_prefix}_selected_ticker",
            f"{key_prefix}_chart_picker",
            f"{key_prefix}_chart_picker_fast",
            f"{key_prefix}_table_fast",
            f"{key_prefix}_table_styled",
            f"{key_prefix}_enable_styling",
        ):
            st.session_state.pop(k, None)

    st.subheader("Results")
    ai_warning = df.attrs.get(WARNING_ATTR)
    ai_trained_at = df.attrs.get(TRAINED_AT_ATTR)
    ai_source = df.attrs.get(SOURCE_ATTR)
    if is_admin_view and ai_warning:
        st.caption(f"⚠️ {ai_warning}")
    if is_admin_view and ai_trained_at:
        source_text = f" • source: {ai_source}" if ai_source else ""
        st.caption(f"AI Confidence model trained at: {ai_trained_at}{source_text}")
    render_track_record_badge()
    try:
        from ui.score_map import render_score_map

        render_score_map(df, key=key_prefix)
    except Exception:
        pass
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
                key=f"{key_prefix}_earn_excl_3",
                help="Hides stocks with earnings 0–3 days away (keeps unknown earnings).",
            )
            within_7 = st.checkbox(
                "Only earnings within 7 days",
                value=False,
                key=f"{key_prefix}_earn_within_7",
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

    # Always use the fast, non-Styler table path. It is more reliable on Streamlit Cloud
    # and avoids optional matplotlib/Pandas Styler deployment failures.
    fast_mode = True

    if fast_mode:
        # Render without Styler for speed
        if can_export_csv:
            from ui.result_helpers import (
                RESULTS_HIDDEN_COLUMNS,
                format_ema_cross_column,
                results_column_config,
            )

            table_df = _format_earnings_for_display(df).drop(
                columns=list(RESULTS_HIDDEN_COLUMNS), errors="ignore"
            )
            table_df = format_ema_cross_column(table_df)
            table_df = move_column_after(table_df, "Spark10D", "Ticker")
            _tbl = st.dataframe(
                table_df,
                width="stretch",
                height=420,
                selection_mode="single-row",
                on_select="rerun",
                key=f"{key_prefix}_table_fast",
                column_config=results_column_config(),
            )
            sync_selected_ticker_from_table(
                _tbl,
                df,
                picker_key=f"{key_prefix}_chart_picker_fast",
                selected_key=f"{key_prefix}_selected_ticker",
            )
        else:
            # Basic: keep non-interactive rendering. Use plain HTML (much faster than styled.to_html).
            render_static_results_table(_format_earnings_for_display(df).drop(columns=["Spark10D"], errors="ignore"), fallback_df=df)

        # Export (tier-gated) still available even in fast mode
        if can_export_csv:
            csv = df.drop(columns=["Spark10D"], errors="ignore").to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=csv,
                file_name="breakout_results.csv",
                mime="text/csv",
                width="content",
                key=f"{key_prefix}_download_csv_fast",
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

                        if is_admin_view:
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
                    picker_key = f"{key_prefix}_chart_picker_fast"
                    selected_key = f"{key_prefix}_selected_ticker"
                    auto_pick = _resolve_chart_pick(tickers, selected_key, picker_key)

                    st.caption(f"Charting: **{auto_pick}** — click a row in the table to change the chart.")

                    render_chart_for_ticker(
                        auto_pick,
                        key=f"{key_prefix}_chart_{auto_pick}_fast",
                    )

        pick = st.session_state.get(f"{key_prefix}_chart_picker_fast")

        # Detail panel (row-click driven)
        selected_ticker = st.session_state.get(f"{key_prefix}_selected_ticker") or pick
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
                    from ui.trade_plan import render_trade_plan

                    render_trade_plan(r0, locked=is_basic, key=key_prefix)

                    # Earnings (only when enrichment is enabled)
                    if show_earnings_in_cards:
                        earn_days = as_optional_float(r0.get(earn_col))
                        if earn_days is None:
                            st.caption("📅 Earnings: TBA")
                        else:
                            st.caption(f"📅 Earnings in {int(earn_days)} days")

                    # ⭐ Add to watchlist action
                    render_watchlist_action(str(selected_ticker), key_prefix=key_prefix)

                    # 🤖 Premium: AI deep-dive on this single ticker
                    if can_ai_notes:
                        try:
                            from ui.ai_summary import render_ticker_analysis
                            render_ticker_analysis(r0, str(selected_ticker))
                        except Exception:
                            pass

                    # Optional: show a tiny raw row preview for debugging (collapsed)
                    if is_admin_view:
                        with st.expander("Show row fields", expanded=False):
                            st.json(row_to_jsonable_dict(r0))
                else:
                    st.caption("No row details available for this ticker.")

        if can_ai_notes:
            st.subheader("AI Notes")
            st.caption("⭐ Premium feature")
            try:
                note_ticker = selected_ticker or pick
                row = find_row_for_ticker(df, note_ticker)
                if row is None:
                    raise ValueError("No matching row for selected ticker")
                auto_note = generate_ai_note(row)
                st.markdown(auto_note.replace("$", "\\$"))
                st.text_area(
                    "Edit or copy these notes (Premium only):",
                    value=auto_note,
                    height=220,
                    key=f"{key_prefix}_ai_notes_fast_{note_ticker}",
                )
            except (RuntimeError, TypeError, ValueError):
                st.caption("AI notes are unavailable for the selected row.")
        else:
            st.info("🔒 Premium feature — AI-powered notes for the selected ticker")

        return

    # --- UI polish: Earnings column (display-only) ---

    df = move_column_after(df, "Spark10D", "Ticker")

    # Move earnings column next to Ticker (if present)
    if earn_col in df.columns and "Ticker" in df.columns:
        try:
            cols = list(df.columns)
            cols.remove(earn_col)
            ticker_idx = cols.index("Ticker")
            cols.insert(ticker_idx + 1, earn_col)
            df = df[cols]
        except (KeyError, ValueError):
            pass

    # --- Pro styling for results table ---
    table_df = _format_earnings_for_display(df)

    # Arrow-safe: the bare "Earnings" column can hold a mix of ints (days) and
    # "—"/text, which fails pyarrow serialization and triggers a Streamlit
    # "automatic fixes" warning. Coerce it to string for display. (The numeric
    # "📅 Earnings in X days" column the styler uses is left untouched.)
    if "Earnings" in table_df.columns:
        try:
            table_df = table_df.copy()
            table_df["Earnings"] = table_df["Earnings"].apply(
                lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
            )
        except (TypeError, ValueError, AttributeError):
            pass

    styled = table_df.style

    # Format earnings column: None/NaN -> — ; ints shown as whole numbers
    if "📅 Earnings in X days" in table_df.columns:
        styled = styled.format(
            {"📅 Earnings in X days": _format_earnings_cell}
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

    can_use_gradient = _can_use_background_gradient()

    # Heatmap for BreakoutScore
    if can_use_gradient and "BreakoutScore" in table_df.columns:
        styled = styled.background_gradient(axis=None, cmap="RdYlGn", subset=["BreakoutScore"])

    # Conditional formatting for RS_Rank (0-100)
    if can_use_gradient and "RS_Rank" in table_df.columns:
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

    if "Trend20D%" in table_df.columns:
        styled = styled.apply(_trend_style, subset=["Trend20D%"])
    if "Trend10D%" in table_df.columns:
        styled = styled.apply(_trend_style, subset=["Trend10D%"])

    # Watchlist-style numeric formatting (Symbol/Last/Change/% Change/etc.)
    watchlist_cols = {"Last", "Change", "% Change", "Prev Close", "Open", "High", "Low"}
    if watchlist_cols.intersection(table_df.columns):
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
        for col in table_df.columns:
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
            if col in table_df.columns:
                styled = styled.map(_change_style, subset=[col])

        # Color Last, Prev Close, Open, High, Low relative to Prev Close (green if above, red if below)
        if "Prev Close" in table_df.columns:
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
                key=f"{key_prefix}_table_styled",
            )
            sync_selected_ticker_from_table(
                _tbl,
                df,
                picker_key=f"{key_prefix}_chart_picker",
                selected_key=f"{key_prefix}_selected_ticker",
            )
        except (ImportError, RuntimeError, TypeError, ValueError):
            # Fallback: keep styled rendering without selection
            st.dataframe(table_df, width="stretch", height=420)
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
            width="content",
            key=f"{key_prefix}_download_csv_styled",
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

                    if is_admin_view:
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
                    picker_key = f"{key_prefix}_chart_picker"
                    selected_key = f"{key_prefix}_selected_ticker"
                    auto_pick = _resolve_chart_pick(tickers, selected_key, picker_key)

                    st.caption(f"Charting: **{auto_pick}** — click a row in the table to change the chart.")

                    render_chart_for_ticker(
                        auto_pick,
                        key=f"{key_prefix}_chart_{auto_pick}_styled",
                    )

        pick = st.session_state.get(f"{key_prefix}_chart_picker")

        # Detail panel (row-click driven)
        selected_ticker = st.session_state.get(f"{key_prefix}_selected_ticker") or pick
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
                    render_watchlist_action(str(selected_ticker), key_prefix=key_prefix)

                    if is_admin_view:
                        with st.expander("Show row fields", expanded=False):
                            st.json(row_to_jsonable_dict(r0))
                else:
                    st.caption("No row details available for this ticker.")

    # AI notes (tier-gated)
    if can_ai_notes:
        st.subheader("AI Notes")
        st.caption("⭐ Premium feature")
        try:
            # Use the same ticker the user selected (row-click), matching the
            # details panel — not just the chart picker.
            note_ticker = st.session_state.get(f"{key_prefix}_selected_ticker") or st.session_state.get(f"{key_prefix}_chart_picker")
            row = find_row_for_ticker(df, note_ticker)
            if row is None:
                raise ValueError("No matching row for selected ticker")
            auto_note = generate_ai_note(row)
            st.markdown(auto_note.replace("$", "\\$"))
            st.text_area(
                "Edit or copy these notes (Premium only):",
                value=auto_note,
                height=220,
                key=f"{key_prefix}_ai_notes_styled_{note_ticker}",
            )
        except (RuntimeError, TypeError, ValueError):
            st.caption("AI notes are unavailable for the selected row.")
    else:
        st.info("🔒 Premium feature — AI-powered notes for the selected ticker")


def _format_earnings_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table-only copy where unknown earnings values render cleanly."""
    table_df = df.copy()
    for col in EARNINGS_TABLE_COLUMNS:
        if col not in table_df.columns:
            continue
        values = pd.to_numeric(table_df[col], errors="coerce")
        table_df[col] = values.map(_format_earnings_cell)
    if "Earnings" in table_df.columns and "earnings_in_days" in table_df.columns:
        table_df = table_df.drop(columns=["earnings_in_days"])
    return table_df


def _format_earnings_cell(value: object) -> str:
    # Uniformly strings: mixing ints (known days) with "—" (unknown) makes an
    # object column pyarrow can't serialize, so every st.dataframe render
    # tripped Streamlit's "applying automatic fixes" warning on the fast path.
    if pd.isna(value):
        return "—"
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)
