"""Results, charts, and AI notes UI module."""

from typing import Callable, Optional

import logging
import re


import pandas as pd
import streamlit as st

# Optional auto-refresh for results-only live mode
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:
    st_autorefresh = None

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

    # Optional Live Mode: user-controlled auto-refresh every 10s for the results table
    if st_autorefresh is not None:
        live_results = st.checkbox(
            "🔁 Live results (10s refresh)",
            value=False,
            key="results_live_mode",
            help="When enabled, the results table refreshes about every 10 seconds.",
        )
        if live_results:
            st_autorefresh(interval=10_000, key="results_autorefresh")
    else:
        st.caption("Tip: install `streamlit-autorefresh` to enable live results refresh.")

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
            st.dataframe(df, use_container_width=True, height=420)
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

        # Continue with charts / AI notes
        st.subheader("Charts")
        tickers = df["Ticker"].tolist() if "Ticker" in df.columns else []
        if not tickers:
            st.caption("No tickers available to chart.")
            return

        pick = st.selectbox("Select ticker to chart", tickers)
        render_chart_for_ticker(pick)

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

    # Chart picker
    st.subheader("Charts")
    tickers = df["Ticker"].tolist() if "Ticker" in df.columns else []
    if not tickers:
        st.caption("No tickers available to chart.")
        return

    pick = st.selectbox("Select ticker to chart", tickers)
    render_chart_for_ticker(pick)

    # AI notes (tier-gated)
    if can_ai_notes:
        st.subheader("AI Notes")
        st.caption("⭐ Premium feature")
        try:
            # Use the same ticker the user selected for the chart
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