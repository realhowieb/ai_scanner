"""Results, charts, and AI notes UI module."""

from typing import Callable, Optional


import pandas as pd
import streamlit as st


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

    st.subheader("Results")
    st.caption(
        f"Showing {len(df)} results. Increase 'Top N Results' in the sidebar to see more, "
        "or relax filters (Min Gap %, price range, Unusual Volume Filter). "
        "If you see 0 results, try lowering Min Gap or turning off the Unusual Volume Filter."
    )

    # --- Pro styling for results table ---
    styled = df.style

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

    st.dataframe(styled, use_container_width=True, height=420)

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
        st.info("CSV export is available on Pro/Premium.")

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
        st.subheader("AI Notes (Premium)")
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
        st.caption("AI Notes are Premium-only.")