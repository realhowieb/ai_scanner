"""Universe diagnostics / info panel UI."""

import streamlit as st


def init_universe_state() -> None:
    """Ensure universe-related keys exist in session_state.

    This keeps first-run startup clean and centralizes state init.
    """
    if "sp500_universe" not in st.session_state:
        st.session_state["sp500_universe"] = []
    if "nasdaq_universe" not in st.session_state:
        st.session_state["nasdaq_universe"] = []
    if "nasdaq_capped" not in st.session_state:
        st.session_state["nasdaq_capped"] = []
    if "combo_capped" not in st.session_state:
        st.session_state["combo_capped"] = []


def render_universe_panel() -> None:
    """Render the Universe Info expander based on session state.

    Shows sizes and sample tickers for SP500 and NASDAQ universes
    populated by the scan controls.
    """
    with st.expander("Universe Info", expanded=True):
        sp500 = st.session_state.get("sp500_universe", [])
        nasdaq_full = st.session_state.get("nasdaq_universe", [])
        nasdaq_capped = st.session_state.get("nasdaq_capped", [])

        if sp500 or nasdaq_full:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"**SP500 size:** {len(sp500)}" if sp500 else "**SP500 size:** (not loaded yet)"
                )
                if sp500:
                    st.caption(f"Sample: {', '.join(sp500[:20])}")
            with c2:
                if nasdaq_full:
                    size_label = len(nasdaq_capped) or len(nasdaq_full)
                    capped_suffix = " (capped)" if nasdaq_capped else ""
                    st.markdown(f"**NASDAQ size:** {size_label}{capped_suffix}")
                    st.caption(f"Sample: {', '.join((nasdaq_capped or nasdaq_full)[:20])}")
                else:
                    st.markdown("**NASDAQ size:** (not loaded yet)")
                    st.caption("Run a NASDAQ or Combo scan to populate NASDAQ universe.")
        else:
            st.caption(
                "Universes will appear here after you run your first scan (SP500, NASDAQ, or Combo)."
            )