import streamlit as st

def render_header() -> None:
    """Render a compact, left-aligned MarketPulse AI header logo."""

    # Logo column + spacer (prevents full-width stretch)
    logo_col, _ = st.columns([1.2, 4])

    with logo_col:
        st.markdown(
            "<div style='padding-top:0.5rem; padding-bottom:0.5rem;'>",
            unsafe_allow_html=True,
        )
        st.image(
            "assets/market_ai_logo_tighter.png",
            width="content",  # replaces deprecated use_container_width=False
        )
        st.markdown("</div>", unsafe_allow_html=True)