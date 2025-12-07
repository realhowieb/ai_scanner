import streamlit as st


def render_header() -> None:
    """Render a compact, left-aligned MarketPulse AI header logo."""

    # Narrow logo column + big spacer so the header doesn't take full width
    logo_col, _ = st.columns([1, 4])

    with logo_col:
        # Tiny vertical padding so it doesn't feel glued to the top
        st.markdown(
            "<div style='padding-top:0.5rem; padding-bottom:0.5rem;'>",
            unsafe_allow_html=True,
        )
        st.image(
            "assets/market_ai_logo_tighter.png",
            use_container_width=False,
            width=600,  # smaller = less perceived padding
        )
        st.markdown("</div>", unsafe_allow_html=True)