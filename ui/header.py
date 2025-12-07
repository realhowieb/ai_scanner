import streamlit as st

def render_header():
    # Center the header content with a middle column
    _, center_col, _ = st.columns([1, 3, 1])

    with center_col:
        st.image(
            "assets/market_ai_logo_tighter.png",
            use_container_width=False,
            width=260,
        )
        st.markdown(
            """
            <p style='margin:0.25rem 0 0; font-size:0.95rem; color:gray; text-align:center;'>
                Money Moves · AI Breakout Score · Subscription Ready
            </p>
            """,
            unsafe_allow_html=True,
        )
