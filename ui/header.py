import streamlit as st

def render_header():
    try:
        st.image("assets/market_ai_logo.png", width=260)
    except Exception:
        pass
    st.markdown(
        """
        <h1 style='margin-bottom:0px;'>📈MarketPulse AI</h1>
        <p style='margin-top:0px; margin-bottom:0px; color:gray;'>Money Moves • AI Breakout Score • Subscription Ready</p>
        """,
        unsafe_allow_html=True,
    )
