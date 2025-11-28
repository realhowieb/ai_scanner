import streamlit as st

def render_header():
    st.markdown(
        """
        <h1 style='margin-bottom:0px;'>📈 Breakout Stock Scanner</h1>
        <p style='margin-top:0px; margin-bottom:0px; color:gray;'>Money Moves • AI Breakout Score • Subscription Ready</p>
        """,
        unsafe_allow_html=True,
    )
