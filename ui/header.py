import streamlit as st

def render_header():
    st.markdown(
        """
        <div style='margin:0.5rem 0 0.25rem 0; display:flex; align-items:center;'>
            <img src="assets/market_ai_logo.png"
                 style="height:52px; width:auto; display:block;" />
        </div>
        <div style='margin:0 0 0.75rem 0;'>
            <p style='margin:0; font-size:0.9rem; color:gray;'>
                Money Moves · AI Breakout Score · Subscription Ready
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
