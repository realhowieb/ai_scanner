import streamlit as st

def render_header():
    st.markdown("""
        <div style='text-align:center; margin-top:0.5rem;'>
            <img src="assets/market_ai_logo_tighter.png" style="width:260px; height:auto;" />
        </div>
        <div style='text-align:center; margin-top:0.25rem;'>
            <p style='margin:0; font-size:0.95rem; color:gray;'>
                Money Moves · AI Breakout Score · Subscription Ready
            </p>
        </div>
    """, unsafe_allow_html=True)
