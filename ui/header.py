import streamlit as st

def render_header():
    try:
        st.image("assets/market_ai_logo.png", width=260)
    except Exception:
        pass
    st.markdown(
        """
        <div style='margin-top:0.75rem; margin-bottom:1.5rem;'>
            <p style='margin:0; font-size:0.95rem; color:gray;'>
                Money Moves · AI Breakout Score · Subscription Ready
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
