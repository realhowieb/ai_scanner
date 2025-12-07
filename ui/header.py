import streamlit as st

def render_header():
    # Left-aligned single column
    col = st.columns([1])[0]

    with col:
        st.image(
            "assets/market_ai_logo_tighter.png",
            use_container_width=False,
            width=260,   # adjust as needed
        )

        st.markdown(
            """
            <p style='margin:0.1rem 0 1rem; 
                      font-size:0.95rem; 
                      color:gray; 
                      text-align:left;'>
                Money Moves · AI Breakout Score · Subscription Ready
            </p>
            """,
            unsafe_allow_html=True,
        )