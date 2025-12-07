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
