import streamlit as st

def render_header():
    col1, col2 = st.columns([1, 5])

    with col1:
        st.image(
            "assets/market_ai_logo_tighter.png",
            use_container_width=False,
            width=160,
        )

    with col2:
        st.markdown(
            """
            <div style='display:flex; align-items:center; height:100%;'>
                <p style='margin:0; font-size:0.95rem; color:gray;'>
                    Money Moves · AI Breakout Score · Subscription Ready
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
