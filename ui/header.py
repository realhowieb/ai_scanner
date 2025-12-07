import streamlit as st

def render_header():
    # Reduce top padding for the main page container
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 0rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Left-aligned single column
    col = st.columns([1])[0]

    with col:
        st.image(
            "assets/market_ai_logo_tighter.png",
            use_container_width=False,
            width=700,  # keep or tweak as you like
        )