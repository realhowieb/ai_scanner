import streamlit as st


def render_footer():
    st.divider()
    st.caption(
        "⚠️ **HSFinest.AI is for informational and educational purposes only and is "
        "not financial, investment, or trading advice.** Breakout scores and alerts "
        "are algorithmic signals, not recommendations. Trading involves risk of loss; "
        "do your own research and consult a licensed financial advisor before making "
        "any investment decision. Past performance does not guarantee future results."
    )
