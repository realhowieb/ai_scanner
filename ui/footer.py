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
    # Run 62: methodology is one click from every page that renders the footer.
    try:
        st.page_link("pages/methodology.py", label="How HSF works")
    except Exception:
        pass
