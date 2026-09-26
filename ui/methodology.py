"""Run 62 — "How HSF works": a short, readable methodology page.

Explains coverage, HSF Score, explanations, freshness, research and the
disclaimer in plain language. Operational/descriptive only: no effectiveness
statistics, no forward-experiment state, no hard-coded universe size (the live
count, when available, comes from the trust banner).
"""
from __future__ import annotations

from typing import List, Tuple

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

from ui.product_copy import DISCLAIMER, POSITIONING_LONG, PRODUCT_NAME, TAGLINE

METHODOLOGY_SECTIONS: List[Tuple[str, str]] = [
    ("Market coverage",
     "HSF scans a broad universe of tradable U.S.-listed stocks and ETFs on the major U.S. exchanges. "
     "The list is rebuilt from the market-data provider's tradable-asset list, so it changes as "
     "listings change. Preferred shares, SPAC units, warrants, rights and malformed symbols are "
     "excluded. Scheduled scans run several times each trading day."),
    ("What HSF Score means",
     "HSF Score is an opportunity-ranking score from 0 to 100. It orders setups by how strongly "
     "the current technical evidence lines up. It is **not** a probability of profit, **not** an "
     "expected return and **not** a prediction or guarantee. A higher score means a stronger "
     "current setup, not a better outcome."),
    ("Other numbers you may see",
     "Some views show model outputs next to the score. **PreBreakout** estimates how likely a "
     "stock is to develop a high-scoring scanner setup in the next few trading days. "
     "**AI Confidence** is a model estimate for a defined short-term price-move event. Both are "
     "research estimates of technical events, not probabilities of profit."),
    ("Why a stock appears",
     "Each result lists the technical and contextual evidence that put it there, such as relative "
     "volume, a price gap, trend over recent days, position against its recent high, strength "
     "against the S&P 500, and upcoming earnings. HSF explains the evidence; it does not tell you "
     "what to do with it."),
    ("Data freshness",
     "Results are point-in-time scanner observations: they show what the scanner saw at the time "
     "shown, and they are never revised with hindsight. Check the scan time before acting on any "
     "result, especially outside market hours."),
    ("How HSF evaluates itself",
     "HSF records its scanner observations as they happen and measures its methods forward, on "
     "data collected after the method was fixed. HSF will not publish performance claims until "
     "that forward research supports them. Any historical figures shown in the app are labelled "
     "as historical research: they are descriptive and do not represent validated forward "
     "performance."),
]


def render_methodology() -> None:
    """Render the methodology page body. Never raises."""
    if st is None:
        return
    try:
        st.title("How HSF works")
        st.caption(PRODUCT_NAME)
        st.markdown(f"**{TAGLINE}**")
        st.markdown(POSITIONING_LONG)
        for heading, body in METHODOLOGY_SECTIONS:
            st.subheader(heading)
            st.markdown(body)
        st.subheader("Not financial advice")
        st.markdown(DISCLAIMER + " Do your own research and consider your own circumstances "
                    "before making any investment decision.")
    except Exception as exc:
        from ui.safe_errors import show_error

        show_error("the methodology page", exc)
