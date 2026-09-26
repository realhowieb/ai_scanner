"""P2-3 — guided first-run tour.

Five short steps through the product: Today → Scanner → HSF Score → Stock
Intelligence → My Stocks. Shown until the user finishes or skips it, then
remembered in this browser (ui.browser_prefs). Copy is factual; no claims.
"""
from __future__ import annotations

from typing import List, Tuple

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

PREF_KEY = "hsf_tour"
STEP_KEY = "hsf_tour_step"
HIDDEN_KEY = "hsf_tour_hidden"

# (title, body, page, link label)
TOUR_STEPS: List[Tuple[str, str, str, str]] = [
    ("Start with Today",
     "Today shows the market status, the top setups from HSF's latest full-market scan, "
     "what's new since your last visit and a recap of the session.",
     "pages/today.py", "Open Today"),
    ("Explore the Scanner",
     "The Scanner shows the latest full-market scan until you run your own. Use the lens "
     "pills (Breakouts, Unusual volume, Gaps…) to narrow it, or switch to Cards on a phone.",
     "app.py", "Open the Scanner"),
    ("Read the HSF Score",
     "HSF Score (0–100) ranks how strongly a setup's current evidence lines up. It is a "
     "ranking, not a probability of profit. Each result lists the evidence behind it.",
     "pages/methodology.py", "How HSF works"),
    ("Understand one stock",
     "Stock Intelligence explains why HSF is showing a ticker, what to watch, and what "
     "changed since HSF first noticed it.",
     "pages/stock.py", "Open Stock Intelligence"),
    ("Track your stocks",
     "My Stocks keeps your watchlists and alerts together, so HSF can tell you when "
     "something meaningful changes.",
     "pages/watchlists.py", "Open My Stocks"),
]


def tour_done() -> bool:
    from ui.browser_prefs import get

    return get(PREF_KEY) == "done"


def clamp_step(i: int) -> int:
    return max(0, min(int(i), len(TOUR_STEPS) - 1))


def _finish() -> None:
    from ui.browser_prefs import put

    put(PREF_KEY, "done")
    st.session_state[HIDDEN_KEY] = True


def render_tour(where: str) -> None:
    """Render the tour card (once per browser until finished/skipped). Never raises."""
    if st is None or st.session_state.get(HIDDEN_KEY):
        return
    try:
        if tour_done():
            st.session_state[HIDDEN_KEY] = True
            return
        i = clamp_step(st.session_state.get(STEP_KEY, 0))
        title, body, page, link = TOUR_STEPS[i]
        with st.container(border=True):
            st.caption(f"Quick tour · step {i + 1} of {len(TOUR_STEPS)}")
            st.markdown(f"#### {title}")
            st.markdown(body)
            st.page_link(page, label=link)
            b1, b2, b3 = st.columns(3)
            if b1.button("Back", key=f"tour_back_{where}", disabled=i == 0):
                st.session_state[STEP_KEY] = i - 1
                st.rerun()
            last = i == len(TOUR_STEPS) - 1
            if b2.button("Finish" if last else "Next", key=f"tour_next_{where}", type="primary"):
                if last:
                    _finish()
                else:
                    st.session_state[STEP_KEY] = i + 1
                st.rerun()
            if b3.button("Skip tour", key=f"tour_skip_{where}"):
                _finish()
                st.rerun()
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("product tour", exc)
