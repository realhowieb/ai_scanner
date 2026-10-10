"""Conversational Q&A scoped to the current scan results (Premium)."""
from __future__ import annotations

import pandas as pd

_CHAT_COLS = ["Ticker", "Symbol", "HSF Score", "BreakoutScore", "GapPct", "Gap%", "Trend20D%", "VolRel20", "DollarVol20", "Volatility20D%"]
_MAX_ROWS = 20
_MAX_HISTORY = 8  # cap turns sent to control cost

_SYSTEM = (
    "You are a concise equity-scan analyst answering follow-up questions about "
    "a specific stock scan. The scan results are provided as CSV in the first "
    "user message. Answer using ONLY those results plus general technical "
    "knowledge; reference specific numbers, including HSF Score when relevant. HSF "
    "Score is HSF's own opportunity ranking and controls ordering; you do not rank, "
    "score, pick, or recommend securities. If asked about a ticker not in the scan, "
    "say it's not in these results. Do not infer institutional activity, future price "
    "movement, or a user's risk tolerance. Never give buy, sell, hold, entry, exit, "
    "position-size, stop-loss, or price-target language. Keep answers under 120 words. "
    "Educational research commentary only — not investment advice."
)


def _table(df: pd.DataFrame) -> str:
    cols = [c for c in _CHAT_COLS if c in df.columns]
    return (df[cols] if cols else df).head(_MAX_ROWS).to_csv(index=False)


def _current_user():
    try:
        import streamlit as st
        return st.session_state.get("username")
    except Exception:
        return None


def render_results_chat(df: pd.DataFrame) -> None:
    """Mini chat scoped to the current scan. History lives in session_state."""
    import streamlit as st

    st.markdown("#### 💬 Ask about these results")
    st.caption("Claude (AI) answers questions about the results shown — research commentary, not investment advice.")
    st.caption('e.g. "What stands out about the top result?" or "Why isn\'t TSLA in here?"')

    if df is None or len(df) == 0:
        st.info("Run a scan first, then ask questions about the results.")
        return

    hist_key = "_ai_chat_history"
    history: list[dict] = st.session_state.get(hist_key, [])

    # Render prior turns.
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    prompt = st.chat_input("Ask about your scan results")
    if not prompt:
        if history and st.button("🧹 Clear chat", key="ai_chat_clear"):
            st.session_state.pop(hist_key, None)
            st.rerun()
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    # Build the message list: scan context as the first user turn, then history.
    table_text = _table(df)
    api_messages: list[dict] = [{
        "role": "user",
        "content": f"Here are the current scan results (CSV):\n\n{table_text}",
    }, {
        "role": "assistant",
        "content": "Got it — I have the scan results. Ask away.",
    }]
    for turn in history[-_MAX_HISTORY:]:
        api_messages.append({"role": turn["role"], "content": turn["content"]})
    api_messages.append({"role": "user", "content": prompt})

    from ui.ai import ask_claude_chat
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, err = ask_claude_chat(
                system=_SYSTEM,
                messages=api_messages,
                max_tokens=500,
                username=_current_user(),
                feature="results_chat",
            )
        if answer:
            st.markdown(answer)
        else:
            st.warning(err or "Could not answer that.")
            return

    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": answer})
    st.session_state[hist_key] = history[-(_MAX_HISTORY * 2):]
