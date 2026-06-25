"""AI scan summary — uses Claude to summarize the top breakout setups.

Premium feature. Takes the latest scan results DataFrame and returns a concise
markdown analysis of the strongest setups and why they stand out.
"""
from __future__ import annotations

import hashlib

import pandas as pd

_SUMMARY_COLUMNS = [
    "Ticker", "Symbol", "BreakoutScore", "Gap%",
    "Trend20D%", "VolRel20", "DollarVol20", "Volatility20D%",
]
_MAX_ROWS = 15

_SYSTEM_PROMPT = (
    "You are a concise equity-scan analyst for a breakout stock scanner. "
    "You are given the top rows of a technical scan. Identify the 3-5 strongest "
    "setups and explain in plain language why each stands out (breakout score, "
    "gap, trend, relative volume, liquidity). Be specific and reference the "
    "numbers. Keep it under 250 words, use markdown bullet points, and end with "
    "one short risk caveat. Do NOT give financial advice or price targets; this "
    "is educational technical commentary only."
)

_TICKER_SYSTEM_PROMPT = (
    "You are a concise equity-scan analyst. You are given the technical scan "
    "metrics for a single stock. Explain this one setup in plain language: what "
    "the breakout score, gap, 20-day trend, relative volume, dollar volume, and "
    "volatility together suggest about the setup's quality and risk. Be specific "
    "and reference the numbers. Keep it under 150 words, use markdown bullets, "
    "and end with one short risk caveat. Do NOT give financial advice or price "
    "targets; this is educational technical commentary only."
)


def _current_user() -> str | None:
    """Best-effort current username for per-user AI usage tracking."""
    try:
        import streamlit as st
        return st.session_state.get("username")
    except Exception:
        return None


def _results_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of the table so we can cache the summary per scan."""
    try:
        cols = [c for c in _SUMMARY_COLUMNS if c in df.columns]
        subset = df[cols].head(_MAX_ROWS)
        return hashlib.sha256(subset.to_csv(index=False).encode()).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(str(len(df)).encode()).hexdigest()[:16]


def _build_table_text(df: pd.DataFrame) -> str:
    cols = [c for c in _SUMMARY_COLUMNS if c in df.columns]
    if not cols:
        return df.head(_MAX_ROWS).to_csv(index=False)
    return df[cols].head(_MAX_ROWS).to_csv(index=False)


def generate_scan_summary(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return (markdown_summary, error). Best-effort; never raises."""
    if df is None or len(df) == 0:
        return None, "No scan results to summarize. Run a scan first."

    table_text = _build_table_text(df)
    from ui.ai import ask_claude
    return ask_claude(
        system=_SYSTEM_PROMPT,
        user=(
            f"Here are the top {min(len(df), _MAX_ROWS)} scan results "
            f"(CSV):\n\n{table_text}\n\nSummarize the strongest setups."
        ),
        max_tokens=1024,
        username=_current_user(),
        feature="scan_summary",
    )


def render_ai_summary(df: pd.DataFrame) -> None:
    """Streamlit UI block — button-gated, cached per scan in session_state."""
    import streamlit as st

    st.markdown("#### 🤖 AI Scan Summary")
    st.caption("Claude reviews your top results and highlights the strongest setups.")

    fp = _results_fingerprint(df) if df is not None and len(df) else None
    cache_key = f"_ai_summary_{fp}" if fp else None

    if cache_key and st.session_state.get(cache_key):
        st.markdown(st.session_state[cache_key])
        if st.button("🔄 Regenerate", key=f"regen_{fp}"):
            st.session_state.pop(cache_key, None)
            st.rerun()
        return

    if st.button("✨ Generate AI summary", key=f"gen_{fp or 'none'}"):
        with st.spinner("Analyzing your scan results…"):
            summary, err = generate_scan_summary(df)
        if summary:
            if cache_key:
                st.session_state[cache_key] = summary
            st.markdown(summary)
        else:
            st.warning(err or "Could not generate summary.")


def generate_ticker_analysis(row) -> tuple[str | None, str | None]:
    """Analyze a single ticker's scan metrics. Returns (markdown, error)."""
    if row is None:
        return None, "No ticker selected."
    try:
        fields = {c: row.get(c) for c in _SUMMARY_COLUMNS if c in getattr(row, "index", row)}
    except Exception:
        fields = {c: (row.get(c) if hasattr(row, "get") else None) for c in _SUMMARY_COLUMNS}
    ticker = fields.get("Ticker") or fields.get("Symbol") or "this stock"
    metrics = "\n".join(f"- {k}: {v}" for k, v in fields.items() if v is not None)
    from ui.ai import ask_claude
    return ask_claude(
        system=_TICKER_SYSTEM_PROMPT,
        user=f"Technical scan metrics for {ticker}:\n\n{metrics}\n\nExplain this setup.",
        max_tokens=600,
        username=_current_user(),
        feature="ticker_deepdive",
    )


def render_ticker_analysis(row, ticker: str) -> None:
    """Button-gated single-ticker AI analysis, cached per ticker+metrics."""
    import hashlib as _h

    import streamlit as st

    try:
        fp = _h.sha256(str(dict(row)).encode()).hexdigest()[:12]
    except Exception:
        fp = (ticker or "x")[:12]
    cache_key = f"_ai_ticker_{fp}"

    if st.session_state.get(cache_key):
        st.markdown(st.session_state[cache_key])
        if st.button("🔄 Regenerate", key=f"ai_tkr_regen_{fp}"):
            st.session_state.pop(cache_key, None)
            st.rerun()
        return

    if st.button(f"🤖 Explain {ticker} setup", key=f"ai_tkr_{fp}"):
        with st.spinner(f"Analyzing {ticker}…"):
            text, err = generate_ticker_analysis(row)
        if text:
            st.session_state[cache_key] = text
            st.markdown(text)
        else:
            st.warning(err or "Could not analyze this ticker.")
