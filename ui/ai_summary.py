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

    try:
        from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
    except Exception:
        return None, "AI summary is not configured."

    if not ANTHROPIC_API_KEY:
        return None, "AI summary is not configured (missing ANTHROPIC_API_KEY)."

    try:
        import anthropic
    except ImportError:
        return None, "AI summary requires the `anthropic` package (add to requirements)."

    table_text = _build_table_text(df)
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"Here are the top {min(len(df), _MAX_ROWS)} scan results "
                    f"(CSV):\n\n{table_text}\n\nSummarize the strongest setups."
                ),
            }],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return (text or None), (None if text else "Empty response from AI.")
    except anthropic.AuthenticationError:
        return None, "AI summary failed: invalid ANTHROPIC_API_KEY."
    except anthropic.RateLimitError:
        return None, "AI summary rate-limited. Try again in a moment."
    except anthropic.APIError as e:
        return None, f"AI summary failed: {e}"
    except Exception as e:
        return None, f"AI summary error: {type(e).__name__}: {e}"


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
