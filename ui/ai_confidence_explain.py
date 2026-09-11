"""Claude-generated "why this AI Confidence score" explanations.

The XGBoost model decides the number; Claude only *explains* it in plain English
and folds in a real, app-tracked catalyst (earnings proximity). It is grounded
strictly on the values passed in and told not to invent news, prices, analyst
views, or catalysts — so it narrates data we already have rather than
hallucinating market information (the reason we never let an LLM produce the
probability itself).
"""
from __future__ import annotations

from typing import Any, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

# The AI Confidence model's features, with human labels for the prompt.
_FEATURE_LABELS = {
    "Trend10D%": "10-day trend %",
    "Trend20D%": "20-day trend %",
    "VolRel20": "relative volume (× 20-day avg)",
    "DollarVol20": "20-day dollar volume",
    "BreakoutScore": "breakout score",
    "GapPct": "gap %",
}


def confidence_tier(pct: Optional[float]) -> str:
    """Coarse conviction tier from the calibrated confidence %."""
    if pct is None:
        return "unknown"
    try:
        value = float(pct)
    except (TypeError, ValueError):
        return "unknown"
    if value >= 40:
        return "High"
    if value >= 25:
        return "Medium"
    return "Low"


def _earnings_days_for(ticker: str) -> Optional[int]:
    """Real, app-tracked catalyst: trading days until this ticker's earnings."""
    try:
        from scheduler.morning_digest import _earnings_days_map

        return _earnings_days_map([str(ticker).upper()]).get(str(ticker).upper())
    except Exception:
        return None


def explain_confidence(
    row: dict[str, Any],
    *,
    earnings_days: Optional[int] = None,
    username: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Ask Claude to explain one ticker's calibrated AI Confidence score.

    Grounded strictly on the provided feature values (+ a real earnings catalyst
    when known). Returns (text, error); never raises.
    """
    try:
        from ui.ai import ask_claude, is_configured
    except Exception:
        return None, "AI is not configured."
    if not is_configured():
        return None, "AI is not configured."

    ticker = str(row.get("ticker") or row.get("Ticker") or row.get("Symbol") or "this ticker")
    conf = row.get("AI Confidence")
    feats = []
    for col, label in _FEATURE_LABELS.items():
        if col in row and row[col] is not None:
            try:
                feats.append(f"- {label}: {float(row[col]):.2f}")
            except (TypeError, ValueError):
                continue
    if not feats:
        return None, "No model feature values available to explain."

    catalyst = ""
    if earnings_days is not None:
        catalyst = f"\nReal catalyst on file: earnings in {int(earnings_days)} trading day(s)."

    system = (
        "You explain a stock scanner's calibrated 'AI Confidence' score to an "
        "experienced trader. The score is a walk-forward-validated, isotonic-"
        "calibrated probability that the setup reaches +4% before -2% within 5 "
        "trading days. In 1-2 sentences, explain WHY the model likely scored it "
        "this way, citing the specific feature values that push it up or down. "
        "Use ONLY the values provided. Do NOT invent news, prices, analyst views, "
        "or any catalyst beyond what is given. Be concise and non-promissory — "
        "this is a probability, not a prediction. No emojis."
    )
    user = (
        f"Ticker: {ticker}\n"
        f"Calibrated AI Confidence: {conf}% (tier: {confidence_tier(conf)})\n"
        f"Model feature values:\n" + "\n".join(feats) + catalyst
    )
    return ask_claude(
        system=system,
        user=user,
        max_tokens=220,
        username=username,
        feature="ai_confidence_explain",
    )


def render_confidence_explainer(df: Any) -> None:
    """Admin expander: pick a ticker, get Claude's 'why this score' + catalyst.

    Safe to call unconditionally — returns quietly when AI is off, the column is
    absent, or Streamlit is unavailable.
    """
    if st is None or df is None or getattr(df, "empty", True):
        return
    try:
        from scan.ai_confidence import CONFIDENCE_COL

        if CONFIDENCE_COL not in df.columns:
            return
        from ui.ai import is_configured

        if not is_configured():
            return
    except Exception:
        return

    tcol = next((c for c in ("ticker", "Ticker", "Symbol") if c in df.columns), None)
    if not tcol:
        return

    username = st.session_state.get("username")
    with st.expander("🧠 Why this AI Confidence? (Claude explains the score)", expanded=False):
        st.caption(
            "Claude explains the model's number from its feature values and any "
            "real earnings catalyst. It never sets the score or invents news."
        )
        tickers = [str(t) for t in df[tcol].head(50).tolist() if str(t).strip()]
        if not tickers:
            st.caption("No tickers to explain.")
            return
        pick = st.selectbox("Ticker", tickers, key="aic_explain_pick", label_visibility="collapsed")
        if st.button("Explain this score", key="aic_explain_btn"):
            row_df = df[df[tcol].astype(str) == pick]
            if row_df.empty:
                st.caption("Ticker not found in results.")
            else:
                row = row_df.iloc[0].to_dict()
                with st.spinner("Asking Claude…"):
                    text, err = explain_confidence(
                        row, earnings_days=_earnings_days_for(pick), username=username
                    )
                if err:
                    st.caption(f"Couldn't explain: {err}")
                else:
                    st.session_state[f"aic_explain_{pick}"] = text
        cached = st.session_state.get(f"aic_explain_{pick}")
        if cached:
            st.markdown(cached)
