"""Natural-language screener — translate plain English into scan filters.

Premium feature. The user types a request like "sub-$20 NASDAQ stocks gapping
up on volume" and Claude returns the corresponding filter values, which the
user can apply to the scan controls with one click.
"""
from __future__ import annotations

import json
import re

# Filters Claude is allowed to set, with their session_state keys and bounds.
# Keeping this list explicit prevents the model from injecting arbitrary state.
_ALLOWED_FILTERS = {
    "universe": str,        # "SP500" | "NASDAQ" | "Combo"
    "min_price": float,
    "max_price": float,
    "min_dollar_vol": float,
    "top_n": int,
    "min_gap": float,
    "apply_gap_filter": bool,
    "premarket": bool,
    "afterhours": bool,
    "unusual_vol": bool,
    "include_ta": bool,
}

_SYSTEM_PROMPT = (
    "You convert a plain-English stock-screening request into a JSON object of "
    "filter values for a breakout scanner. Respond with ONLY a JSON object, no "
    "prose, no markdown fences. Allowed keys (omit any you can't infer):\n"
    '  universe: one of "SP500", "NASDAQ", "Combo"\n'
    "  min_price, max_price: dollars (numbers)\n"
    "  min_dollar_vol: minimum 20-day average dollar volume (number)\n"
    "  top_n: max results (integer)\n"
    "  min_gap: minimum gap percent (number)\n"
    "  apply_gap_filter, premarket, afterhours, unusual_vol, include_ta: booleans\n"
    'Also include a short "explanation" string describing what you set. '
    'Example: {"universe":"NASDAQ","max_price":20,"apply_gap_filter":true,'
    '"min_gap":2,"unusual_vol":true,"explanation":"NASDAQ under $20 gapping up '
    '2%+ on unusual volume"}'
)


def _coerce(key: str, value):
    """Coerce a model-supplied value to the expected type, or None if invalid."""
    expected = _ALLOWED_FILTERS.get(key)
    if expected is None:
        return None
    try:
        if expected is bool:
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"true", "1", "yes", "on"}
        if expected is int:
            return int(float(value))
        if expected is float:
            return float(value)
        if expected is str:
            v = str(value).strip()
            if key == "universe":
                v_up = v.upper()
                mapping = {"SP500": "SP500", "S&P500": "SP500", "SPX": "SP500",
                           "NASDAQ": "NASDAQ", "COMBO": "Combo"}
                return mapping.get(v_up)
            return v
    except (TypeError, ValueError):
        return None
    return None


def parse_screen_request(query: str) -> tuple[dict, str | None, str | None]:
    """Return (filters, explanation, error). Filters are validated + coerced."""
    if not (query or "").strip():
        return {}, None, "Type what you're looking for first."

    username = None
    try:
        import streamlit as st
        username = st.session_state.get("username")
    except Exception:
        username = None

    from ui.ai import ask_claude
    text, err = ask_claude(
        system=_SYSTEM_PROMPT,
        user=query.strip(),
        max_tokens=400,
        username=username,
        feature="nl_screener",
    )
    if err:
        return {}, None, err
    if not text:
        return {}, None, "Empty response from AI."

    # Strip any stray markdown fences, then grab the first {...} block.
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {}, None, "Could not parse the screen request. Try rephrasing."
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}, None, "Could not parse the screen request. Try rephrasing."

    explanation = raw.get("explanation") if isinstance(raw, dict) else None
    filters: dict = {}
    for key, value in (raw.items() if isinstance(raw, dict) else []):
        if key == "explanation":
            continue
        coerced = _coerce(key, value)
        if coerced is not None:
            filters[key] = coerced

    if not filters:
        return {}, explanation, "Couldn't infer any filters from that. Be more specific."
    return filters, explanation, None


def render_nl_screener() -> None:
    """Streamlit UI: NL input -> proposed filters -> apply to scan controls."""
    import streamlit as st

    st.markdown("#### 🗣️ Natural-language screener")
    st.caption('Describe what you want, e.g. "NASDAQ stocks under $20 gapping up on volume".')

    query = st.text_input(
        "Describe your screen",
        key="nl_screen_query",
        placeholder="e.g. liquid S&P 500 names with a 3%+ gap",
        label_visibility="collapsed",
    )

    if st.button("🔎 Interpret", key="nl_screen_run"):
        with st.spinner("Interpreting…"):
            filters, explanation, err = parse_screen_request(query)
        if err:
            st.warning(err)
        else:
            st.session_state["_nl_pending_filters"] = filters
            st.session_state["_nl_pending_explanation"] = explanation or ""

    pending = st.session_state.get("_nl_pending_filters")
    if pending:
        if st.session_state.get("_nl_pending_explanation"):
            st.info(st.session_state["_nl_pending_explanation"])
        st.write({k: v for k, v in pending.items()})
        c1, c2 = st.columns(2)
        if c1.button("✅ Apply these filters", key="nl_screen_apply"):
            for key, value in pending.items():
                st.session_state[key] = value
            st.session_state.pop("_nl_pending_filters", None)
            st.session_state.pop("_nl_pending_explanation", None)
            st.success("Filters applied. Run a scan to see results.")
            st.rerun()
        if c2.button("✖️ Discard", key="nl_screen_discard"):
            st.session_state.pop("_nl_pending_filters", None)
            st.session_state.pop("_nl_pending_explanation", None)
            st.rerun()
