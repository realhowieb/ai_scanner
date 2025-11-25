from __future__ import annotations

from typing import List
import pandas as pd
import streamlit as st

# Flexible imports so this works whether you run as a package (ai_scanner.*)
# or straight from the repo root.
try:  # package-style
    from ai_scanner.utils.symbols import sanitize_ticker_list, as_ticker_list
except Exception:  # local-style
    from utils.symbols import sanitize_ticker_list, as_ticker_list

try:
    from ai_scanner.data.fetch import load_sp500_tickers, load_sp600_tickers
except Exception:
    from data.fetch import load_sp500_tickers, load_sp600_tickers

try:
    from ai_scanner.data.market import fetch_market_heat
except Exception:
    from data.market import fetch_market_heat


_HEAT_OPTIONS = {
    "Trending": "trending",
    "Most Active": "most_active",
    "Top Gainers": "gainers",
}


def _render_custom_input(key_prefix: str) -> list[str]:
    txt = st.text_area(
        "Paste tickers (comma, space or newline separated)",
        key=f"{key_prefix}_custom_text",
        height=110,
        placeholder="AAPL, MSFT, NVDA\nTSLA AMZN ...",
    )
    raw = [t.strip() for t in txt.replace("\n", ",").replace(" ", ",").split(",") if t.strip()]
    return sanitize_ticker_list(raw)


def _render_heat_picker(kind_label: str, key_prefix: str) -> list[str]:
    cols = st.columns([1, 1, 1])
    with cols[0]:
        limit = st.number_input("Limit", min_value=10, max_value=1000, value=200, step=10,
                                key=f"{key_prefix}_limit")
    with cols[1]:
        refresh = st.button("Refresh", key=f"{key_prefix}_refresh")
    if refresh:
        st.session_state.pop(f"{key_prefix}_cache", None)

    cache_key = f"{key_prefix}_cache"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = fetch_market_heat(_HEAT_OPTIONS[kind_label], int(limit))
    return st.session_state.get(cache_key, [])[: int(limit)]


def _render_index_picker(which: str, key_prefix: str) -> list[str]:
    if which == "S&P 500":
        try:
            tickers = load_sp500_tickers()
        except Exception:
            tickers = []
    else:
        try:
            tickers = load_sp600_tickers()
        except Exception:
            tickers = []
    return sanitize_ticker_list(as_ticker_list(tickers))


def render_universe_picker(key_prefix: str = "universe", default_source: str = "S&P 500") -> list[str]:
    """Render a compact universe selector and return the selected tickers list.

    Parameters
    ----------
    key_prefix : str
        Unique key prefix for Streamlit widget state.
    default_source : str
        One of ["S&P 500", "S&P 600", "Trending", "Most Active", "Top Gainers", "Custom"].
    """
    st.subheader("Universe")

    options: List[str] = [
        "S&P 500",
        "S&P 600",
        "Trending",
        "Most Active",
        "Top Gainers",
        "Custom",
    ]
    source = st.selectbox(
        "Source",
        options,
        index=max(0, options.index(default_source) if default_source in options else 0),
        key=f"{key_prefix}_source",
    )

    if source == "Custom":
        tickers = _render_custom_input(key_prefix)
    elif source in _HEAT_OPTIONS:
        tickers = _render_heat_picker(source, key_prefix)
    else:
        tickers = _render_index_picker(source, key_prefix)

    # Optional limiter for large universes
    limit = st.slider("Preview limit", min_value=50, max_value=1000, value=min(200, len(tickers) or 200),
                      step=50, key=f"{key_prefix}_preview_limit")

    preview_df = pd.DataFrame({"Ticker": tickers[:limit]})
    st.dataframe(preview_df, use_container_width=True, height=min(500, 30 + 24 * max(1, len(preview_df))))

    st.caption(f"Selected: {len(tickers):,} tickers")

    # Store final list (so other pages can consume without recomputing widgets)
    st.session_state[f"{key_prefix}_final_universe"] = tickers
    return tickers


__all__ = ["render_universe_picker"]