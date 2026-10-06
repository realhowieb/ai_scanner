"""US market universe for Custom Scan (Premium, 2026-10-06): the same list of
every tradable U.S. stock the automatic full-market scans use."""
from __future__ import annotations

from typing import List

import streamlit as st


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def load_us_market_universe() -> List[str]:
    """Every active, tradable U.S.-listed stock: the same list the automatic
    full-market scans use (data.us_market_universe, live from Alpaca with a
    last-known-good cache). Raises when neither is available, so a failure is
    not cached for hours (the scan buttons call us_market_symbols())."""
    from data.us_market_universe import build_us_market_universe

    symbols = list(build_us_market_universe().get("symbols") or [])
    if not symbols:
        raise RuntimeError("US market list unavailable (no live list and no cached copy)")
    return symbols


US_MARKET_UNAVAILABLE = ("The US market list isn't available right now. Try again in a few minutes, "
                         "or run the Combo scan.")


def us_market_symbols() -> List[str]:
    """load_us_market_universe for the scan buttons: [] instead of an error box."""
    try:
        return load_us_market_universe()
    except Exception:
        return []
