"""Sidebar filters UI module."""

from typing import Tuple

import streamlit as st


def render_filters(tier) -> Tuple[float, float, float, int, int, int, bool, bool, bool, bool]:
    """Render the sidebar filters and return the selected values.

    Returns:
        (
            min_gap,
            min_price,
            max_price,
            top_n,
            max_nasdaq_scan,
            max_combo_scan,
            premarket,
            afterhours,
            unusual_vol,
            diagnostics,
        )
    """
    st.sidebar.markdown("## Filters")
    min_gap = st.sidebar.slider("Min Gap %", -10.0, 20.0, 1.0, 0.5)
    min_price = st.sidebar.number_input("Min Price", 0.5, 500.0, 1.0, 0.5)
    max_price = st.sidebar.number_input("Max Price", 1.0, 5000.0, 1000.0, 1.0)
    top_n = st.sidebar.slider("Top N Results", 5, tier.max_results, min(25, tier.max_results), 5)

    max_nasdaq_scan = st.sidebar.number_input(
        "Max NASDAQ tickers to scan",
        min_value=100,
        max_value=6000,
        value=1200,
        step=100,
        help="Caps NASDAQ universe to speed up scans. Applied to NASDAQ + Combo scans.",
    )

    max_combo_scan = st.sidebar.number_input(
        "Max Combo tickers to scan",
        min_value=100,
        max_value=6000,
        value=1000,
        step=100,
        help="Caps SP500+NASDAQ universe for Combo scans.",
    )

    premarket = st.sidebar.checkbox(
        "Include Premarket Scan",
        value=False,
        disabled=not getattr(tier, "can_premarket", False),
    )
    afterhours = st.sidebar.checkbox(
        "Include After-hours Scan",
        value=False,
        disabled=not getattr(tier, "can_afterhours", False),
    )
    unusual_vol = st.sidebar.checkbox(
        "Unusual Volume Filter",
        value=False,
        disabled=not getattr(tier, "can_unusual_volume", False),
    )

    st.sidebar.divider()
    diagnostics = st.sidebar.checkbox("Show diagnostics", value=True)

    return (
        float(min_gap),
        float(min_price),
        float(max_price),
        int(top_n),
        int(max_nasdaq_scan),
        int(max_combo_scan),
        bool(premarket),
        bool(afterhours),
        bool(unusual_vol),
        bool(diagnostics),
    )