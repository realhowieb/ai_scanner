"""Sidebar filters UI module."""

from typing import Tuple

import streamlit as st


def render_filters(tier) -> Tuple[float, float, float, int, int, int, bool, bool, bool, bool, float, bool, bool]:
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
            min_dollar_vol,
            include_ta,
            apply_gap_filter,
        )
    """
    st.sidebar.markdown("## Filters")
    # Seed defaults from session_state if present (loaded from user_settings)
    default_min_gap = float(st.session_state.get("min_gap", 1.0))
    default_min_price = float(st.session_state.get("min_price", 1.0))
    default_max_price = float(st.session_state.get("max_price", 1000.0))
    default_top_n = int(st.session_state.get("top_n", min(25, tier.max_results)))
    default_max_nasdaq_scan = int(st.session_state.get("max_nasdaq_scan", 1200))
    default_max_combo_scan = int(st.session_state.get("max_combo_scan", 1000))
    default_premarket = bool(st.session_state.get("premarket", False))
    default_afterhours = bool(st.session_state.get("afterhours", False))
    default_unusual_vol = bool(st.session_state.get("unusual_vol", False))
    default_min_dollar_vol = float(st.session_state.get("min_dollar_vol", 1_000_000.0))
    default_include_ta = bool(st.session_state.get("include_ta", True))
    default_apply_gap_filter = bool(st.session_state.get("apply_gap_filter", False))
    default_diagnostics = bool(st.session_state.get("show_diagnostics_ui", True))
    min_gap = st.sidebar.slider(
        "Min Gap %",
        -10.0,
        20.0,
        default_min_gap,
        0.5,
        key="min_gap",
    )
    min_price = st.sidebar.number_input(
        "Min Price",
        0.5,
        500.0,
        default_min_price,
        0.5,
        key="min_price",
    )
    max_price = st.sidebar.number_input(
        "Max Price",
        1.0,
        5000.0,
        default_max_price,
        1.0,
        key="max_price",
    )
    top_n = st.sidebar.slider(
        "Top N Results",
        5,
        tier.max_results,
        default_top_n,
        5,
        key="top_n",
    )

    max_nasdaq_scan = st.sidebar.number_input(
        "Max NASDAQ tickers to scan",
        min_value=100,
        max_value=6000,
        value=default_max_nasdaq_scan,
        step=100,
        key="max_nasdaq_scan",
        help="Caps NASDAQ universe to speed up scans. Applied to NASDAQ + Combo scans.",
    )

    max_combo_scan = st.sidebar.number_input(
        "Max Combo tickers to scan",
        min_value=100,
        max_value=6000,
        value=default_max_combo_scan,
        step=100,
        key="max_combo_scan",
        help="Caps SP500+NASDAQ universe for Combo scans.",
    )

    # Minimum Dollar Volume
    min_dollar_vol = st.sidebar.number_input(
        "Min Dollar Volume",
        min_value=0.0,
        value=default_min_dollar_vol,
        step=100_000.0,
        key="min_dollar_vol",
        help="Only include stocks with minimum dollar volume."
    )

    premarket = st.sidebar.checkbox(
        "Include Premarket Scan",
        value=default_premarket,
        key="premarket",
        disabled=not getattr(tier, "can_premarket", False),
    )
    afterhours = st.sidebar.checkbox(
        "Include After-hours Scan",
        value=default_afterhours,
        key="afterhours",
        disabled=not getattr(tier, "can_afterhours", False),
    )
    unusual_vol = st.sidebar.checkbox(
        "Unusual Volume Filter",
        value=default_unusual_vol,
        key="unusual_vol",
        disabled=not getattr(tier, "can_unusual_volume", False),
    )

    include_ta = st.sidebar.checkbox(
        "Include Technical Indicators",
        value=default_include_ta,
        key="include_ta",
    )

    apply_gap_filter = st.sidebar.checkbox(
        "Apply Gap Filter",
        value=default_apply_gap_filter,
        key="apply_gap_filter",
    )

    st.sidebar.divider()
    diagnostics = st.sidebar.checkbox(
        "Show diagnostics",
        value=default_diagnostics,
        key="show_diagnostics_ui",
    )

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
        float(min_dollar_vol),
        bool(include_ta),
        bool(apply_gap_filter),
    )
