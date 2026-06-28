"""Sidebar filters UI module."""

from typing import Tuple

import streamlit as st

from auth.tiering import has_min_tier  # fallback only; entitlements are preferred


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
    # Centralized entitlements (preferred). Fallback to tier checks if missing.
    ent = st.session_state.get("entitlements") or {}

    # 🗣️ Premium: natural-language screener (sets filters from plain English).
    if ent.get("can_ai_notes"):
        try:
            from ui.ai import is_configured
            if is_configured():
                with st.sidebar.expander("🤖 AI Assistant", expanded=False):
                    from ui.ai_screener import render_nl_screener
                    render_nl_screener()
        except Exception:
            pass

    def _entitled(feature: str, *, fallback_min_tier: str | None = None) -> bool:
        v = ent.get(feature)
        if v is not None:
            return bool(v)
        if fallback_min_tier is None:
            return False
        try:
            return bool(has_min_tier(tier, fallback_min_tier))
        except Exception:
            return False

    is_pro_plus = _entitled("can_scan_nasdaq", fallback_min_tier="pro")
    is_premium_plus = _entitled("can_full_universe", fallback_min_tier="premium")
    can_diagnostics = _entitled("can_diagnostics", fallback_min_tier=None)
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
    # Default $5M liquidity floor (matches the scheduled cron) so manual scans
    # exclude illiquid micro-cap/pump names out of the box.
    default_min_dollar_vol = float(st.session_state.get("min_dollar_vol", 5_000_000.0))
    # Include Technical Indicators and Show diagnostics pull from session_state keys
    # that can be populated from user_settings. Default them to False when not set.
    default_include_ta = bool(st.session_state.get("include_ta", False))
    default_apply_gap_filter = bool(st.session_state.get("apply_gap_filter", False))
    default_diagnostics = bool(st.session_state.get("show_diagnostics_ui", False))
    # Initialize min_gap through session_state if not already set
    if "min_gap" not in st.session_state:
        st.session_state["min_gap"] = default_min_gap

    # Pro+ tiers get advanced filters (TA + Gap); Basic sees them disabled.

    min_gap_disabled = (not is_pro_plus) or (not st.session_state.get("apply_gap_filter", False))

    min_gap = st.sidebar.slider(
        "Min Gap %",
        0.0,
        20.0,
        step=0.5,
        key="min_gap",
        disabled=min_gap_disabled,
        help="Minimum gap-up percentage required to include a stock.",
    )

    if not is_pro_plus:
        st.sidebar.caption("🔒 Pro+ feature")
        st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # Initialize min_price through session_state if not already set
    if "min_price" not in st.session_state:
        st.session_state["min_price"] = default_min_price

    min_price = st.sidebar.number_input(
        "Min Price",
        0.5,
        500.0,
        step=0.5,
        key="min_price",
    )

    # Initialize max_price through session_state if not already set
    if "max_price" not in st.session_state:
        st.session_state["max_price"] = default_max_price

    max_price = st.sidebar.number_input(
        "Max Price",
        1.0,
        5000.0,
        step=1.0,
        key="max_price",
    )

    # Initialize top_n through session_state if not already set
    if "top_n" not in st.session_state:
        st.session_state["top_n"] = default_top_n

    top_n = st.sidebar.slider(
        "Top N Results",
        5,
        tier.max_results,
        step=5,
        key="top_n",
    )

    # Tier-based cap for how many NASDAQ tickers can be scanned
    # Basic: 1000, Pro: 4000, Premium+ (and higher): 6000
    if is_premium_plus:
        nasdaq_cap = 6000
    elif is_pro_plus:
        nasdaq_cap = 4000
    else:
        nasdaq_cap = 1000

    # Clamp the default NASDAQ max to the tier cap and a sane minimum
    if default_max_nasdaq_scan > nasdaq_cap:
        default_max_nasdaq_scan = nasdaq_cap
    if default_max_nasdaq_scan < 100:
        default_max_nasdaq_scan = 100

    # Initialize max_nasdaq_scan through session_state if not already set,
    # and ensure it respects the tier cap on every run.
    if "max_nasdaq_scan" not in st.session_state:
        st.session_state["max_nasdaq_scan"] = default_max_nasdaq_scan
    else:
        try:
            current_max_nasdaq = int(st.session_state["max_nasdaq_scan"])
        except Exception:
            current_max_nasdaq = default_max_nasdaq_scan
        if current_max_nasdaq > nasdaq_cap:
            current_max_nasdaq = nasdaq_cap
        if current_max_nasdaq < 100:
            current_max_nasdaq = 100
        st.session_state["max_nasdaq_scan"] = current_max_nasdaq

    if not is_pro_plus:
        max_nasdaq_scan = st.sidebar.number_input(
            "Max NASDAQ tickers to scan",
            min_value=100,
            max_value=nasdaq_cap,
            step=100,
            key="max_nasdaq_scan",
            disabled=True,
            help="NASDAQ scan limits are a Pro+ feature.",
        )
        st.sidebar.caption("🔒 Pro+ feature – NASDAQ scans are not available on Basic.")
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
    else:
        max_nasdaq_scan = st.sidebar.number_input(
            "Max NASDAQ tickers to scan",
            min_value=100,
            max_value=nasdaq_cap,
            step=100,
            key="max_nasdaq_scan",
            help="Caps NASDAQ universe to speed up scans. Applied to NASDAQ + Combo scans.",
        )
        # Tiny hint so users know why they can't go higher
        st.sidebar.caption(f"🔒 Your plan caps NASDAQ scans at {nasdaq_cap} tickers.")

    # Initialize max_combo_scan through session_state if not already set
    if "max_combo_scan" not in st.session_state:
        st.session_state["max_combo_scan"] = default_max_combo_scan

    if not is_pro_plus:
        max_combo_scan = st.sidebar.number_input(
            "Max Combo tickers to scan",
            min_value=100,
            max_value=6000,
            step=100,
            key="max_combo_scan",
            disabled=True,
            help="Combo scans are a Pro+ feature.",
        )
        st.sidebar.caption("🔒 Pro+ feature – Combo scans are not available on Basic.")
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
    else:
        max_combo_scan = st.sidebar.number_input(
            "Max Combo tickers to scan",
            min_value=100,
            max_value=6000,
            step=100,
            key="max_combo_scan",
            help="Caps SP500+NASDAQ universe for Combo scans.",
        )

    # Initialize min_dollar_vol through session_state if not already set
    if "min_dollar_vol" not in st.session_state:
        st.session_state["min_dollar_vol"] = default_min_dollar_vol

    min_dollar_vol = st.sidebar.number_input(
        "Min Dollar Volume",
        min_value=0.0,
        step=100_000.0,
        key="min_dollar_vol",
        help="Only include stocks with minimum dollar volume."
    )

    # Session mode selector (mutually exclusive)
    # Derive a default mode based on saved session_state flags.
    if default_premarket:
        default_session_mode = "Premarket"
    elif default_afterhours:
        default_session_mode = "After-hours"
    else:
        default_session_mode = "Regular"

    # Build options based on tier capabilities.
    session_options = ["Regular"]
    if getattr(tier, "can_premarket", False):
        session_options.append("Premarket")
    if getattr(tier, "can_afterhours", False):
        session_options.append("After-hours")

    # Ensure the default is valid given the available options.
    if default_session_mode not in session_options:
        default_session_mode = "Regular"
    default_index = session_options.index(default_session_mode)

    session_mode = st.sidebar.radio(
        "Session mode",
        options=session_options,
        index=default_index,
        key="session_mode",
        help="Choose which market session to use for price data.",
    )

    premarket = session_mode == "Premarket"
    afterhours = session_mode == "After-hours"

    # Optional: keep these in session_state so defaults persist smoothly.
    try:
        st.session_state["premarket"] = premarket
        st.session_state["afterhours"] = afterhours
    except Exception:
        pass

    # Initialize unusual_vol through session_state if not already set
    if "unusual_vol" not in st.session_state:
        st.session_state["unusual_vol"] = default_unusual_vol

    unusual_vol = st.sidebar.checkbox(
        "Unusual Volume Filter",
        key="unusual_vol",
        disabled=not getattr(tier, "can_unusual_volume", False),
    )
    if not getattr(tier, "can_unusual_volume", False):
        st.sidebar.caption("🔒 Pro+ feature")
        st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # Initialize include_ta through session_state if not already set
    if "include_ta" not in st.session_state:
        st.session_state["include_ta"] = default_include_ta

    # Basic users should never have TA enabled, even if a profile says so
    if not is_pro_plus:
        st.session_state["include_ta"] = False

    include_ta = st.sidebar.checkbox(
        "Include Technical Indicators",
        key="include_ta",
        disabled=not is_pro_plus,
    )
    if not is_pro_plus:
        st.sidebar.caption("🔒 Pro+ feature")
        st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # Initialize apply_gap_filter through session_state if not already set
    if "apply_gap_filter" not in st.session_state:
        st.session_state["apply_gap_filter"] = default_apply_gap_filter

    # Basic users should never have gap filter enabled, even if a profile says so
    if not is_pro_plus:
        st.session_state["apply_gap_filter"] = False

    apply_gap_filter = st.sidebar.checkbox(
        "Apply Gap Filter",
        key="apply_gap_filter",
        disabled=not is_pro_plus,
    )
    if not is_pro_plus:
        st.sidebar.caption("🔒 Pro+ feature")
        st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # Initialize show_diagnostics_ui through session_state if not already set
    if "show_diagnostics_ui" not in st.session_state:
        st.session_state["show_diagnostics_ui"] = default_diagnostics

    st.sidebar.divider()
    diagnostics = st.sidebar.checkbox(
        "Show diagnostics",
        key="show_diagnostics_ui",
        disabled=not can_diagnostics,
    )
    if not can_diagnostics:
        st.sidebar.caption("🔒 Admin feature")
        st.sidebar.markdown("<br>", unsafe_allow_html=True)

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
