"""Custom Scan page body: filters, scan buttons and the 3-step scanner.

Moved out of the Scanner page (owner, 2026-10-06), which now shows only the
latest results and links here. The scan itself still stores its results in the
session (`results_df`); when one finishes, this page opens the Scanner, which
shows "your scan" with a button back to the latest market scan.

Watchlist "Run Watchlist Scan" / "View as table" on the Scanner and My Stocks
pages hand off here with `request_watchlist_scan()`; the handoff flag is consumed
by ui.scans.render_scan_controls.
"""
from __future__ import annotations

from typing import Any, Dict

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

PAGE = "pages/custom_scan.py"
SCANNER_PAGE = "app.py"
_NO_TOOLS = (False, False, False, False, False, "", False)


def request_watchlist_scan(*, view: bool, scan_all: bool) -> None:
    """Open Custom Scan and run (or view) the active watchlist there."""
    st.session_state["_wl_pending_scan"] = "view" if view else "run"
    st.session_state["_wl_pending_scan_all"] = bool(scan_all)
    st.switch_page(PAGE)


def handle_watchlist_tools(username: str) -> None:
    """Act on the watchlist tool buttons rendered on a page without scan
    controls: add/remove/clear update the list here; run/view hand off to
    Custom Scan. Reads the state ui.watchlists.render_watchlists_panel wrote."""
    view, run, clear, add, remove, symbol, scan_all = st.session_state.get("_wl_tools_state", _NO_TOOLS)
    st.session_state["_wl_tools_state"] = _NO_TOOLS  # one click acts once
    if add or remove or clear:
        from ui.watchlists import handle_active_watchlist_actions

        handle_active_watchlist_actions(
            view_watchlist=False, run_watchlist=False, clear_watchlist=clear,
            add_symbol=add, remove_symbol=remove, symbol=symbol, username=username,
            do_scan=lambda *_a, **_k: None, banner=lambda msg, kind="info": st.toast(msg),
            scan_all=bool(scan_all),
        )
        st.rerun()
    if run or view:
        request_watchlist_scan(view=bool(view and not run), scan_all=bool(scan_all))


def _earnings_panel():
    try:
        from ui.earnings import render_earnings_this_week_panel

        return render_earnings_this_week_panel
    except Exception:
        def _unavailable(*_a, **_k):
            st.info("Earnings panel not available right now.")
        return _unavailable


def plan_tier(tier: Any, tier_key: str, username: str) -> Any:
    """The full plan object the filters need (row cap, extended hours). The
    session's "tier" can be a lightweight stand-in (key/name only) when the
    database plan differs from the legacy lookup, so build it from the key."""
    try:
        from auth.tiering import Tier, get_user_tier
    except Exception:
        return tier
    if isinstance(tier, Tier) and tier.key == tier_key:
        return tier
    return get_user_tier(username, {username: {"tier": tier_key or "basic"}})


def render_custom_scan(username: str, tier: Any, flags: Dict[str, Any]) -> None:
    """Filters, market buttons, single-ticker and watchlist scans, 3-step scanner."""
    tier = plan_tier(tier, str(st.session_state.get("tier_key") or getattr(tier, "key", "") or "basic"), username)
    from ui.app_runtime import get_market_session, render_active_filters_summary
    from ui.app_user_profile import apply_admin_scan_caps, load_saved_user_settings
    from ui.earnings_results import render_earnings_controls
    from ui.filters import render_filters
    from ui.scans import render_scan_controls, render_three_step_scanner
    from ui.user_settings import render_user_settings_footer

    try:
        from db.user_settings import get_user_settings, upsert_user_settings
    except Exception:
        get_user_settings = upsert_user_settings = None

    is_admin = bool(st.session_state.get("is_admin"))
    load_saved_user_settings(username=username, get_user_settings=get_user_settings, is_admin=is_admin)

    filters_box = st.container(border=True)
    filters_box.markdown("#### Scan filters")
    # Pre-clamp diagnostics BEFORE filters render widgets.
    # Streamlit forbids mutating widget-bound session_state keys after widget creation.
    if not flags.get("can_diagnostics"):
        st.session_state["show_diagnostics_ui"] = False

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
    ) = render_filters(tier, container=filters_box)
    # Enforce admin-only diagnostics (even if UI/modules accidentally expose it)
    if not flags.get("can_diagnostics"):
        diagnostics = False
    render_active_filters_summary(
        universe=st.session_state.get("universe"),
        min_price=float(min_price),
        max_price=float(max_price),
        min_dollar_vol=float(min_dollar_vol),
        top_n=int(top_n),
        premarket=bool(premarket),
        afterhours=bool(afterhours),
        include_ta=bool(include_ta),
        unusual_vol=bool(unusual_vol),
        apply_gap_filter=bool(apply_gap_filter),
        min_gap=float(min_gap),
        max_nasdaq_scan=int(max_nasdaq_scan),
        max_combo_scan=int(max_combo_scan),
    )

    # -------- Market session gating for extended-hours toggles --------
    session = get_market_session()
    filters_box.caption(f"Market session (US/Eastern): {session.capitalize()}")
    if premarket and session != "premarket":
        # Clamp to regular mode for this run; avoid mutating widget state directly.
        premarket = False
        filters_box.info(
            "Premarket scans only run between 4:00–9:30am ET on trading days. "
            "The toggle has been reset to Regular mode for this scan."
        )
    if afterhours and session != "afterhours":
        afterhours = False
        filters_box.info(
            "After-hours scans only run between 4:00–8:00pm ET on trading days. "
            "The toggle has been reset to Regular mode for this scan."
        )

    render_user_settings_footer(
        username,
        min_price=float(min_price) if min_price is not None else None,
        max_price=float(max_price) if max_price is not None else None,
        diagnostics=bool(diagnostics) if diagnostics is not None else None,
        get_user_settings=get_user_settings,
        upsert_user_settings=upsert_user_settings,
        container=filters_box,
    )

    # Admin can test at scale even if UI defaults are capped.
    max_nasdaq_scan, max_combo_scan, top_n = apply_admin_scan_caps(
        max_nasdaq_scan=max_nasdaq_scan,
        max_combo_scan=max_combo_scan,
        top_n=top_n,
        is_admin=is_admin,
    )

    render_earnings_controls(flags=flags, render_earnings_this_week_panel=_earnings_panel())

    render_scan_controls(
        can_scan_sp500=flags["can_scan_sp500"],
        can_scan_nasdaq=flags["can_scan_nasdaq"],
        max_nasdaq_scan=int(max_nasdaq_scan) if max_nasdaq_scan is not None else 0,
        max_combo_scan=int(max_combo_scan) if max_combo_scan is not None else 0,
        min_gap=float(min_gap),
        apply_gap_filter=bool(apply_gap_filter),
        min_price=float(min_price),
        max_price=float(max_price),
        top_n=int(top_n) if top_n is not None else 0,
        premarket=bool(premarket),
        afterhours=bool(afterhours),
        unusual_vol=bool(unusual_vol),
        diagnostics=bool(diagnostics),
        username=username,
        can_scan_us_market=bool(flags.get("can_full_universe")),  # Premium
    )

    # A finished scan opens the Scanner, where its results render.
    if st.session_state.pop("force_results_refresh", False):
        open_scanner_with_results()

    render_three_step_scanner()


def open_scanner_with_results() -> None:
    try:
        from ui.results import get_results_df

        get_results_df.clear()  # works if get_results_df is @st.cache_data
    except Exception:
        pass
    st.session_state["hsf_scan_just_ran"] = True
    st.switch_page(SCANNER_PAGE)
