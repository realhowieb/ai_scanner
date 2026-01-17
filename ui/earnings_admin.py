

"""Admin-only Earnings tools.

This file intentionally contains *no* scan execution logic.
Earnings refresh must never run inside the scan flow (ui/scans.py).

What this provides:
- Admin-only button to refresh earnings for a chosen universe (SP500/NASDAQ/custom)
- Lightweight single-ticker lookup to verify DB contents

It is safe to import in Streamlit pages.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import streamlit as st


def _is_admin() -> bool:
    # Support both explicit flag and tier naming.
    if bool(st.session_state.get("is_admin", False)):
        return True
    tier = str(st.session_state.get("tier") or "").strip().lower()
    return tier == "admin"


def _normalize_symbols(symbols: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for s in symbols:
        if not s:
            continue
        sym = str(s).strip().upper()
        if not sym:
            continue
        if sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _get_universe_from_session(label: str) -> List[str]:
    """Best-effort: pull an existing universe list from session state."""
    keys = []
    l = label.strip().lower()
    if l in ("sp500", "s&p500", "s&p 500"):
        keys = [
            "sp500_universe",
            "sp500_tickers",
            "sp500",
            "universe_sp500",
        ]
    elif l in ("nasdaq", "nasdaq100", "nasdaq 100"):
        keys = [
            "nasdaq_universe",
            "nasdaq_tickers",
            "nasdaq",
            "universe_nasdaq",
        ]

    for k in keys:
        v = st.session_state.get(k)
        if isinstance(v, (list, tuple)):
            return _normalize_symbols(list(v))

    return []


def _import_repo() -> Tuple[Optional[Callable], Optional[Callable]]:
    """Return (populate_earnings_calendar, get_earnings_for_symbols) if available."""
    populate = None
    get_map = None

    # Prefer db.earnings (your repo functions live here)
    try:
        from db.earnings import populate_earnings_calendar as _pop  # type: ignore

        populate = _pop
    except Exception:
        populate = None

    # Optional getter (names vary across versions)
    try:
        from db.earnings import get_earnings_for_symbols as _get  # type: ignore

        get_map = _get
    except Exception:
        try:
            from db.earnings import get_earnings_map as _get  # type: ignore

            get_map = _get
        except Exception:
            get_map = None

    return populate, get_map


def render_earnings_admin() -> None:
    """Render the admin-only Earnings panel."""

    if not _is_admin():
        st.info("🔒 Admin only")
        return

    st.subheader("📅 Earnings Admin")
    st.caption(
        "Run earnings refresh manually. This does NOT run during scans (by design), so scans post results fast."
    )

    populate, get_map = _import_repo()

    if populate is None:
        st.error(
            "Earnings repo functions not found. Expected `db/earnings.py` to export `populate_earnings_calendar(...)`."
        )
        return

    colA, colB = st.columns([1, 1])

    with colA:
        universe = st.selectbox(
            "Universe",
            ["SP500", "NASDAQ", "Custom"],
            index=0,
            key="earn_admin_universe",
        )

    custom_raw = ""
    if universe == "Custom":
        custom_raw = st.text_area(
            "Paste tickers (comma / space / newline separated)",
            value="",
            height=120,
            key="earn_admin_custom_tickers",
        )

    limit = st.number_input(
        "Limit symbols (safety)",
        min_value=25,
        max_value=12000,
        value=250,
        step=25,
        key="earn_admin_limit",
        help="Keep this small if Yahoo/finance is flaky. Increase when stable.",
    )

    def _resolve_symbols() -> List[str]:
        if universe == "Custom":
            raw = custom_raw.replace(",", " ")
            syms = [x for x in raw.split() if x.strip()]
            return _normalize_symbols(syms)[: int(limit)]

        syms = _get_universe_from_session(universe)
        if not syms:
            # If the universe isn't loaded in session state yet, tell the admin.
            st.warning(
                f"No {universe} universe found in session_state yet. Run a scan once (or load universe) and try again."
            )
            return []
        return syms[: int(limit)]

    with colB:
        st.markdown("#### Actions")
        run = st.button("Run earnings refresh now", key="earn_admin_run", width="stretch")

    if run:
        symbols = _resolve_symbols()
        if not symbols:
            st.stop()

        with st.spinner(f"Refreshing earnings for {len(symbols)} symbols…"):
            try:
                # populate_earnings_calendar should internally upsert to DB.
                populate(symbols)
                st.success(f"✅ Earnings refresh complete for {len(symbols)} symbols")
            except Exception as e:
                st.error(f"Earnings refresh failed: {e}")

    st.divider()
    st.markdown("#### Quick verify (DB)")

    ticker = st.text_input("Ticker", value="AAPL", key="earn_admin_verify_ticker")
    if st.button("Lookup earnings in DB", key="earn_admin_lookup", width="stretch"):
        sym = _normalize_symbols([ticker])
        if not sym:
            st.warning("Enter a ticker.")
            st.stop()

        if get_map is None:
            st.info("No DB lookup helper exported (optional). Refresh still works.")
            st.stop()

        try:
            m = get_map(sym)  # type: ignore[misc]
            st.json(m if isinstance(m, dict) else {"result": m})
        except Exception as e:
            st.error(f"Lookup failed: {e}")


# Backwards-compatible alias if older pages import a different name.
render_earnings_panel = render_earnings_admin