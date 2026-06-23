"""Data-provider diagnostics panel for scan UI."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

from ui.scan_providers import (
    fetch_alpaca_snapshot_debug,
    get_alpaca_extended_last_prices,
    get_alpaca_headers,
)


def _banner(msg: str, level: str = "info") -> None:
    if level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    elif level == "error":
        st.error(msg)
    else:
        st.info(msg)


def _render_debug_payload(debug_payload: object) -> None:
    if not isinstance(debug_payload, dict):
        st.text_area(
            "Raw Alpaca response (non-JSON, truncated)",
            value=str(debug_payload)[:1200],
            height=200,
        )
        return

    debug_json = debug_payload
    st.write("Alpaca debug top-level keys:", list(debug_json.keys()))

    raw_snaps = debug_json.get("snapshots")
    if isinstance(raw_snaps, dict) and raw_snaps:
        snaps = raw_snaps
    elif isinstance(debug_json, dict):
        snaps = debug_json
    else:
        snaps = {}

    aapl_snap = snaps.get("AAPL") or {}

    st.write(
        "Alpaca debug 'snapshots' keys:",
        list(snaps.keys()) if isinstance(snaps, dict) else snaps,
    )
    st.write(
        "Alpaca debug AAPL snapshot keys:",
        list(aapl_snap.keys()) if isinstance(aapl_snap, dict) else aapl_snap,
    )
    st.text_area(
        "Raw Alpaca JSON (truncated)",
        value=str(debug_json)[:1200],
        height=200,
    )


def render_data_provider_diagnostics() -> None:
    """Render Alpaca / data provider diagnostics."""
    st.markdown("### 🧪 Data Provider Diagnostics")

    if not st.button(
        "Test Alpaca Market Data (AAPL)",
        key="btn_test_alpaca",
        width="stretch",
    ):
        return

    headers = get_alpaca_headers()
    if not headers:
        _banner(
            "Alpaca API keys are not configured in Streamlit secrets. "
            "Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY in .streamlit/secrets.toml.",
            "error",
        )
        return

    with st.spinner("Contacting Alpaca for AAPL snapshot..."):
        quotes = get_alpaca_extended_last_prices(["AAPL"])
    price = quotes.get("AAPL")
    if price is not None:
        st.success(f"✅ Alpaca Market Data OK. AAPL extended price: ${price:.2f}")
        return

    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        st.info("🟦 Market is closed (weekend). Extended-hours data is usually unavailable.")
    else:
        st.info("Market may be outside active extended-hours windows.")

    _banner(
        "Connected to Alpaca but no price was returned for AAPL. "
        "Attempting to show raw Alpaca response for AAPL below for debugging.",
        "warning",
    )

    try:
        status_code, debug_payload = fetch_alpaca_snapshot_debug("AAPL")
    except (RuntimeError, OSError, ValueError) as exc:
        st.error(f"Alpaca debug request failed: {exc}")
        return

    st.write("Alpaca debug HTTP status:", status_code)
    _render_debug_payload(debug_payload)
