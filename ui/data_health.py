"""Market-data health banner: surface a rejected/broken Alpaca key in-app.

When the app's ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are present but Alpaca
rejects them (401) or the data feed is otherwise down, scans silently return
empty and the only signal is a flood of log errors. This renders a visible
banner so the operator notices immediately instead of via the logs.

The probe is a single cached AAPL quote (2-min TTL) so it costs almost nothing
and never spams Alpaca. Best-effort — never raises into the app.
"""
from __future__ import annotations

from typing import Tuple

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


def _probe() -> Tuple[bool, bool]:
    """Return (configured, working): are keys present, and does a live quote work?"""
    configured = False
    try:
        from data.alpaca_config import get_alpaca_config

        configured = get_alpaca_config() is not None
    except Exception:
        configured = False
    if not configured:
        return False, False
    try:
        from market_data import get_latest_quotes

        quotes = get_latest_quotes(["AAPL"]) or {}
        info = quotes.get("AAPL")
        working = isinstance(info, dict) and info.get("last") is not None
        return True, bool(working)
    except Exception:
        return True, False


if st is not None:

    @st.cache_data(ttl=120, show_spinner=False)
    def _probe_cached() -> Tuple[bool, bool]:
        return _probe()

else:  # pragma: no cover
    _probe_cached = _probe


def render_data_health_banner(*, is_admin: bool = False) -> None:
    """Warn when Alpaca market data is unavailable. Never raises.

    Admins get the actionable cause + fix; everyone else gets a soft notice so
    they understand why quotes/scan results may be blank.
    """
    if st is None:
        return
    try:
        configured, working = _probe_cached()
    except Exception:
        return
    if working:
        return  # healthy — no banner

    if not is_admin:
        st.warning(
            "⚠️ Live market data is temporarily unavailable — quotes and scan "
            "results may be blank or stale. This is being looked into."
        )
        return

    if not configured:
        st.error(
            "⚠️ **Alpaca market-data key not configured.** Set `ALPACA_API_KEY_ID` "
            "and `ALPACA_API_SECRET_KEY` in the app secrets — scans and quotes "
            "can't fetch data until then."
        )
    else:
        st.error(
            "⚠️ **Alpaca market-data key is being rejected (401 / no data).** The "
            "key is set but the live probe failed — it was likely rotated or "
            "invalidated. Refresh `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` in "
            "the app secrets (Streamlit Cloud **and** GitHub Actions). Until then, "
            "scans and quotes come back empty."
        )
