"""Live activity feed for the Alpaca paper account (poll-on-refresh).

Premium-gated. On render, polls Alpaca for the user's open positions and recent
orders, upserts orders into the durable feed (db.paper_events), and shows live
positions (marked to Alpaca's current price) plus a recent-orders/fills feed. A
Refresh button re-polls; near-real-time while the user is viewing. Never raises.
"""
from __future__ import annotations

from typing import Any, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


def _user() -> str:
    try:
        return (st.session_state.get("username") or "").strip().lower()
    except Exception:
        return ""


def _can_paper_trade() -> bool:
    try:
        return bool((st.session_state.get("entitlements") or {}).get("can_paper_trade"))
    except Exception:
        return False


def _f(v) -> Optional[float]:
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def render_activity_feed(user_id: str) -> None:
    """Live positions + order feed for the connected paper account. Never raises."""
    if st is None or not user_id:
        return
    if not _can_paper_trade():
        return
    try:
        from db.paper_trading import get_paper_account, has_paper_account
    except Exception:
        return
    try:
        if not has_paper_account(user_id):
            return
    except Exception:
        return

    try:
        with st.expander("📡 Paper account — live activity", expanded=False):
            if st.button("🔄 Refresh", key="pe_refresh"):
                st.rerun()

            acct = None
            try:
                acct = get_paper_account(user_id)
            except Exception:
                acct = None
            if not acct:
                st.caption("Reconnect your paper account to see live activity.")
                return

            positions, orders = _poll(acct)
            # Persist orders into the durable feed, then read it back so the feed
            # survives reloads even if a later poll transiently fails.
            if orders:
                try:
                    from db.paper_events import sync_orders

                    sync_orders(user_id, orders)
                except Exception:
                    pass
            _render_positions(positions)
            _render_orders(user_id)
            st.caption(
                "Near-real-time: refreshes each time you open or reload. Paper "
                "account only — no real money."
            )
    except Exception:
        pass


def _poll(acct):
    positions = orders = None
    try:
        from data.alpaca_trading import get_orders, get_positions

        with st.spinner("Polling Alpaca…"):
            positions = get_positions(acct["api_key"], acct["api_secret"])
            orders = get_orders(acct["api_key"], acct["api_secret"], status="all", limit=25)
    except Exception:
        pass
    return positions, orders


def _render_positions(positions) -> None:
    try:
        st.markdown("**Open positions**")
        if not positions:
            st.caption("No open positions.")
            return
        rows = []
        for p in positions:
            plpc = _f(p.get("unrealized_plpc"))
            rows.append({
                "Symbol": p.get("symbol"),
                "Qty": p.get("qty"),
                "Avg entry": _money(p.get("avg_entry_price")),
                "Last": _money(p.get("current_price")),
                "Value": _money(p.get("market_value")),
                "Unreal. P&L": _money(p.get("unrealized_pl")),
                "P&L %": f"{plpc * 100:+.2f}%" if plpc is not None else "—",
            })
        st.dataframe(rows, hide_index=True, width="stretch")
    except Exception:
        pass


def _render_orders(user_id: str) -> None:
    try:
        from db.paper_events import list_events

        events = list_events(user_id, limit=25)
        st.markdown("**Recent orders**")
        if not events:
            st.caption("No orders yet.")
            return
        rows = []
        for e in events:
            rows.append({
                "Symbol": e.get("symbol"),
                "Side": (e.get("side") or "").upper(),
                "Qty": _int(e.get("qty")),
                "Type": e.get("order_type"),
                "Status": e.get("status"),
                "Filled @": _money(e.get("filled_avg_price")),
                "Updated": _when(e.get("updated_at")),
            })
        st.dataframe(rows, hide_index=True, width="stretch")
    except Exception:
        pass


def _money(v) -> str:
    f = _f(v)
    return f"${f:,.2f}" if f is not None else "—"


def _int(v) -> Any:
    f = _f(v)
    return int(f) if f is not None else "—"


def _when(v) -> str:
    try:
        return f"{v:%b %d %H:%M}"
    except Exception:
        return str(v or "")
