"""Trade journal panel: open positions with live P&L, close/delete, stats.

Trades enter via the "Log this trade" button on a result's trade plan. Open
positions are marked to the latest quote at render; closed trades feed a
simple win-rate / average-return summary. Degrades to nothing when Neon is
unavailable — never raises into the app.
"""
from __future__ import annotations

from typing import Dict, Optional

import streamlit as st


def _live_prices(tickers) -> Dict[str, float]:
    try:
        from market_data import get_latest_quotes

        quotes = get_latest_quotes(sorted(set(tickers))) or {}
        return {
            sym: float(info["last"])
            for sym, info in quotes.items()
            if isinstance(info, dict) and info.get("last") is not None
        }
    except Exception:
        return {}


def render_journal_panel(user_id: str) -> None:
    if not user_id:
        return
    try:
        from db.trades import close_trade, delete_trade, journal_stats, list_trades

        trades = list_trades(user_id)
    except Exception:
        return
    if not trades:
        return  # panel appears once the first trade is logged

    with st.expander(f"📓 Trade journal ({len(trades)})", expanded=False):
        try:
            stats = journal_stats(user_id)
        except Exception:
            stats = None
        if stats:
            avg = stats.get("avg_return_pct")
            avg_s = f" · avg {avg:+.1f}%" if avg is not None else ""
            st.caption(
                f"Closed: **{stats['wins']}/{stats['closed']}** winners{avg_s}. "
                "Past performance is not indicative of future results."
            )

        open_trades = [t for t in trades if not t.get("closed_at")]
        live = _live_prices([t["ticker"] for t in open_trades]) if open_trades else {}

        for t in trades:
            is_open = not t.get("closed_at")
            cols = st.columns([5, 2, 2])
            entry = float(t["entry_price"])
            if is_open:
                last: Optional[float] = live.get(t["ticker"])
                pnl = ((last - entry) / entry * 100.0) if (last and entry) else None
                pnl_s = f" · **{pnl:+.1f}%** (last {last:,.2f})" if pnl is not None else ""
                cols[0].markdown(
                    f"🟢 **{t['ticker']}** — {t['shares']} @ {entry:,.2f}{pnl_s}"
                )
                default_exit = float(last) if last else entry
                exit_px = cols[1].number_input(
                    "Exit", min_value=0.01, value=round(default_exit, 2),
                    key=f"jr_exit_{t['id']}", label_visibility="collapsed",
                )
                if cols[2].button("Close", key=f"jr_close_{t['id']}"):
                    try:
                        close_trade(t["id"], user_id, float(exit_px))
                        st.rerun()
                    except Exception:
                        st.warning("Could not close trade.")
            else:
                exit_px = float(t["exit_price"] or 0)
                ret = ((exit_px - entry) / entry * 100.0) if entry else 0.0
                icon = "✅" if ret > 0 else "🔻"
                cols[0].markdown(
                    f"{icon} {t['ticker']} — {t['shares']} @ {entry:,.2f} → "
                    f"{exit_px:,.2f} (**{ret:+.1f}%**)"
                )
                if cols[2].button("Delete", key=f"jr_del_{t['id']}"):
                    try:
                        delete_trade(t["id"], user_id)
                        st.rerun()
                    except Exception:
                        st.warning("Could not delete trade.")


def log_trade_button(row, *, shares: int, key: str) -> None:
    """'Log this trade' button used from the trade-plan block. Never raises."""
    try:
        user = (st.session_state.get("username") or "").strip().lower()
        if not user:
            return
        entry = row.get("Last") if hasattr(row, "get") else None
        ticker = str(row.get("Ticker") or "").upper() if hasattr(row, "get") else ""
        if not ticker or entry is None:
            return
        if st.button(f"📓 Log this trade ({ticker})", key=key):
            from db.trades import log_trade

            log_trade(user, ticker, float(entry), int(max(shares, 0)), source="scan")
            st.toast(f"Logged {ticker} to your journal")
            st.rerun()
    except Exception:
        pass
