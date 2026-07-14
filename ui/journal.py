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
        _render_journal_charts([t for t in trades if t.get("closed_at")])

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


def _render_journal_charts(closed: list) -> None:
    """Equity curve + per-trade P&L bars over closed trades. Never raises."""
    if len(closed) < 2:
        return
    try:
        import plotly.graph_objects as go

        # Chronological (list_trades returns closed newest-last already mixed;
        # sort by closed_at to be safe).
        closed = sorted(closed, key=lambda t: t.get("closed_at") or 0)
        rets = []
        labels = []
        for t in closed:
            entry = float(t.get("entry_price") or 0)
            exit_px = float(t.get("exit_price") or 0)
            if entry <= 0:
                continue
            rets.append((exit_px - entry) / entry * 100.0)
            labels.append(str(t.get("ticker")))
        if len(rets) < 2:
            return
        cum = []
        total = 0.0
        for r in rets:
            total += r
            cum.append(total)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(
                go.Scatter(
                    x=list(range(1, len(cum) + 1)), y=cum, mode="lines+markers",
                    line=dict(color="#60a5fa", width=2), marker=dict(size=7),
                    hovertemplate="after trade %{x}: %{y:+.1f}%<extra></extra>",
                )
            )
            fig.add_hline(y=0, line_color="rgba(128,128,128,0.4)", line_width=1)
            fig.update_layout(
                title=dict(text="Cumulative return (sum of trade %)", font=dict(size=13)),
                height=180, margin=dict(l=0, r=0, t=28, b=0), showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, title=None),
                yaxis=dict(gridcolor="rgba(128,128,128,0.15)", ticksuffix="%"),
            )
            st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")
        with c2:
            colors = ["#16a34a" if r > 0 else "#dc2626" for r in rets]
            fig = go.Figure(
                go.Bar(
                    x=labels, y=rets, marker_color=colors,
                    hovertemplate="%{x}: %{y:+.1f}%<extra></extra>",
                )
            )
            fig.add_hline(y=0, line_color="rgba(128,128,128,0.4)", line_width=1)
            fig.update_layout(
                title=dict(text="Per-trade return", font=dict(size=13)),
                height=180, margin=dict(l=0, r=0, t=28, b=0), showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(128,128,128,0.15)", ticksuffix="%"),
            )
            st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")
        if len(rets) >= 8:
            _render_return_histogram(rets)
    except Exception:
        pass


def _render_return_histogram(rets: list) -> None:
    """Distribution of closed-trade returns with zero and average marked."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Histogram(
                x=rets, nbinsx=12, marker_color="#60a5fa",
                hovertemplate="%{x}: %{y} trades<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_color="rgba(128,128,128,0.5)", line_dash="dash")
        avg = sum(rets) / len(rets)
        fig.add_vline(
            x=avg, line_color="#f59e0b", line_dash="dot",
            annotation_text=f"avg {avg:+.1f}%", annotation_font_color="#f59e0b",
        )
        fig.update_layout(
            title=dict(text="Return distribution (closed trades)", font=dict(size=13)),
            height=180, margin=dict(l=0, r=0, t=28, b=0), showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(ticksuffix="%", showgrid=False),
            yaxis=dict(gridcolor="rgba(128,128,128,0.12)"),
        )
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")
    except Exception:
        pass
