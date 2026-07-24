"""Strategy Lab: signal leaderboard UI.

Reads the daily-computed leaderboard (db.signal_leaderboard) and shows which
scanner signal actually predicted forward moves, ranked by benchmark-excess
return. Read-only; the heavy compute runs in the cron. Never raises into the
results view.
"""
from __future__ import annotations

from typing import Any, Dict, List

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

DEFAULT_HORIZON = 5


def _fmt_pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:+.2f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_win(v: Any) -> str:
    try:
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def render_signal_leaderboard() -> None:
    """Signal leaderboard section for the Track Record / Strategy Lab. Never raises."""
    if st is None:
        return
    try:
        st.markdown("### 🧪 Strategy Lab — signal leaderboard")
        st.caption(
            "Which scanner signal actually led to the best next-move? Each signal's "
            "top picks per day, measured as excess return vs SPY over the holding "
            "period. Past performance is not indicative of future results."
        )
        from db.signal_leaderboard import leaderboard_horizons, load_leaderboard

        horizons = leaderboard_horizons() or [1, 5, 20]
        default_idx = horizons.index(DEFAULT_HORIZON) if DEFAULT_HORIZON in horizons else 0
        c1, c2 = st.columns(2)
        horizon = c1.selectbox(
            "Holding period (trading days)", horizons, index=default_idx,
            key="strategy_lab_horizon",
        )
        entry_label = c2.selectbox(
            "Entry", ["Close", "Open"], index=0, key="strategy_lab_entry",
            help=(
                "Close: enter at the signal-day close. Open: enter at the next "
                "open — the realistic fill for an early/premarket signal, which "
                "usually reads very differently for momentum picks."
            ),
        )
        entry_mode = "open" if entry_label == "Open" else "close"
        rows = load_leaderboard(int(horizon), entry_mode=entry_mode)
        if not rows:
            st.info(
                "The leaderboard is still gathering history — it populates once "
                "enough daily snapshots have a complete forward window."
            )
            return
        _render_leaderboard_table(rows)
        _render_leaderboard_chart(rows)
    except Exception:
        pass


def _render_leaderboard_table(rows: List[Dict[str, Any]]) -> None:
    try:
        best = rows[0]
        st.caption(
            f"🥇 Best signal: **{best.get('display') or best.get('signal')}** — "
            f"avg excess {_fmt_pct(best.get('avg_excess'))}, "
            f"beats SPY {_fmt_win(best.get('win_rate'))} of the time "
            f"(n={int(best.get('sample_size') or 0)})."
        )
        table = [
            {
                "Signal": r.get("display") or r.get("signal"),
                "Avg excess": _fmt_pct(r.get("avg_excess")),
                "Median excess": _fmt_pct(r.get("median_excess")),
                "Beats SPY": _fmt_win(r.get("win_rate")),
                "Samples": int(r.get("sample_size") or 0),
            }
            for r in rows
        ]
        st.dataframe(table, hide_index=True, width="stretch")
    except Exception:
        pass


def _render_leaderboard_chart(rows: List[Dict[str, Any]]) -> None:
    try:
        import plotly.graph_objects as go

        labels = [r.get("display") or r.get("signal") for r in rows]
        vals = [float(r.get("avg_excess") or 0.0) * 100.0 for r in rows]
        colors = ["#16a34a" if v >= 0 else "#dc2626" for v in vals]
        fig = go.Figure(
            go.Bar(
                x=vals, y=labels, orientation="h", marker_color=colors,
                text=[f"{v:+.2f}%" for v in vals], textposition="outside",
                hovertemplate="%{y}: %{x:+.2f}% excess<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_color="rgba(128,128,128,0.4)", line_width=1)
        fig.update_layout(
            title=dict(text="Avg excess return vs SPY, by signal", font=dict(size=13)),
            height=max(220, 40 * len(rows) + 60),
            margin=dict(l=0, r=0, t=28, b=0), showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(ticksuffix="%", gridcolor="rgba(128,128,128,0.12)"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")
    except Exception:
        pass
