"""Track-record dashboard for saved scanner signals."""
from __future__ import annotations

from typing import Any

import streamlit as st

RANKING_LABELS = {
    "breakout": "BreakoutScore",
    "prebreakout": "PreBreakoutProb",
}
DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
MIN_SAMPLE_SIZE = 25


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):+.1%}"
    except (TypeError, ValueError):
        return "-"


def _fmt_win(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "-"


def _load_summary_rows(horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> list[dict[str, Any]]:
    try:
        from db.track_record import load_latest_track_record
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        for ranking in RANKING_LABELS:
            try:
                row = load_latest_track_record(horizon, ranking=ranking)
            except Exception:
                row = None
            if not row:
                continue
            row = dict(row)
            row["ranking"] = row.get("ranking") or ranking
            row["ranking_label"] = RANKING_LABELS.get(str(row["ranking"]), str(row["ranking"]))
            rows.append(row)
    return rows


def _best_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [r for r in rows if r.get("avg_return") is not None]
    if not eligible:
        return None
    return max(eligible, key=lambda r: float(r.get("avg_return") or 0.0))


def _render_summary_metrics(rows: list[dict[str, Any]]) -> None:
    best = _best_summary(rows)
    if not best:
        st.info("No track-record summary is available yet. Run scheduled scans long enough to build history.")
        return

    sample = int(best.get("sample_size") or 0)
    runs = int(best.get("runs_used") or 0)
    bench = best.get("benchmark") or "SPY"
    top_n = best.get("top_n") or 5
    horizon = best.get("horizon_days")
    label = best.get("ranking_label") or best.get("ranking") or "Signal"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Best signal vs {bench}", _fmt_pct(best.get("avg_return")), help=f"{label}, {horizon}D, top-{top_n}")
    c2.metric("Beat rate", _fmt_win(best.get("win_rate")))
    c3.metric("Samples", f"{sample:,}")
    c4.metric("Runs used", f"{runs:,}")

    if sample < MIN_SAMPLE_SIZE:
        st.caption("Small sample so far. Treat direction as more meaningful than exact percentages.")


def _render_summary_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "Signal": row.get("ranking_label") or row.get("ranking"),
                "Horizon": f"{row.get('horizon_days')}D",
                "Avg vs SPY": _fmt_pct(row.get("avg_return")),
                "Median vs SPY": _fmt_pct(row.get("median_return")),
                "Beat Rate": _fmt_win(row.get("win_rate")),
                "Samples": int(row.get("sample_size") or 0),
                "Runs": int(row.get("runs_used") or 0),
                "Computed": str(row.get("computed_at") or "")[:19],
            }
        )
    try:
        import pandas as pd

        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)
    except Exception:
        st.write(table_rows)


def _render_ab_chart(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        import plotly.graph_objects as go
    except Exception:
        return

    horizons = sorted({int(r.get("horizon_days")) for r in rows if r.get("horizon_days") is not None})
    if not horizons:
        return

    by_key = {
        (str(row.get("ranking")), int(row.get("horizon_days"))): row
        for row in rows
        if row.get("horizon_days") is not None
    }
    fig = go.Figure()
    for ranking, label in RANKING_LABELS.items():
        vals = []
        for horizon in horizons:
            row = by_key.get((ranking, horizon))
            vals.append(None if row is None or row.get("avg_return") is None else float(row["avg_return"]) * 100.0)
        fig.add_trace(
            go.Bar(
                x=[f"{h}D" for h in horizons],
                y=vals,
                name=label,
                hovertemplate=label + " %{x}: %{y:+.2f}% vs SPY<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color="rgba(128,128,128,0.45)", line_width=1)
    fig.update_layout(
        barmode="group",
        height=280,
        margin=dict(l=0, r=0, t=12, b=0),
        legend=dict(orientation="h", y=1.12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(ticksuffix="%", gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", key="track_record_ab_chart")


def _render_daily_heatmap(ranking: str, horizon: int) -> None:
    try:
        from db.track_record import load_daily_excess

        daily = [d for d in load_daily_excess(ranking, horizon, days=120) if d[1] is not None]
    except Exception:
        daily = []
    if len(daily) < 5:
        return
    try:
        import plotly.graph_objects as go
    except Exception:
        return

    weeks, weekdays, vals, texts = [], [], [], []
    for day, excess in daily:
        weeks.append(day.isocalendar()[1] + day.year * 53)
        weekdays.append(day.weekday())
        vals.append(float(excess) * 100.0)
        texts.append(f"{day:%b %d}: {float(excess) * 100.0:+.2f}% vs SPY")
    wmin = min(weeks)
    weeks = [w - wmin for w in weeks]
    bound = max(abs(v) for v in vals) or 1.0
    fig = go.Figure(
        go.Heatmap(
            x=weeks,
            y=weekdays,
            z=vals,
            text=texts,
            hovertemplate="%{text}<extra></extra>",
            colorscale=[[0, "#dc2626"], [0.5, "#334155"], [1, "#16a34a"]],
            zmid=0,
            zmin=-bound,
            zmax=bound,
            xgap=3,
            ygap=3,
            colorbar=dict(title="% vs SPY", thickness=10),
        )
    )
    fig.update_layout(
        height=210,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(
            tickvals=[0, 1, 2, 3, 4],
            ticktext=["Mon", "Tue", "Wed", "Thu", "Fri"],
            autorange="reversed",
        ),
    )
    st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", key=f"track_record_heatmap_{ranking}_{horizon}")


def render_track_record_dashboard() -> None:
    """Render a user-facing backtested signal performance dashboard."""
    st.markdown("## Track Record")
    st.caption("Forward returns of saved scan picks compared with SPY. Past performance is not indicative of future results.")

    rows = _load_summary_rows()
    _render_summary_metrics(rows)
    if not rows:
        return

    st.markdown("### Signal Comparison")
    _render_ab_chart(rows)
    _render_summary_table(rows)

    st.markdown("### Daily Result Heatmap")
    c1, c2 = st.columns([1, 1])
    with c1:
        ranking = st.selectbox(
            "Signal",
            list(RANKING_LABELS.keys()),
            format_func=lambda key: RANKING_LABELS.get(key, key),
            key="track_record_heatmap_ranking",
        )
    with c2:
        horizon = st.selectbox("Horizon", [1, 3, 5, 10, 20], index=2, key="track_record_heatmap_horizon")
    _render_daily_heatmap(str(ranking), int(horizon))
