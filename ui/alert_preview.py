"""Smart-create insights for alerts: score distribution + would-have-fired preview.

Drives the breakout-threshold input with observed data instead of guesses, so
users can't create dead alerts (threshold above anything the market produces)
or spam alerts (threshold every scan clears). One cached pass over recent
stored snapshots; pure functions below it are unit-testable without Streamlit.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import streamlit as st
except Exception:  # pragma: no cover - headless envs; pure functions still work
    st = None  # type: ignore[assignment]

from config import SCORE_EPOCH  # single source of truth

_TOP_SCORES_PER_SNAPSHOT = 50


def _score_history_uncached(max_days: int = 12) -> List[Dict[str, Any]]:
    """[{'day': date, 'scores': [(ticker, score) desc]}] — latest snapshot per
    UTC day (manual runs can save several snapshots a day; extras would skew
    'fired in N of last M scans')."""
    try:
        from db.runs import list_snapshot_runs, load_many_run_results
        from ui.app_runtime import normalize_results_to_df
    except Exception:
        return []

    try:
        runs = list_snapshot_runs(days=max_days * 3, limit=max_days * 5) or []
    except Exception:
        return []

    # Pick the newest snapshot per day first, then batch-fetch their JSON in
    # one query instead of a round trip per snapshot.
    chosen: Dict[Any, Dict[str, Any]] = {}
    for r in runs:  # newest-first
        created = r.get("created_at")
        if created is None or not hasattr(created, "date"):
            continue
        day = created.date()
        if day < SCORE_EPOCH:
            break  # runs are newest-first; everything older is pre-clipping
        if day in chosen:
            continue
        chosen[day] = r
        if len(chosen) >= max_days:
            break
    payloads = load_many_run_results([r["id"] for r in chosen.values()])

    by_day: Dict[Any, Dict[str, Any]] = {}
    for day, r in chosen.items():
        try:
            raw = payloads.get(int(r["id"]))
            df = normalize_results_to_df(raw) if raw else None
        except Exception:
            continue
        if df is None or len(df) == 0 or "BreakoutScore" not in df.columns:
            continue
        tick_col = "Ticker" if "Ticker" in df.columns else ("Symbol" if "Symbol" in df.columns else None)
        if not tick_col:
            continue
        top = df[[tick_col, "BreakoutScore"]].dropna().sort_values("BreakoutScore", ascending=False)
        scores: List[Tuple[str, float]] = [
            (str(t).upper(), float(s))
            for t, s in top.head(_TOP_SCORES_PER_SNAPSHOT).itertuples(index=False)
        ]
        if scores:
            by_day[day] = {"day": day, "scores": scores}
    return sorted(by_day.values(), key=lambda h: h["day"], reverse=True)


if st is not None:

    @st.cache_data(ttl=900, show_spinner=False)
    def load_score_history(max_days: int = 12) -> List[Dict[str, Any]]:
        return _score_history_uncached(max_days)

else:  # pragma: no cover - headless fallback without caching
    load_score_history = _score_history_uncached


def summarize_scores(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """{'snapshots': n, 'median_top': x, 'max_top': y} across daily snapshots."""
    tops = sorted(h["scores"][0][1] for h in history if h.get("scores"))
    if not tops:
        return None
    return {
        "snapshots": len(tops),
        "median_top": tops[len(tops) // 2],
        "max_top": tops[-1],
    }


def preview_breakout_threshold(
    history: List[Dict[str, Any]], threshold: float
) -> Optional[Dict[str, Any]]:
    """How often `threshold` would have fired across the daily snapshots.

    {'fired': k, 'total': n, 'example': ('CRNX', 61.9, date) | None}
    """
    if not history:
        return None
    fired = 0
    example = None
    for h in history:  # newest-first
        hits = [(t, s) for t, s in h.get("scores", []) if s >= float(threshold)]
        if hits:
            fired += 1
            if example is None:
                example = (hits[0][0], hits[0][1], h["day"])
    return {"fired": fired, "total": len(history), "example": example}


def render_breakout_threshold_insight(threshold: float) -> None:
    """Live caption under the breakout-threshold input. Never raises."""
    try:
        history = load_score_history()
        stats = summarize_scores(history)
        preview = preview_breakout_threshold(history, threshold)
    except Exception:
        return
    if not stats or not preview:
        return

    st.caption(
        f"Last {stats['snapshots']} daily scans: top score median "
        f"**{stats['median_top']:.0f}**, max **{stats['max_top']:.0f}**."
    )
    _render_threshold_plot(history, threshold)
    if preview["fired"] == 0:
        st.warning(
            f"⚠️ A threshold of {threshold:g} would not have fired once in the "
            f"last {preview['total']} daily scans — consider a lower value.",
            icon="⚠️",
        )
        return
    ex = preview["example"]
    ex_s = f" — most recently {ex[0]} ({ex[1]:.1f}) on {ex[2]:%b %d}" if ex else ""
    if preview["fired"] == preview["total"]:
        st.caption(
            f"🔔 Would have fired in **every** one of the last {preview['total']} "
            f"daily scans{ex_s}. Expect frequent alerts."
        )
    else:
        st.caption(
            f"🔔 Would have fired in **{preview['fired']} of the last "
            f"{preview['total']}** daily scans{ex_s}."
        )


def _render_threshold_plot(history: List[Dict[str, Any]], threshold: float) -> None:
    """Daily top scores as dots with the user's threshold as a reference line.

    Shows exactly where the chosen threshold sits in the observed distribution
    — above every dot = dead alert, below every dot = fires daily.
    """
    try:
        import plotly.graph_objects as go

        days = [h["day"] for h in reversed(history) if h.get("scores")]
        tops = [h["scores"][0][1] for h in reversed(history) if h.get("scores")]
        if len(tops) < 2:
            return
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=days, y=tops, mode="markers", name="Daily top score",
                marker=dict(size=9, color="#60a5fa"),
                hovertemplate="%{x|%b %d}: top score %{y:.0f}<extra></extra>",
            )
        )
        fig.add_hline(
            y=float(threshold), line_color="#f59e0b", line_dash="dash",
            annotation_text=f"your threshold ({threshold:g})",
            annotation_font_color="#f59e0b",
        )
        fig.update_layout(
            height=160, margin=dict(l=0, r=0, t=8, b=0), showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        )
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch")
    except Exception:
        pass
