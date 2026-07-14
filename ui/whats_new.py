"""'What changed since yesterday' strip: new entrants + biggest score movers.

Compares the two most recent daily snapshots (via the cached score history the
smart-create insight already loads) and renders a one-line summary. Pure diff
logic is streamlit-free and unit-testable; rendering never raises.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover - headless envs
    st = None  # type: ignore[assignment]

NEW_ENTRANT_TOP_N = 20
MOVERS_SHOWN = 3


def diff_snapshots(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Diff the two newest daily snapshots.

    Returns {'new': [tickers], 'movers': [(ticker, delta)], 'day': date} or None
    when fewer than two days of history exist.
    """
    if len(history) < 2:
        return None
    today, prior = history[0], history[1]
    t_scores = dict(today.get("scores") or [])
    p_scores = dict(prior.get("scores") or [])
    if not t_scores or not p_scores:
        return None

    top_today = [t for t, _ in (today.get("scores") or [])[:NEW_ENTRANT_TOP_N]]
    new = [t for t in top_today if t not in p_scores]

    common = [t for t in t_scores if t in p_scores]
    movers = sorted(
        ((t, t_scores[t] - p_scores[t], t_scores[t]) for t in common),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:MOVERS_SHOWN]
    movers = [(t, d, s) for t, d, s in movers if abs(d) >= 1.0]

    return {"new": new[:5], "movers": movers, "day": today.get("day")}


def render_whats_new_strip() -> None:
    """One-line 'since yesterday' caption on the main page. Never raises."""
    if st is None:
        return
    try:
        from ui.alert_preview import load_score_history

        diff = diff_snapshots(load_score_history(max_days=2))
    except Exception:
        return
    if not diff or (not diff["new"] and not diff["movers"]):
        return

    # Stat-tile row: each mover is a headline number with polarity, which is
    # st.metric's exact job — big ticker, today's score, signed delta with the
    # built-in green/red arrow (theme-correct in dark mode, no custom colors).
    # Header line carries the new entrants as text (a ticker list in a metric
    # tile rendered as an oversized headline); movers stay as compact tiles.
    header = "**Since yesterday**"
    if diff["new"]:
        header += " · 🆕 new in results: **" + ", ".join(diff["new"]) + "**"
    st.markdown(header)
    movers = diff["movers"][:4]
    if movers:
        # Fixed grid: with fewer movers, empty columns keep tiles compact on
        # the left instead of spreading two tiles across the full width.
        cols = st.columns(4)
        for col, (ticker, delta, score) in zip(cols, movers):
            col.metric(
                ticker,
                f"{score:.0f}",
                delta=f"{delta:+.0f} score",
                help=f"BreakoutScore {score:.0f} today ({delta:+.0f} vs yesterday)",
            )
