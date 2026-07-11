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

_TOP_SCORES_PER_SNAPSHOT = 50


def _score_history_uncached(max_days: int = 30) -> List[Dict[str, Any]]:
    """[{'day': date, 'scores': [(ticker, score) desc]}] — latest snapshot per
    UTC day (manual runs can save several snapshots a day; extras would skew
    'fired in N of last M scans')."""
    try:
        from db.runs import list_runs, load_run_results
        from ui.app_runtime import normalize_results_to_df
    except Exception:
        return []

    try:
        runs = list_runs(limit=5000) or []
    except Exception:
        return []

    by_day: Dict[Any, Dict[str, Any]] = {}
    for r in runs:  # newest-first
        if not r.get("is_snapshot"):
            continue
        created = r.get("created_at")
        if created is None or not hasattr(created, "date"):
            continue
        day = created.date()
        if day in by_day:
            continue  # keep only the newest snapshot per day
        try:
            raw = load_run_results(r["id"])
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
        if len(by_day) >= max_days:
            break
    return sorted(by_day.values(), key=lambda h: h["day"], reverse=True)


if st is not None:

    @st.cache_data(ttl=900, show_spinner=False)
    def load_score_history(max_days: int = 30) -> List[Dict[str, Any]]:
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
