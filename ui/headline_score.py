"""P0-7 — one headline score in scanner results: HSF Score.

HSF Score (0–100, ui.opportunities) is the product's opportunity ranking. The
results grid used to lead with BreakoutScore (labelled just "Score") and show
PreBreakout next to it, so users met several numbers with no clear headline.

This module is presentation only:
- It adds an "HSF Score" column computed by the CANONICAL path the intelligence
  panel and Stock Intelligence already use (results_intelligence.
  consolidate_scanner_results → ui.opportunities.score_breakdown), so the same
  ticker shows the same score everywhere. No new formula, no new signal.
- Model outputs (BreakoutScore, PreBreakout %, AI Confidence) stay available
  as "model details", hidden from the grid unless the user asks for them.
- Row order is untouched: the table keeps the scanner's ranking.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

HSF_SCORE_COL = "HSF Score"
# Model outputs shown only under "model details" (all are inputs or side
# estimates, none is the headline).
MODEL_DETAIL_COLS = ("BreakoutScore", "PreBreakoutProb%", "AI Confidence")
HSF_SCORE_HELP = ("HSF Score (0–100) ranks how strongly a setup's current evidence lines up: "
                  "confirming signals, model strength and momentum. It is a ranking, not a "
                  "probability of profit.")
DETAILS_TOGGLE_LABEL = "Show model details (Breakout score, PreBreakout)"


def hsf_scores_by_ticker(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Ticker → HSF Score via the canonical consolidation (non-qualifying
    names are absent, exactly as in the intelligence panel)."""
    from ui.results_intelligence import consolidate_scanner_results

    return {o["ticker"]: o["score"] for o in consolidate_scanner_results(list(rows or []), top_n=None)}


def _ticker_col(df: Any) -> Optional[str]:
    cols = list(getattr(df, "columns", []))
    return "Ticker" if "Ticker" in cols else ("Symbol" if "Symbol" in cols else None)


def add_hsf_score_column(df: Any) -> Any:
    """Return a copy of df with an "HSF Score" column after Ticker/Why.

    No-op for None/empty frames, frames without a ticker column, or frames
    that already carry the column. Never reorders rows.
    """
    if df is None or getattr(df, "empty", True) or HSF_SCORE_COL in df.columns:
        return df
    tcol = _ticker_col(df)
    if tcol is None:
        return df
    try:
        scores = hsf_scores_by_ticker(df.to_dict(orient="records"))
    except Exception:
        return df
    out = df.copy()
    values = [scores.get(str(t or "").strip().upper()) for t in out[tcol]]
    after = "Why" if "Why" in out.columns else tcol
    out.insert(list(out.columns).index(after) + 1, HSF_SCORE_COL, values)
    return out


def visible_columns(columns: Iterable[str], *, show_details: bool) -> List[str]:
    """Grid column order: everything except model details unless requested."""
    cols = list(columns)
    if show_details:
        return cols
    return [c for c in cols if c not in MODEL_DETAIL_COLS]


def hsf_score_of(row: Any) -> Optional[int]:
    """HSF Score from a results row (Series/dict) that went through
    add_hsf_score_column; None when absent or not a number."""
    try:
        v = row.get(HSF_SCORE_COL)
        if v is None or v != v:  # None / NaN
            return None
        return int(round(float(v)))
    except (AttributeError, TypeError, ValueError):
        return None


def hsf_metric_text(row: Any) -> str:
    """Detail-card value for the HSF Score metric."""
    s = hsf_score_of(row)
    return "—" if s is None else str(s)


def breakout_score_help(bs: Optional[float]) -> str:
    """Detail-card help: the model input behind the headline, plus what it means."""
    base = HSF_SCORE_HELP
    return base if bs is None else f"{base} Breakout score (model input): {bs:.2f}."


def model_details_view(df: Any, *, key: str) -> Any:
    """Render the model-details toggle (only when such columns exist) and
    return df limited to the visible columns. Never raises."""
    try:
        if df is None or not any(c in df.columns for c in MODEL_DETAIL_COLS):
            return df
        import streamlit as st

        show = st.toggle(DETAILS_TOGGLE_LABEL, value=False, key=f"{key}_model_details")
        return df[visible_columns(df.columns, show_details=bool(show))]
    except Exception:
        return df
