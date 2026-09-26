"""P1-3 / P1-4 / P1-9 — Discover controls above the scanner results.

- Lens pills (P1-3): views on the SAME ranked list — Top setups, Breakouts,
  Unusual volume, Gaps, Momentum, Early breakout, New since last visit — plus a
  link to the live Day Trader movers. A lens only filters rows; it never
  re-ranks. Thresholds reuse definitions the product already shows users
  (ui.result_explain "Why" text and ui.results_intelligence signal rules).
  Lenses with no matching rows are not offered, so a lens never shows an empty
  table.
- Card view (P1-4): a phone-friendly alternative to the wide table.
- New since last visit (P1-9): on the market view, names that were not in the
  full-market scan you last saw (remembered in this browser) are marked 🆕.

Presentation only: reads the frame it is given; no scoring or ranking change.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

from ui.results_intelligence import (
    GAINER_SIGNAL_MIN,
    GAP_SIGNAL_MIN,
    PREBREAKOUT_SIGNAL_MIN,
)

RVOL_MIN = 1.5            # same as the "Why" text's "N× avg volume" rule
TREND10_MIN = 5.0         # same as the "Why" text's "+N% over 10d" rule
AT_HIGH_MIN = 0.97        # same as the "Why" text's "at 20d high" rule
NEW_MARK = "🆕 new"
LENS_KEY = "hsf_lens"
VIEW_KEY = "hsf_results_view"
NEW_SET_KEY = "hsf_new_since_visit"


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "t"}
    try:
        return bool(v) and v == v
    except Exception:
        return False


def _ticker(row: Dict[str, Any]) -> str:
    return str(row.get("Ticker") or row.get("Symbol") or "").strip().upper()


def _breakout(r):
    pos = _num(r.get("BreakoutPos20D"))
    return _truthy(r.get("IsBreakout")) or (pos is not None and pos >= AT_HIGH_MIN)


def _volume(r):
    v = _num(r.get("VolRel20", r.get("RelVol")))
    return v is not None and v >= RVOL_MIN


def _gap(r):
    g = _num(r.get("GapPct"))
    return g is not None and abs(g) >= GAP_SIGNAL_MIN


def _momentum(r):
    c, t = _num(r.get("PctChange")), _num(r.get("Trend10D%"))
    return (c is not None and c >= GAINER_SIGNAL_MIN) or (t is not None and t >= TREND10_MIN)


def _early(r):
    p = _num(r.get("PreBreakoutProb%"))
    return p is not None and p >= PREBREAKOUT_SIGNAL_MIN


# key -> (label, predicate). "all" and "new" are handled specially.
LENSES: Dict[str, tuple] = {
    "all": ("All setups", None),
    "breakouts": ("Breakouts", _breakout),
    "volume": ("Unusual volume", _volume),
    "gaps": ("Gaps", _gap),
    "momentum": ("Momentum", _momentum),
    "early": ("Early breakout", _early),
    "new": ("New since last visit", None),
}


def lens_mask(df: Any, lens: str, new_set: Optional[Set[str]] = None) -> List[bool]:
    rows = df.to_dict(orient="records")
    if lens == "new":
        s = new_set or set()
        return [_ticker(r) in s for r in rows]
    pred = LENSES.get(lens, (None, None))[1]
    if pred is None:
        return [True] * len(rows)
    return [bool(pred(r)) for r in rows]


def apply_lens(df: Any, lens: str, new_set: Optional[Set[str]] = None) -> Any:
    """Rows matching the lens, in the original order (never re-ranked)."""
    if df is None or getattr(df, "empty", True) or lens in (None, "all"):
        return df
    return df[lens_mask(df, lens, new_set)]


def lens_counts(df: Any, new_set: Optional[Set[str]] = None) -> Dict[str, int]:
    if df is None or getattr(df, "empty", True):
        return {}
    return {k: sum(lens_mask(df, k, new_set)) for k in LENSES}


def available_lenses(counts: Dict[str, int]) -> List[str]:
    """'all' always; every other lens only when it has matching rows."""
    return ["all"] + [k for k in LENSES if k != "all" and counts.get(k, 0) > 0]


def mark_new(df: Any, new_set: Set[str]) -> Any:
    """Prefix the Why text of new names with 🆕 (copy; order unchanged)."""
    if not new_set or df is None or getattr(df, "empty", True) or "Why" not in df.columns:
        return df
    out = df.copy()
    tick = [_ticker(r) for r in out.to_dict(orient="records")]
    out["Why"] = [f"{NEW_MARK} · {w}" if t in new_set and not str(w).startswith(NEW_MARK) else w
                  for t, w in zip(tick, out["Why"])]
    return out


# ---- Streamlit ----------------------------------------------------------------------------------
def render_discover_bar(df: Any) -> Any:
    """Lens pills + view switch above the results; returns the filtered frame.
    Never raises (falls back to the unfiltered frame)."""
    if st is None or df is None or getattr(df, "empty", True):
        return df
    try:
        from ui.market_default import MARKET_VIEW_KEY

        new_set: Set[str] = set()
        if st.session_state.get(MARKET_VIEW_KEY):
            from ui.last_visit import new_since_last_visit

            new_set = new_since_last_visit(df)
        st.session_state[NEW_SET_KEY] = sorted(new_set)
        df = mark_new(df, new_set)
        counts = lens_counts(df, new_set)
        options = available_lenses(counts)
        if st.session_state.get(LENS_KEY) not in options:
            st.session_state[LENS_KEY] = "all"
        c1, c2 = st.columns([4, 1])
        with c1:
            lens = st.pills(
                "Show", options, key=LENS_KEY, selection_mode="single",
                format_func=lambda k: LENSES[k][0] if k == "all" else f"{LENSES[k][0]} ({counts.get(k, 0)})",
                label_visibility="collapsed",
            ) or "all"
        with c2:
            st.segmented_control("View", ["Table", "Cards"], key=VIEW_KEY, default="Table",
                                 label_visibility="collapsed")
        st.page_link("pages/day_trader.py", label="Live movers (Day Trader)", icon="⚡")
        return apply_lens(df, lens, new_set)
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("discover controls", exc)
        return df


def with_card_view(render_results: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap the table renderer: Cards view shows result cards instead."""
    def _render(df, *args, **kwargs):
        if st is not None and st.session_state.get(VIEW_KEY) == "Cards" and df is not None and not df.empty:
            from ui.result_cards import render_result_cards

            return render_result_cards(df)
        return render_results(df, *args, **kwargs)
    return _render
