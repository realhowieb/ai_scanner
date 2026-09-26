"""P1-3 / P1-4 / P1-9 — Discover controls above the scanner results.

- Lens pills (P1-3, multi-select since P2-4 — combined with AND): views on the
  SAME ranked list — Breakouts,
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
PENDING_LENS_KEY = "hsf_lens_pending"
VIEW_KEY = "hsf_results_view"
NEW_SET_KEY = "hsf_new_since_visit"
VIEW_PREF = "hsf_results_view_pref"
LENS_PREF = "hsf_lens_pref"
PREFS_LOADED_KEY = "hsf_discover_prefs_loaded"


def mobile_request(headers: Any = None) -> bool:
    """Conservative phone detection used only for an unset view preference."""
    try:
        ua = str((headers or {}).get("User-Agent", "")).lower()
    except Exception:
        return False
    return any(token in ua for token in ("iphone", "android", "mobile"))


def load_discover_preferences(session_state: Any, headers: Any = None) -> None:
    """Load browser conveniences once; no database or account data is involved."""
    if session_state.get(PREFS_LOADED_KEY):
        return
    from ui.browser_prefs import get, get_json

    saved_view = get(VIEW_PREF)
    session_state.setdefault(VIEW_KEY, saved_view if saved_view in {"Table", "Cards"}
                             else ("Cards" if mobile_request(headers) else "Table"))
    saved_lenses = get_json(LENS_PREF, [])
    if isinstance(saved_lenses, list):
        session_state.setdefault(LENS_KEY, [k for k in saved_lenses if k in LENSES and k != "all"])
    session_state[PREFS_LOADED_KEY] = True


def _persist_view() -> None:
    from ui.browser_prefs import put

    value = st.session_state.get(VIEW_KEY)
    if value in {"Table", "Cards"}:
        put(VIEW_PREF, value)


def _persist_lenses() -> None:
    from ui.browser_prefs import put_json

    value = st.session_state.get(LENS_KEY, [])
    put_json(LENS_PREF, [k for k in value if k in LENSES and k != "all"])


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


def apply_lenses(df: Any, lenses: List[str], new_set: Optional[Set[str]] = None) -> Any:
    """Rows matching ALL selected lenses (P2-4 screens), original order kept."""
    keys = [k for k in (lenses or []) if k in LENSES and k != "all"]
    if df is None or getattr(df, "empty", True) or not keys:
        return df
    mask = [True] * len(df)
    for k in keys:
        mask = [a and b for a, b in zip(mask, lens_mask(df, k, new_set))]
    return df[mask]


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


# ---- Saved screens (P2-4) ------------------------------------------------------------------------
SCREENS_PREF = "hsf_screens"
MAX_SCREENS = 10


def normalize_screens(raw: Any) -> Dict[str, List[str]]:
    """{name: [lens keys]} with unknown lenses dropped, names trimmed, ≤ MAX_SCREENS."""
    out: Dict[str, List[str]] = {}
    if not isinstance(raw, dict):
        return out
    for name, lenses in list(raw.items())[:MAX_SCREENS]:
        n = str(name or "").strip()[:30]
        keys = [k for k in (lenses or []) if k in LENSES and k != "all"] if isinstance(lenses, list) else []
        if n and keys:
            out[n] = keys
    return out


def save_screen(screens: Dict[str, List[str]], name: str, lenses: List[str]) -> Dict[str, List[str]]:
    """Return screens with `name` saved (replacing a same-named one)."""
    n = str(name or "").strip()[:30]
    keys = [k for k in lenses if k in LENSES and k != "all"]
    if not n or not keys:
        return dict(screens)
    out = {k: v for k, v in screens.items() if k != n}
    out[n] = keys
    return dict(list(out.items())[-MAX_SCREENS:])


def _render_screens(selected: List[str]) -> None:
    """Compact saved-screens row: apply / save / delete. Never raises."""
    from ui.browser_prefs import get_json, put_json

    screens = normalize_screens(get_json(SCREENS_PREF, {}))
    with st.expander("Saved screens", expanded=False):
        if screens:
            pick = st.selectbox("Apply a saved screen", ["—"] + list(screens), key="hsf_screen_pick")
            c1, c2 = st.columns(2)
            if c1.button("Apply", key="hsf_screen_apply", disabled=pick == "—"):
                # The pills already rendered this run; apply on the next run.
                st.session_state[PENDING_LENS_KEY] = list(screens[pick])
                st.rerun()
            if c2.button("Delete", key="hsf_screen_delete", disabled=pick == "—"):
                put_json(SCREENS_PREF, {k: v for k, v in screens.items() if k != pick})
                st.rerun()
        keys = [k for k in selected if k != "all"]
        name = st.text_input("Save the current lenses as", key="hsf_screen_name",
                             placeholder="e.g. Breakouts on volume", max_chars=30)
        if st.button("Save screen", key="hsf_screen_save", disabled=not (keys and name.strip())):
            if put_json(SCREENS_PREF, save_screen(screens, name, keys)):
                st.toast(f"Saved “{name.strip()}”")
            else:
                st.warning("HSF couldn't save that screen in this browser.")
        if not keys:
            st.caption("Pick one or more lenses above, then save them as a screen.")


# ---- Streamlit ----------------------------------------------------------------------------------
def render_discover_bar(df: Any) -> Any:
    """Lens pills + view switch above the results; returns the filtered frame.
    Never raises (falls back to the unfiltered frame)."""
    if st is None or df is None or getattr(df, "empty", True):
        return df
    try:
        from ui.market_default import MARKET_VIEW_KEY

        try:
            headers = st.context.headers
        except Exception:
            headers = {}
        load_discover_preferences(st.session_state, headers)

        new_set: Set[str] = set()
        if st.session_state.get(MARKET_VIEW_KEY):
            from ui.last_visit import new_since_last_visit

            new_set = new_since_last_visit(df)
        st.session_state[NEW_SET_KEY] = sorted(new_set)
        df = mark_new(df, new_set)
        counts = lens_counts(df, new_set)
        options = [k for k in available_lenses(counts) if k != "all"]
        pending = st.session_state.pop(PENDING_LENS_KEY, None)
        current = pending if pending is not None else st.session_state.get(LENS_KEY)
        if isinstance(current, str):            # older single-lens state (e.g. Today's link)
            current = [] if current == "all" else [current]
        st.session_state[LENS_KEY] = [k for k in (current or []) if k in options]
        if pending is not None:
            _persist_lenses()
        c1, c2 = st.columns([4, 1])
        with c1:
            chosen = st.pills(
                "Lenses", options, key=LENS_KEY, selection_mode="multi",
                format_func=lambda k: f"{LENSES[k][0]} ({counts.get(k, 0)})",
                label_visibility="collapsed", on_change=_persist_lenses,
            ) or []
        with c2:
            st.segmented_control("View", ["Table", "Cards"], key=VIEW_KEY,
                                 label_visibility="collapsed", on_change=_persist_view)
        st.page_link("pages/day_trader.py", label="Live movers (Day Trader)", icon="⚡")
        _render_screens(list(chosen))
        filtered = apply_lenses(df, list(chosen), new_set)
        if chosen and (filtered is None or filtered.empty):
            st.info("No setups match all the selected lenses right now. Showing all setups.")
            return df
        return filtered
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
