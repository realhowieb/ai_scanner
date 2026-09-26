"""P1-4 — phone-friendly result cards (alternative to the wide results table).

One bordered card per setup: ticker, HSF Score, primary setup, the "Why"
evidence, price / day change / relative volume, and an Open button that goes
straight to Stock Intelligence for that ticker. Built from native Streamlit
containers, so cards stack cleanly at phone width with no horizontal scroll.
Presentation only (row order and values as given).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

CARD_LIMIT = 25


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def card_model(row: Dict[str, Any]) -> Dict[str, Any]:
    """Plain data for one card (testable without Streamlit)."""
    from ui.headline_score import HSF_SCORE_COL
    from ui.opportunities import _primary_setup
    from ui.results_intelligence import _row_to_signal_fields

    f = _row_to_signal_fields(row)
    order = ["golden_cross", "breakout", "prebreakout", "gapper", "gainer"]
    setup = _primary_setup([s for s in order if s in f["signals"]], f["fading"])
    score = _num(row.get(HSF_SCORE_COL))
    why = str(row.get("Why") or "")
    facts: List[str] = []
    last, chg, rvol = _num(row.get("Last")), _num(row.get("PctChange")), _num(row.get("VolRel20"))
    if last is not None:
        facts.append(f"${last:,.2f}")
    if chg is not None:
        facts.append(f"{chg:+.2f}% today")
    if rvol is not None:
        facts.append(f"RVOL {rvol:.2f}×")
    return {
        "ticker": str(row.get("Ticker") or row.get("Symbol") or "").strip().upper(),
        "score": None if score is None else int(round(score)),
        "setup": setup if setup != "Signal" else None,
        "why": [w.strip() for w in why.split("·") if w.strip()],
        "facts": facts,
    }


def card_markdown(m: Dict[str, Any]) -> str:
    head = f"**{m['ticker']}**"
    if m["score"] is not None:
        head += f" · HSF Score **{m['score']}**"
    if m["setup"]:
        head += f" · {m['setup']}"
    return head


def render_result_cards(df: Any, *, limit: int = CARD_LIMIT) -> None:
    """Render up to `limit` cards. Never raises."""
    if st is None or df is None or getattr(df, "empty", True):
        return
    try:
        rows = df.to_dict(orient="records")
        for i, row in enumerate(rows[:limit]):
            m = card_model(row)
            if not m["ticker"]:
                continue
            with st.container(border=True):
                st.markdown(card_markdown(m))
                if m["why"]:
                    st.markdown(" · ".join(m["why"]))
                if m["facts"]:
                    st.caption(" · ".join(m["facts"]))
                if st.button("Open in Stock Intelligence", key=f"hsf_card_open_{i}_{m['ticker']}"):
                    st.session_state["hsf_stock_ticker"] = m["ticker"]
                    st.session_state.pop("hsf_stock_opp", None)
                    st.switch_page("pages/stock.py")
        if len(rows) > limit:
            st.caption(f"Showing the first {limit} of {len(rows)}. Switch to Table to see them all.")
    except Exception as exc:
        from ui.safe_errors import show_error

        show_error("the result cards", exc)
