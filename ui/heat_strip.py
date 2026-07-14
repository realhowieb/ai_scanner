"""Watchlist heat strip: one row of colored pills with each name's day move.

A one-second market read above the results — diverging color (green up, red
down, neutral gray at ~0) with intensity by magnitude; the number is printed in
text ink on every pill so color is never the only encoding.
"""
from __future__ import annotations

from typing import List, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover - headless envs
    st = None  # type: ignore[assignment]


def pill_color(chg_pct: Optional[float]) -> str:
    """Diverging background for a day move; neutral near zero."""
    if chg_pct is None:
        return "rgba(128,128,128,0.15)"
    if abs(chg_pct) < 0.15:
        return "rgba(128,128,128,0.18)"
    alpha = min(abs(chg_pct) / 4.0, 1.0) * 0.45 + 0.10
    return (
        f"rgba(22,163,74,{alpha:.2f})" if chg_pct > 0 else f"rgba(220,38,38,{alpha:.2f})"
    )


def render_watchlist_heat(watch_tickers: List[str] | None, limit: int = 14) -> None:
    """Render the strip. Hidden when no watchlist / no quotes. Never raises."""
    if st is None or not watch_tickers:
        return
    try:
        from market_data import build_day_trader_metrics

        rows = build_day_trader_metrics(
            [str(t).upper() for t in watch_tickers][:limit], with_rvol=False
        )
    except Exception:
        return
    if not rows:
        return
    pills = []
    for r in rows:
        chg = r.get("chg_pct")
        chg_s = f"{chg:+.1f}%" if chg is not None else "—"
        pills.append(
            f"<span style='display:inline-block;padding:2px 8px;margin:2px;"
            f"border-radius:10px;font-size:12px;background:{pill_color(chg)}'>"
            f"<b>{r.get('ticker')}</b> {chg_s}</span>"
        )
    st.markdown(
        "<div style='line-height:1.9'>📋 " + "".join(pills) + "</div>",
        unsafe_allow_html=True,
    )
