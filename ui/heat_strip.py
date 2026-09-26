"""Diverging day-move color for watchlist heat pills (green up, red down,
neutral gray near zero; intensity by magnitude).

Run 81: the unused Streamlit strip renderer was removed; the color helper is
kept because its contract is tested (tests/test_whats_new.py).
"""
from __future__ import annotations

from typing import Optional


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
