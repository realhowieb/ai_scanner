"""Optional product-showcase presentation mode.

This module is deliberately pure at its boundary: it reads one environment
flag and only changes rendering. It never receives or mutates scoring inputs,
rankings, persistence objects, or model artifacts.
"""
from __future__ import annotations

import os
from typing import Any, Iterable

import pandas as pd

_TRUE = {"1", "true", "yes", "on"}


def screenshot_mode(env: dict[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return str(source.get("HSF_SCREENSHOT_MODE", "")).strip().lower() in _TRUE


def initial_sidebar_state(env: dict[str, str] | None = None) -> str:
    return "collapsed" if screenshot_mode(env) else "expanded"


def honest_display(value: Any, *, kind: str = "text") -> str:
    """Compact display formatting; unknown stays unknown rather than becoming 0."""
    try:
        missing = value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = value is None
    if missing:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if kind == "price":
        return f"${number:,.2f}"
    if kind == "percent":
        return f"{number:+.2f}%"
    if kind == "probability":
        pct = number * 100 if 0 <= number <= 1 else number
        return f"{pct:.0f}%"
    if kind == "rvol":
        return f"{number:.1f}x"
    if kind == "score":
        return f"{number:.0f}"
    return f"{number:g}"


def select_columns(frame: pd.DataFrame, preferred: Iterable[str]) -> pd.DataFrame:
    """Return a copy with existing preferred columns, preserving row order exactly."""
    columns = [name for name in preferred if name in frame.columns]
    return frame.loc[:, columns].copy()


DAY_TRADER_SHOWCASE_COLUMNS = (
    "Ticker", "Last", "Change $", "Gap %", "Direction", "DT Score", "Setup",
    "ADX", "vs VWAP", "RVOL", "SuperTrend (13,2)", "EWO", "Volume (M)",
    "Volume (IEX M)",
)


def apply_showcase_styles() -> None:
    if not screenshot_mode():
        return
    try:
        import streamlit as st
    except Exception:  # pragma: no cover
        return
    st.markdown(
        """
        <style>
        .block-container {max-width: 1540px; padding-top: 1.35rem; padding-bottom: 2rem;}
        [data-testid="stSidebar"] {min-width: 17rem;}
        [data-testid="stHeader"] {background: transparent;}
        [data-testid="stToolbar"], [data-testid="stStatusWidget"], #MainMenu, footer {display:none !important;}
        h1, h2, h3 {letter-spacing: 0 !important;}
        h2 {margin-bottom: .15rem;}
        [data-testid="stCaptionContainer"] {margin-top: -.15rem;}
        [data-testid="stMetric"] {padding: .55rem .7rem; border: 1px solid rgba(128,128,128,.22); border-radius: 6px;}
        [data-testid="stDataFrame"] {border: 1px solid rgba(128,128,128,.22); border-radius: 6px; overflow: hidden;}
        hr {margin: 1rem 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def showcase_caption() -> None:
    if not screenshot_mode():
        return
    try:
        import streamlit as st

        st.caption("Product showcase · current application data")
    except Exception:
        pass
