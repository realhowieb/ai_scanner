"""Watchlist badges and filtering for the existing scanner results table."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st


def _active_watchlist_symbols() -> set[str]:
    return {
        str(t).strip().upper()
        for t in (st.session_state.get("active_watchlist_tickers") or [])
        if str(t).strip()
    }


def _ticker_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("Ticker", "Symbol", "ticker", "symbol"):
        if col in df.columns:
            return col
    return None


def apply_watchlist_result_view(df: pd.DataFrame, *, key_prefix: str) -> pd.DataFrame:
    """Annotate/filter already-computed results against the active watchlist."""
    watch = _active_watchlist_symbols()
    ticker_col = _ticker_column(df)
    if not watch or ticker_col is None:
        return df

    out = df.copy()
    normalized = out[ticker_col].astype(str).str.strip().str.upper()
    watched_mask = normalized.isin(watch)
    out["Watching"] = watched_mask.map({True: "★ Watching", False: ""})

    c1, c2 = st.columns([1, 3])
    watchlist_only = c1.checkbox(
        "Watchlist only",
        value=False,
        key=f"{key_prefix}_watchlist_only",
        help="Show only active watchlist tickers from the current scan results.",
    )
    c2.caption(f"{int(watched_mask.sum())} active watchlist ticker(s) matched this scan.")
    if not watchlist_only:
        return out
    filtered = out[watched_mask].copy()
    if filtered.empty:
        st.info("None of your active watchlist tickers are in the current scan results.")
    return filtered
