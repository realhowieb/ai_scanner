"""Pure strategy post-filters for scanner result DataFrames."""
from __future__ import annotations

from typing import Callable

import pandas as pd


STRATEGY_NAMES = {
    "gap_up",
    "gap_down",
    "most_active",
    "unusual_vol",
    "momentum",
    "breakout_only",
}


def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.head(0)


def strategy_gap_up(frame: pd.DataFrame) -> pd.DataFrame:
    if "GapPct" not in frame.columns:
        return _empty_like(frame)
    return frame[frame["GapPct"] > 0].sort_values("GapPct", ascending=False)


def strategy_gap_down(frame: pd.DataFrame) -> pd.DataFrame:
    if "GapPct" not in frame.columns:
        return _empty_like(frame)
    return frame[frame["GapPct"] < 0].sort_values("GapPct", ascending=True)


def strategy_most_active(frame: pd.DataFrame) -> pd.DataFrame:
    col = "DollarVol20" if "DollarVol20" in frame.columns else (
        "Volume" if "Volume" in frame.columns else None
    )
    if col is None:
        return _empty_like(frame)
    return frame.sort_values(col, ascending=False)


def strategy_unusual_vol(frame: pd.DataFrame) -> pd.DataFrame:
    if "VolRel20" not in frame.columns:
        return _empty_like(frame)
    return frame[frame["VolRel20"] >= 2].sort_values("VolRel20", ascending=False)


def strategy_momentum(frame: pd.DataFrame) -> pd.DataFrame:
    if "Trend20D%" not in frame.columns or "Trend10D%" not in frame.columns:
        return _empty_like(frame)
    mask = (frame["Trend20D%"] > 0) & (frame["Trend10D%"] > 0)
    return frame[mask].sort_values("Trend20D%", ascending=False)


def strategy_breakout_only(frame: pd.DataFrame) -> pd.DataFrame:
    if "IsBreakout" not in frame.columns:
        return _empty_like(frame)
    base = frame[frame["IsBreakout"] == True]
    if "BreakoutScore" in base.columns:
        return base.sort_values("BreakoutScore", ascending=False)
    return base


STRATEGY_FILTERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "gap_up": strategy_gap_up,
    "gap_down": strategy_gap_down,
    "most_active": strategy_most_active,
    "unusual_vol": strategy_unusual_vol,
    "momentum": strategy_momentum,
    "breakout_only": strategy_breakout_only,
}


def apply_strategy_filter(strategy: object, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply a named strategy post-filter, returning frame unchanged for unknown strategies."""
    if frame is None or not isinstance(frame, pd.DataFrame):
        return pd.DataFrame()
    strategy_name = str(strategy or "").strip().lower()
    filter_fn = STRATEGY_FILTERS.get(strategy_name)
    if filter_fn is None:
        return frame
    return filter_fn(frame)
