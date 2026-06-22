"""Shared scan execution helpers.

The Streamlit UI owns progress widgets and persistence; this module owns the
testable engine-call and result-filtering rules used by manual scan buttons.
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .strategies import apply_strategy_filter


ScanRunner = Callable[..., pd.DataFrame]
ProgressCallback = Callable[..., None]


def _call_scan_runner(
    runner: ScanRunner,
    *,
    tickers: list[str],
    premarket: bool,
    afterhours: bool,
    unusual_volume: bool,
    min_gap: float,
    min_price: float,
    max_price: float,
    top_n: int,
    profile: str,
    diagnostics: bool,
    progress_cb: ProgressCallback | None,
    snapshot_id: object = None,
) -> pd.DataFrame:
    """Call the scan runner with compatibility fallbacks for older signatures."""
    kwargs = {
        "tickers": list(tickers),
        "premarket": premarket,
        "afterhours": afterhours,
        "unusual_volume": unusual_volume,
        "min_gap": min_gap,
        "min_price": min_price,
        "max_price": max_price,
        "top_n": top_n,
        "profile": profile,
        "diagnostics": diagnostics,
    }
    try:
        return runner(**kwargs, progress_cb=progress_cb, snapshot_id=snapshot_id)
    except TypeError:
        try:
            return runner(**kwargs, snapshot_id=snapshot_id)
        except TypeError:
            return runner(**kwargs)


def run_manual_scan_execution(
    *,
    runner: ScanRunner,
    tickers: list[str],
    premarket: bool,
    afterhours: bool,
    unusual_volume: bool,
    min_gap: float,
    min_price: float,
    max_price: float,
    top_n: int,
    profile: str,
    apply_gap_filter: bool,
    strategy: object = None,
    diagnostics: bool = False,
    progress_cb: ProgressCallback | None = None,
    snapshot_id: object = None,
    extended_price_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Run the manual scanner and apply shared post-processing rules."""
    frame = _call_scan_runner(
        runner,
        tickers=tickers,
        premarket=premarket,
        afterhours=afterhours,
        unusual_volume=unusual_volume,
        min_gap=min_gap,
        min_price=min_price,
        max_price=max_price,
        top_n=top_n,
        profile=profile,
        diagnostics=diagnostics,
        progress_cb=progress_cb,
        snapshot_id=snapshot_id,
    )
    if frame is None or not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame()

    if apply_gap_filter and not frame.empty:
        if "GapPct" in frame.columns:
            frame = frame.copy()
            frame["GapPct"] = pd.to_numeric(frame["GapPct"], errors="coerce").fillna(0.0)
            frame = frame[frame["GapPct"] >= float(min_gap)]
        else:
            frame = frame.head(0)

    if strategy and not frame.empty:
        frame = apply_strategy_filter(strategy, frame)

    if not frame.empty:
        frame = frame.head(int(top_n)).reset_index(drop=True)
        if (premarket or afterhours) and extended_price_transform is not None:
            frame = extended_price_transform(frame)

    return frame
