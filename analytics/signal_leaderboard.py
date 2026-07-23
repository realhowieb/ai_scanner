"""Signal leaderboard: which scanner signal actually predicted forward moves.

Generalizes the two-way A/B in analytics.track_record to a whole slate of
signals. For each candidate signal we take each historical snapshot's top-N by
that signal and measure benchmark-excess forward return, then rank the signals
by average excess. Reuses the same snapshot-loading, bar-download, and
forward-return machinery as the track record (one shared bar download).

Heavy (bar downloads) — run once/day from the cron, not per UI render. Best-
effort and deterministic; returns None rather than raising.
"""
from __future__ import annotations

from statistics import mean, median
from typing import Any, Dict, List, Optional

from analytics.track_record import (
    BENCHMARK,
    _bars_for,
    _eligible_snapshots,
    _forward_return,
    _symbol_column,
)

TOP_N = 5

# Candidate signals to rank. Each entry: (key, display, column-candidates).
# We rank top-N by the first present column (descending = most bullish).
SIGNALS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("breakout", "Breakout score", ("BreakoutScore",)),
    ("prebreakout", "Pre-breakout prob", ("PreBreakoutProb%",)),
    ("ai_confidence", "AI confidence", ("AI Confidence",)),
    ("rel_volume", "Relative volume", ("VolRel20", "RelVol")),
    ("day_move", "Day move %", ("PctChange",)),
    ("gap", "Gap %", ("GapPct",)),
    ("trend20", "20-day trend", ("Trend20D%",)),
)


def _signal_column(df, candidates: tuple[str, ...]) -> Optional[str]:
    for col in candidates:
        if col in getattr(df, "columns", []):
            return col
    return None


def _top_symbols_by(df, sym_col: str, value_col: str, top_n: int) -> List[str]:
    """Top-N symbols by a numeric column (descending), NaNs dropped."""
    try:
        work = df[[sym_col, value_col]].copy()
        work = work.dropna(subset=[value_col])
        if work.empty:
            return []
        # A flat column (all equal, e.g. all-zero PreBreakoutProb% with no model)
        # is not a ranking — skip it so we don't fake the snapshot's stored order.
        if float(work[value_col].max()) == float(work[value_col].min()):
            return []
        work = work.sort_values(value_col, ascending=False)
        return [
            str(s).upper()
            for s in work[sym_col].head(top_n).tolist()
            if str(s).strip()
        ]
    except Exception:
        return []


def compute_signal_leaderboard(
    horizon_days: int = 5,
    lookback_days: int = 45,
    max_snapshots: int = 30,
    entry_mode: str = "close",
) -> Optional[List[Dict[str, Any]]]:
    """Rank each signal by benchmark-excess forward return across snapshots.

    Returns a list of per-signal summary dicts, best average-excess first, or
    None when there isn't enough history.
    """
    snapshots = _eligible_snapshots(horizon_days, lookback_days, max_snapshots)
    if not snapshots:
        return None

    # Per-snapshot top-N picks under each signal; collect the symbol union once.
    per_snapshot: List[tuple] = []  # (run_date, {signal_key: [symbols]})
    all_symbols = {BENCHMARK}
    for run_date, df in snapshots:
        sym_col = _symbol_column(df)
        if not sym_col:
            continue
        picks: Dict[str, List[str]] = {}
        for key, _display, cols in SIGNALS:
            value_col = _signal_column(df, cols)
            syms = _top_symbols_by(df, sym_col, value_col, TOP_N) if value_col else []
            if syms:
                picks[key] = syms
                all_symbols.update(syms)
        per_snapshot.append((run_date, picks))

    if all_symbols == {BENCHMARK}:
        return None

    try:
        from data.price_alpaca import download_multi_alpaca

        bars_by_symbol = download_multi_alpaca(
            sorted(all_symbols),
            period=f"{lookback_days + horizon_days + 10}d",
            interval="1d",
            prepost=False,
            timeout_s=25.0,
        )
    except Exception:
        return None
    if not bars_by_symbol:
        return None
    spy_bars = bars_by_symbol.get(BENCHMARK)
    if spy_bars is None:
        return None

    display_by_key = {key: display for key, display, _ in SIGNALS}
    rows: List[Dict[str, Any]] = []
    for key, _display, _cols in SIGNALS:
        excess: List[float] = []
        runs_counted = 0
        for run_date, picks in per_snapshot:
            syms = picks.get(key) or []
            if not syms:
                continue
            spy_ret = _forward_return(spy_bars, run_date, horizon_days, entry_mode)
            if spy_ret is None:
                continue
            day_had = False
            for sym in syms:
                bars = _bars_for(bars_by_symbol, sym)
                if bars is None:
                    continue
                r = _forward_return(bars, run_date, horizon_days, entry_mode)
                if r is not None:
                    excess.append(r - spy_ret)
                    day_had = True
            if day_had:
                runs_counted += 1
        if not excess:
            continue
        beats = sum(1 for e in excess if e > 0)
        rows.append({
            "signal": key,
            "display": display_by_key.get(key, key),
            "horizon_days": horizon_days,
            "avg_excess": mean(excess),
            "median_excess": median(excess),
            "win_rate": beats / len(excess),
            "sample_size": len(excess),
            "runs_used": runs_counted,
            "benchmark": BENCHMARK,
            "top_n": TOP_N,
        })
    if not rows:
        return None
    rows.sort(key=lambda r: r["avg_excess"], reverse=True)
    return rows
