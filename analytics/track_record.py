"""Compute the signal track record: forward returns of past scan candidates.

For each historical daily snapshot old enough to have a complete forward window,
take its top candidates and measure the return from the scan date to `horizon`
trading days later using daily bars. Aggregate into avg/median return, win rate,
and sample size. Heavy (bar downloads) — run once/day from the cron, not per UI
render. Deterministic and best-effort; returns None rather than raising.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import mean, median
from typing import Any, Dict, List, Optional

TOP_N = 5
BENCHMARK = "SPY"
RANKINGS = ("breakout", "prebreakout")


def _symbol_column(df) -> Optional[str]:
    for col in ("Symbol", "Ticker", "symbol", "ticker"):
        if col in df.columns:
            return col
    return None


def _ranked_symbols(df, ranking: str, top_n: int) -> List[str]:
    """Top-N symbols under a ranking method ('breakout' | 'prebreakout')."""
    sym_col = _symbol_column(df)
    if not sym_col:
        return []
    work = df
    if ranking == "prebreakout":
        try:
            from ml_prebreakout import score_prebreakout

            scored = score_prebreakout(df)
            if scored is None or "PreBreakoutProb%" not in scored.columns:
                return []
            work = scored.sort_values("PreBreakoutProb%", ascending=False)
        except Exception:
            return []
    else:  # breakout
        if "BreakoutScore" in df.columns:
            work = df.sort_values("BreakoutScore", ascending=False)
    return [str(s).upper() for s in work[sym_col].head(top_n).tolist() if str(s).strip()]


def _eligible_snapshots(horizon_days: int, lookback_days: int, max_snapshots: int):
    """Yield (run_date, df) for snapshots with a complete forward window."""
    try:
        from db.runs import list_runs, load_run_results
        from ui.app_runtime import normalize_results_to_df
    except Exception:
        return []

    now = datetime.now(timezone.utc)
    # A snapshot needs `horizon` trading days after it; ~1.5 calendar days each.
    complete_before = now - timedelta(days=int(horizon_days * 1.6) + 2)
    oldest = now - timedelta(days=int(lookback_days))

    # Use a large limit: snapshots are a small fraction of rows and can be buried
    # far down the (newest-first) run log, so a small window misses the aged
    # snapshots that actually have a complete forward return.
    try:
        runs = list_runs(limit=5000) or []
    except Exception:
        return []

    out: List[tuple] = []
    for r in runs:
        created = r.get("created_at")
        if not isinstance(created, datetime):
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        # Rows are newest-first; once we pass the lookback floor we can stop.
        if created < oldest:
            break
        if not r.get("is_snapshot"):
            continue
        if created > complete_before:
            continue
        try:
            raw = load_run_results(r["id"])
            df = normalize_results_to_df(raw) if raw else None
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        if not _symbol_column(df):
            continue
        out.append((created.date(), df))
        if len(out) >= max_snapshots:
            break
    return out


def _forward_return(bars, run_date, horizon_days: int) -> Optional[float]:
    """Return over `horizon` trading days from the first bar on/after run_date."""
    try:
        closes = bars["Close"].dropna()
        if closes.empty:
            return None
        # Position of the first bar dated on/after the scan date (entry).
        entry_pos = None
        for pos, ts in enumerate(closes.index):
            d = ts.date() if hasattr(ts, "date") else ts
            if d >= run_date:
                entry_pos = pos
                break
        if entry_pos is None:
            return None
        exit_pos = entry_pos + int(horizon_days)
        if exit_pos >= len(closes):
            return None
        entry = float(closes.iloc[entry_pos])
        exit_ = float(closes.iloc[exit_pos])
        if entry <= 0:
            return None
        return (exit_ - entry) / entry
    except Exception:
        return None


def _bars_for(bars_by_symbol, sym: str):
    """Resolve a symbol's bars, tolerating dash/dot class-share aliasing."""
    bars = bars_by_symbol.get(sym)
    if bars is None:
        bars = bars_by_symbol.get(sym.replace("-", "."))
    return bars


def compute_track_record(
    horizon_days: int = 5,
    lookback_days: int = 45,
    max_snapshots: int = 30,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """A/B forward performance of both rankings across recent snapshots.

    Returns {ranking: summary_dict} for each ranking that produced data, or None.
    Both rankings share one bar download (union of their picks + benchmark).
    """
    snapshots = _eligible_snapshots(horizon_days, lookback_days, max_snapshots)
    if not snapshots:
        return None

    # Per-snapshot top-N picks under each ranking; collect the symbol union once.
    per_snapshot: List[tuple] = []  # (run_date, {ranking: [symbols]})
    all_symbols = {BENCHMARK}
    for run_date, df in snapshots:
        picks = {rk: _ranked_symbols(df, rk, TOP_N) for rk in RANKINGS}
        per_snapshot.append((run_date, picks))
        for syms in picks.values():
            all_symbols.update(syms)

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

    results: Dict[str, Dict[str, Any]] = {}
    for ranking in RANKINGS:
        excess: List[float] = []
        runs_counted = 0
        for run_date, picks in per_snapshot:
            spy_ret = _forward_return(spy_bars, run_date, horizon_days)
            if spy_ret is None:
                continue
            used = False
            for sym in picks.get(ranking, []):
                bars = _bars_for(bars_by_symbol, sym)
                if bars is None:
                    continue
                r = _forward_return(bars, run_date, horizon_days)
                if r is not None:
                    excess.append(r - spy_ret)
                    used = True
            if used:
                runs_counted += 1
        if not excess:
            continue
        beats = sum(1 for e in excess if e > 0)
        results[ranking] = {
            "horizon_days": horizon_days,
            "avg_return": mean(excess),
            "median_return": median(excess),
            "win_rate": beats / len(excess),
            "sample_size": len(excess),
            "runs_used": runs_counted,
            "benchmark": BENCHMARK,
            "top_n": TOP_N,
            "ranking": ranking,
        }
    return results or None
