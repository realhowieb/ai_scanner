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

from config import SCORE_EPOCH  # single source of truth

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
            # score_prebreakout returns all-zero probabilities when no trained
            # model is available. Zeros are not a ranking — treating them as one
            # silently degrades to the snapshot's stored order and fakes the
            # A/B comparison. Require a real signal.
            if float(scored["PreBreakoutProb%"].max() or 0.0) <= 0.0:
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
        from db.runs import list_snapshot_runs, load_many_run_results
        from ui.app_runtime import normalize_results_to_df
    except Exception:
        return []

    now = datetime.now(timezone.utc)
    # A snapshot needs `horizon` trading days after it; ~1.5 calendar days each.
    complete_before = now - timedelta(days=int(horizon_days * 1.6) + 2)
    oldest = now - timedelta(days=int(lookback_days))

    # Snapshot-only, recency-bounded SQL (replaces scanning thousands of run
    # rows), then one batched fetch of the chosen snapshots' JSON.
    try:
        runs = list_snapshot_runs(days=int(lookback_days) + 2, limit=max_snapshots * 6) or []
    except Exception:
        return []

    chosen: List[dict] = []
    seen_dates: set = set()
    for r in runs:  # newest-first
        created = r.get("created_at")
        if not isinstance(created, datetime):
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if created < oldest:
            break
        if created > complete_before:
            continue
        run_date = created.date()
        # Fence out pre-epoch snapshots (they carry the old un-clipped
        # BreakoutScore ceiling; mixing them in poisons the track record).
        if run_date < SCORE_EPOCH:
            continue
        # One snapshot per trading day. Multiple same-day snapshots (premarket /
        # regular / postmarket sessions + forced runs) share near-identical picks
        # and would otherwise be counted as independent observations, inflating n
        # and letting a single bad day dominate. Newest-first, so the first seen
        # for a date is that day's latest (most complete) snapshot.
        if run_date in seen_dates:
            continue
        seen_dates.add(run_date)
        r["_created"] = created
        chosen.append(r)
        if len(chosen) >= max_snapshots:
            break
    payloads = load_many_run_results([r["id"] for r in chosen])

    out: List[tuple] = []
    for r in chosen:
        try:
            raw = payloads.get(int(r["id"]))
            df = normalize_results_to_df(raw) if raw else None
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        if not _symbol_column(df):
            continue
        out.append((r["_created"].date(), df))
    return out


def _forward_return(bars, run_date, horizon_days: int, entry_mode: str = "close") -> Optional[float]:
    """Return over `horizon` trading days from the first bar on/after run_date.

    entry_mode:
      "close" — enter at the signal-day close (default; original behavior).
      "open"  — enter at the signal-day open. For premarket signals this is the
                realistic actionable fill and captures the signal-day move,
                which the close-entry convention discards.
    Exit is always the close `horizon_days` bars later. The same mode is applied
    to the benchmark by the caller, so excess stays apples-to-apples.
    """
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
        if entry_mode == "open" and "Open" in getattr(bars, "columns", []):
            entry_series = bars["Open"].reindex(closes.index)
            entry = float(entry_series.iloc[entry_pos])
            if entry != entry or entry <= 0:  # NaN/invalid open → fall back to close
                entry = float(closes.iloc[entry_pos])
        else:
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
    entry_mode: str = "close",
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
        daily: List[tuple] = []  # (run_date, mean_excess, n_picks)
        runs_counted = 0
        for run_date, picks in per_snapshot:
            spy_ret = _forward_return(spy_bars, run_date, horizon_days, entry_mode)
            if spy_ret is None:
                continue
            day_vals: List[float] = []
            for sym in picks.get(ranking, []):
                bars = _bars_for(bars_by_symbol, sym)
                if bars is None:
                    continue
                r = _forward_return(bars, run_date, horizon_days, entry_mode)
                if r is not None:
                    excess.append(r - spy_ret)
                    day_vals.append(r - spy_ret)
            if day_vals:
                runs_counted += 1
                daily.append((run_date, mean(day_vals), len(day_vals)))
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
            "daily": daily,
        }
    return results or None
