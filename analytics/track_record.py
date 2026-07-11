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


def _symbol_column(df) -> Optional[str]:
    for col in ("Symbol", "Ticker", "symbol", "ticker"):
        if col in df.columns:
            return col
    return None


def _eligible_snapshots(horizon_days: int, lookback_days: int, max_snapshots: int):
    """Yield (run_date, symbols) for snapshots with a complete forward window."""
    try:
        from db.runs import list_runs, load_run_results
        from ui.app_runtime import normalize_results_to_df
    except Exception:
        return []

    now = datetime.now(timezone.utc)
    # A snapshot needs `horizon` trading days after it; ~1.5 calendar days each.
    complete_before = now - timedelta(days=int(horizon_days * 1.6) + 2)
    oldest = now - timedelta(days=int(lookback_days))

    try:
        runs = list_runs(limit=200) or []
    except Exception:
        return []

    out: List[tuple] = []
    for r in runs:
        if not r.get("is_snapshot"):
            continue
        created = r.get("created_at")
        if not isinstance(created, datetime):
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if not (oldest <= created <= complete_before):
            continue
        try:
            raw = load_run_results(r["id"])
            df = normalize_results_to_df(raw) if raw else None
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        sym_col = _symbol_column(df)
        if not sym_col:
            continue
        # Top candidates as presented (already ranked); bound per-snapshot.
        symbols = [str(s).upper() for s in df[sym_col].head(20).tolist() if str(s).strip()]
        if symbols:
            out.append((created.date(), symbols))
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


def compute_track_record(
    horizon_days: int = 5,
    lookback_days: int = 45,
    max_snapshots: int = 30,
) -> Optional[Dict[str, Any]]:
    """Compute forward-return performance across recent snapshot candidates."""
    snapshots = _eligible_snapshots(horizon_days, lookback_days, max_snapshots)
    if not snapshots:
        return None

    all_symbols = sorted({s for _, syms in snapshots for s in syms})
    if not all_symbols:
        return None

    try:
        from data.price_alpaca import download_multi_alpaca

        bars_by_symbol = download_multi_alpaca(
            all_symbols,
            period=f"{lookback_days + horizon_days + 10}d",
            interval="1d",
            prepost=False,
            timeout_s=20.0,
        )
    except Exception:
        return None
    if not bars_by_symbol:
        return None

    returns: List[float] = []
    for run_date, symbols in snapshots:
        for sym in symbols:
            # Avoid `a or b` here: bars are DataFrames and their truth value is
            # ambiguous. Resolve the alias (dash vs dot class shares) explicitly.
            bars = bars_by_symbol.get(sym)
            if bars is None:
                bars = bars_by_symbol.get(sym.replace("-", "."))
            if bars is None:
                continue
            r = _forward_return(bars, run_date, horizon_days)
            if r is not None:
                returns.append(r)

    if not returns:
        return None

    wins = sum(1 for r in returns if r > 0)
    return {
        "horizon_days": horizon_days,
        "avg_return": mean(returns),
        "median_return": median(returns),
        "win_rate": wins / len(returns),
        "sample_size": len(returns),
        "runs_used": len(snapshots),
    }
