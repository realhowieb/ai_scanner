"""Versioned saved-signal research with timestamp-aware daily-bar proxies.

The default is next eligible session open to horizon-last close. Every selected
pick is evaluated or excluded explicitly; this is not a trading simulation.
Date-only `_forward_return` remains a separately documented legacy research
helper for economic-label callers, with no opening-price substitution.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import isfinite
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
        from analytics.prediction_provenance import models_from_row

        # Use only prediction-time evidence. Loading today's model here rewrites
        # past selections whenever the champion or its calibration changes.
        def saved_probability(row):
            evidence = models_from_row(row).get("prebreakout")
            if isinstance(evidence, dict):
                if evidence.get("status") != "captured":
                    return None
                value, scale = evidence.get("calibrated_probability"), 1.0
            elif "PreBreakoutProb%" in row:
                value, scale = row.get("PreBreakoutProb%"), 100.0
            else:
                value, scale = row.get("PreBreakoutProb"), 1.0
            try:
                value = float(value) / scale
                return value if isfinite(value) and 0.0 <= value <= 1.0 else None
            except (TypeError, ValueError):
                return None

        probabilities = df.apply(saved_probability, axis=1)
        # Legacy unavailable-model fallbacks wrote all zeros; never infer an order
        # from those, or from snapshots with no saved prediction.
        valid = probabilities.notna()
        if not valid.any() or probabilities[valid].max() <= 0:
            return []
        work = df.loc[valid].assign(_saved_probability=probabilities[valid]).sort_values(
            "_saved_probability", ascending=False, kind="stable")
    else:  # breakout
        if "BreakoutScore" in df.columns:
            work = df.sort_values("BreakoutScore", ascending=False, kind="stable")
    return list(dict.fromkeys(str(s).strip().upper() for s in work[sym_col].tolist() if str(s).strip()))[:top_n]


def _eligible_snapshots(horizon_days: int, lookback_days: int, max_snapshots: int, *, now=None):
    """Latest snapshot per ET exchange date; preserve saved availability time.

    Recent, incomplete outcomes are retained for explicit coverage reporting.
    ``created_at`` is the snapshot write/refresh instant, not bar/scan start time.
    """
    try:
        from db.runs import list_snapshot_runs, load_many_run_results
        from ui.app_runtime import normalize_results_to_df
    except Exception:
        return []

    now = now or datetime.now(timezone.utc)
    oldest = now - timedelta(days=int(lookback_days))

    # Snapshot-only, recency-bounded SQL (replaces scanning thousands of run
    # rows), then one batched fetch of the chosen snapshots' JSON.
    try:
        runs = list_snapshot_runs(days=int(lookback_days) + 2, limit=max_snapshots * 6, public_only=True) or []
    except Exception:
        return []

    chosen: List[dict] = []
    seen_dates: set = set()
    for r in runs:  # newest-first
        created = r.get("created_at")
        if not isinstance(created, datetime):
            continue
        if created.tzinfo is None:
            continue  # unknown availability cannot support time-aware entry
        if created < oldest:
            break
        if created > now:
            continue
        from analytics.market_calendar import ET, calendar_covered, is_trading_day
        run_date = created.astimezone(ET).date()
        # Fence out pre-epoch snapshots (they carry the old un-clipped
        # BreakoutScore ceiling; mixing them in poisons the track record).
        if run_date < SCORE_EPOCH:
            continue
        # Skip weekend (forced/manual) snapshots. Their picks enter on the next
        # session, so they duplicate that trading day's result — double-counting
        # the sample and painting spurious weekend cells in the daily heatmap.
        if not calendar_covered(run_date) or not is_trading_day(run_date):
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
        df.attrs["snapshot_id"] = int(r["id"])
        out.append((r["_created"], df))
    return out


def _forward_return(bars, run_date, horizon_days: int, entry_mode: str = "close") -> Optional[float]:
    """Return over `horizon` trading days from the first bar on/after run_date.

    entry_mode:
      "close" — enter at the signal-day close (default; original behavior).
      "open"  — enter at the signal-day open. For premarket signals this is the
                date-only research proxy; callers must not describe it as an
                executable fill. Missing Open is excluded, never replaced.
    Exit is always the close `horizon_days` bars later. The same mode is applied
    to the benchmark by the caller, so excess stays apples-to-apples.
    """
    try:
        if isinstance(run_date, datetime):
            from analytics.track_method import outcome, window
            result = outcome(bars, window(run_date, horizon_days, entry_mode))
            return result.get("return")
        if entry_mode not in ("open", "close"):
            return None
        closes = bars["Close"]
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
        if entry_mode == "open":
            if "Open" not in bars.columns:
                return None
            entry = float(bars["Open"].reindex(closes.index).iloc[entry_pos])
        else:
            entry = float(closes.iloc[entry_pos])
        exit_ = float(closes.iloc[exit_pos])
        if not isfinite(entry) or not isfinite(exit_) or entry <= 0 or exit_ <= 0:
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
    entry_mode: str = "open",
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Versioned time-aware research with explicit missing-data denominators."""
    from analytics.track_method import evaluate

    snapshots = _eligible_snapshots(horizon_days, lookback_days, max_snapshots)
    if not snapshots:
        return None
    selected = {rk: [(at, _ranked_symbols(df, rk, TOP_N), df.attrs.get("snapshot_id")) for at, df in snapshots] for rk in RANKINGS}
    symbols = {BENCHMARK} | {sym for rows in selected.values() for _, picks, _ in rows for sym in picks}
    try:
        from data.price_alpaca import download_multi_alpaca
        bars = download_multi_alpaca(sorted(symbols), period=f"{lookback_days + horizon_days*2 + 10}d",
                                    interval="1d", prepost=False, timeout_s=25.0) or {}
    except Exception:
        bars = {}  # selected picks still appear as exclusions
    results = {}
    for ranking, selections in selected.items():
        summary = evaluate(selections, bars, horizon_days, mode=entry_mode)
        summary.update(ranking=ranking, top_n=TOP_N, snapshots_considered=len(snapshots),
                       ranking_unavailable_snapshots=sum(not picks for _, picks, _ in selections))
        results[ranking] = summary
    return results
