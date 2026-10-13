"""Deliberate, user-independent freeze of HSF opportunities for calibration.

The scheduled cron pipeline owns the freeze, so signal-time HSF opportunities are
collected whether or not anyone opens the UI. They are frozen from the Scanner's
ranking of the latest saved full-market run (consolidate_scanner_results, the
same scores /v1/scans/latest shows), keyed on that run's created_at, so the
Track record buckets use the scores people actually saw. Idempotent per
(snapshot, ticker) via ON CONFLICT DO NOTHING.

Earlier rows came from the Market Brief's top-5 lists (build_opportunities),
which scored lower than the Scanner for the same name; readers drop those once a
snapshot has Scanner rows (db.signal_outcomes._SCANNER_SUPERSEDES_BRIEF), and
scripts/backfill_scanner_opportunities.py re-freezes past runs the same way.

Read-safe: returns 0 and never raises when data/DB is unavailable.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Ranked Scanner setups frozen per run: what a Basic plan sees in the Scanner
# (config.TIERS_CONFIG basic max_results). Deeper than the Brief's top 5, so the
# lower score buckets get evidence too.
FREEZE_TOP_N = 25


def market_runs(limit: int = 60) -> List[Dict[str, Any]]:
    """Scheduled full-market runs, newest first (parsed created_at)."""
    from db.runs import list_runs
    from ui.market_scans import MARKET_USER
    from ui.market_scans import market_runs as _market_runs

    return _market_runs(list_runs(limit=limit, include_snapshots=True, username=MARKET_USER) or [])


def scanner_opportunities(run_id: int, top_n: int = FREEZE_TOP_N) -> List[Dict[str, Any]]:
    """The Scanner's top-n ranked HSF opportunities of one saved run."""
    from ui.market_scans import _run_df_uncached, top_setups

    return top_setups(_run_df_uncached(int(run_id)), n=top_n)


def freeze_run(run: Dict[str, Any], top_n: int = FREEZE_TOP_N) -> int:
    """Freeze one saved run's Scanner opportunities at the run's created_at."""
    from db.signal_outcomes import SCANNER_SIGNAL_TYPE, freeze_opportunities

    ts = run.get("created_at")
    if ts is None or run.get("id") is None:
        return 0
    opps = scanner_opportunities(int(run["id"]), top_n)
    if not opps:
        return 0
    return freeze_opportunities(ts, opps, signal_type=SCANNER_SIGNAL_TYPE)


def freeze_latest_opportunities() -> int:
    """Freeze the latest full-market run's Scanner opportunities.

    Returns the number of freeze attempts (existing rows are no-ops via ON
    CONFLICT). Best-effort; 0 on any failure.
    """
    try:
        runs = market_runs(limit=5)
        return freeze_run(runs[0]) if runs else 0
    except Exception:
        return 0

