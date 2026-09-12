"""Deliberate, user-independent freeze of HSF opportunities for calibration.

Run 19 made Scanner Results read-only, and the Market Brief render freezes
opportunities only when a user opens the brief. This module lets the scheduled
cron pipeline own the freeze too, so signal-time HSF opportunities are collected
whether or not anyone views the UI. Reuses the SAME canonical build +
freeze_opportunities, keyed on the scan snapshot time, so it is idempotent with
the brief's freeze (ON CONFLICT DO NOTHING on the same (snapshot, ticker)).

Read-safe: returns 0 and never raises when data/DB is unavailable.
"""
from __future__ import annotations


def freeze_latest_opportunities() -> int:
    """Compute the latest snapshot's HSF opportunities and freeze them.

    Returns the number of freeze attempts (existing rows are no-ops via ON
    CONFLICT). Best-effort; 0 on any failure.
    """
    try:
        from db.signal_outcomes import freeze_opportunities
        from ui.market_brief import _base_ticker, _compute_brief
        from ui.opportunities import build_opportunities
    except Exception:
        return 0
    try:
        data = _compute_brief()
        if not data:
            return 0
        ts = data.get("snapshot_time")
        if ts is None:
            return 0
        opps = build_opportunities(data, base_ticker=_base_ticker)
        if not opps:
            return 0
        return freeze_opportunities(ts, opps)
    except Exception:
        return 0
