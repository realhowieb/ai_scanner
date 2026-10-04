"""P1-56: one real-time price-alert pass, dispatched every 5 minutes by cron-job.org.

Same evaluation as the Render worker (billing_service.realtime_alerts.check_once):
Alpaca snapshots for all enabled price alerts past their throttle, then an
in-app event and, for verified Pro+ accounts, an email. The Render worker and
the scheduled scans share user_alerts.last_fired_at, so running all three
never sends the same alert twice.

Exits quietly outside extended hours (4:00-20:00 ET) and on market holidays.
Exit 1 only when the pass itself fails, so a broken run shows up in Actions.

    DATABASE_URL=... ALPACA_API_KEY_ID=... python scripts/realtime_alerts_once.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def should_run(now: dt.datetime) -> tuple[bool, str]:
    """(run?, reason). Extended hours on a trading day only."""
    from analytics import market_calendar as mc
    from billing_service.realtime_alerts import market_session_open

    if not market_session_open(now):  # weekends included
        return False, "outside extended hours (4:00-20:00 ET, Mon-Fri)"
    day = now.astimezone(mc.ET).date()
    if mc.calendar_covered(day) and not mc.is_trading_day(day):
        return False, "market holiday"
    return True, ""


def main(now: dt.datetime | None = None) -> int:
    from billing_service.realtime_alerts import check_once

    now = now or dt.datetime.now(dt.timezone.utc)
    ok, reason = should_run(now)
    if not ok:
        print(f"[realtime_alerts] skipped: {reason}")
        return 0
    try:
        fired = check_once()
    except Exception as e:  # logged without values; the worker redacts emails itself
        print(f"[realtime_alerts] pass failed: {type(e).__name__}")
        return 1
    print(f"[realtime_alerts] pass complete: fired {fired} alert(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
