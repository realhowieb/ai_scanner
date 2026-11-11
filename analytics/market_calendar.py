"""Run 59 — deterministic US equity market calendar for health checks (pure).

The scheduler skips weekends and asks Alpaca's calendar API about holidays at
run time. The health plane must reason about the past and future offline and
deterministically, so it uses this explicit NYSE calendar instead of a naive
"now - last_scan > X hours" rule. Dates outside `COVERED_YEARS` fall back to
weekday-only logic and are flagged `calendar_covered = False`.

Sources: NYSE published holiday / early-close schedules.
"""
from __future__ import annotations

import datetime as _dt
from typing import Iterator, List, Optional, Tuple
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = _dt.timezone.utc
COVERED_YEARS = (2025, 2026, 2027)

NYSE_HOLIDAYS = frozenset(_dt.date.fromisoformat(d) for d in (
    # 2025
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
))
# 13:00 ET close.
NYSE_EARLY_CLOSES = frozenset(_dt.date.fromisoformat(d) for d in (
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-11-27", "2026-12-24",
    "2027-11-26",
))

# The external scheduler (cron-job.org -> workflow_dispatch on main) fires the
# scheduled scan at these New York times on weekdays; measured reliable to +-1
# minute. The scan itself skips non-trading days. The jobs run in the
# America/New_York time zone, so the slots keep the same market times all year
# (pre-market 8:35, open 9:35, midday 12:35, late day 15:35, after hours 16:35
# and 17:35 ET) and their UTC times move with daylight saving time.
SCAN_SLOTS_ET: Tuple[Tuple[int, int], ...] = ((8, 35), (9, 35), (12, 35), (15, 35), (16, 35), (17, 35))

# Until 2026-11-03 the cron-job.org jobs ran on fixed UTC times. Those equal the
# New York slots above in summer (EDT), but after DST ended on 2026-11-01 every
# scan ran an hour early (7:35 ... 16:35 ET, nothing between 12:00 and 14:00 ET).
# Days before SCAN_SLOTS_ET_SINCE are judged against the UTC times that really ran.
LEGACY_SCAN_SLOTS_UTC: Tuple[Tuple[int, int], ...] = ((12, 35), (13, 35), (16, 35), (19, 35), (20, 35), (21, 35))
SCAN_SLOTS_ET_SINCE = _dt.date(2026, 11, 4)

# Hourly refresh scans (cron-job.org HSF_Refresh_Scan, record_research=false,
# since 2026-11-10) update the Scanner between the slots above and record no
# research observations by design.
REFRESH_SCAN_SLOTS_ET: Tuple[Tuple[int, int], ...] = ((10, 35), (11, 35), (13, 35), (14, 35))


def is_refresh_scan_time(ts: _dt.datetime) -> bool:
    """True when `ts` falls in a refresh scan window (slot -5 min to +20 min, New York time)."""
    et = ts.astimezone(ET)
    minute = et.hour * 60 + et.minute
    return any(-5 <= minute - (h * 60 + m) <= 20 for h, m in REFRESH_SCAN_SLOTS_ET)


def calendar_covered(d: _dt.date) -> bool:
    return d.year in COVERED_YEARS


def is_trading_day(d: _dt.date) -> bool:
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def close_time_et(d: _dt.date) -> _dt.time:
    return _dt.time(13, 0) if d in NYSE_EARLY_CLOSES else _dt.time(16, 0)


def session_bounds_utc(d: _dt.date) -> Tuple[_dt.datetime, _dt.datetime]:
    open_et = _dt.datetime.combine(d, _dt.time(9, 30), ET)
    close_et = _dt.datetime.combine(d, close_time_et(d), ET)
    return open_et.astimezone(UTC), close_et.astimezone(UTC)


def is_market_open(now: _dt.datetime) -> bool:
    d = now.astimezone(ET).date()
    if not is_trading_day(d):
        return False
    o, c = session_bounds_utc(d)
    return o <= now < c


def trading_days_between(start: _dt.date, end: _dt.date) -> Iterator[_dt.date]:
    """Trading days in [start, end]."""
    d = start
    while d <= end:
        if is_trading_day(d):
            yield d
        d += _dt.timedelta(days=1)


def previous_trading_day(d: _dt.date) -> _dt.date:
    d -= _dt.timedelta(days=1)
    while not is_trading_day(d):
        d -= _dt.timedelta(days=1)
    return d


def expected_scan_slots(d: _dt.date) -> List[_dt.datetime]:
    """UTC datetimes at which a scheduled scan is expected on ET date `d`."""
    if not is_trading_day(d):
        return []
    if d < SCAN_SLOTS_ET_SINCE:
        return [_dt.datetime(d.year, d.month, d.day, h, m, tzinfo=UTC) for h, m in LEGACY_SCAN_SLOTS_UTC]
    return [_dt.datetime(d.year, d.month, d.day, h, m, tzinfo=ET).astimezone(UTC) for h, m in SCAN_SLOTS_ET]


def slot_label(slot: _dt.datetime) -> str:
    """'HH:MM' New York time of a scan slot (stable across daylight saving time)."""
    return slot.astimezone(ET).strftime("%H:%M")


def snapshot_scan_slots(d: _dt.date) -> List[_dt.datetime]:
    """Expected scan slots on `d` that fall in regular hours.

    Only regular-session scans save a Scanner snapshot; pre-market and
    after-hours slots run their own session scan and freeze nothing. The split
    moves with daylight saving time and early closes, so it is computed per day.
    """
    o, c = session_bounds_utc(d)
    return [s for s in expected_scan_slots(d) if o <= s < c]


def next_expected_scan(now: _dt.datetime, *, horizon_days: int = 10) -> Optional[_dt.datetime]:
    d = now.astimezone(ET).date()
    for i in range(horizon_days):
        day = d + _dt.timedelta(days=i)
        for s in expected_scan_slots(day):
            if s > now:
                return s
    return None


def completed_trading_days_since(ts: _dt.datetime, now: _dt.datetime) -> int:
    """Trading days that fully closed after `ts` and before `now` (freshness in
    market time: a Friday-evening artifact is not stale on Sunday)."""
    start = ts.astimezone(ET).date()
    n = 0
    for d in trading_days_between(start, now.astimezone(ET).date()):
        _o, c = session_bounds_utc(d)
        if c > ts and c <= now:
            n += 1
    return n
