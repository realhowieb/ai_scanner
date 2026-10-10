"""Calendar-aware presentation metadata; never changes scores or market inputs."""
import datetime as dt

from analytics import market_calendar as mc

UTC = dt.timezone.utc
SCAN_SLOT_GRACE = dt.timedelta(minutes=45)
SCAN_SLOT_EARLY = dt.timedelta(minutes=10)


def timestamp(value):
    try:
        parsed = value if isinstance(value, dt.datetime) else dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC)
    except (ValueError, TypeError):
        return None


def last_due_market_scan(now, days=10):
    now = timestamp(now)
    today = now.astimezone(mc.ET).date()
    for i in range(days):
        day = today - dt.timedelta(days=i)
        opened, closed = mc.session_bounds_utc(day)
        slots = [s for s in mc.expected_scan_slots(day) if opened <= s < closed and s + SCAN_SLOT_GRACE <= now]
        if slots:
            return max(slots)
    return None


def scan_freshness(latest, now):
    due = last_due_market_scan(now)
    latest = timestamp(latest)
    stale = due is not None and (latest is None or latest < due - SCAN_SLOT_EARLY)
    return {'stale': stale, 'expected_scan_at': due.isoformat() if stale else None}


def describe(scan_saved_at, *, market_data_at=None, scan_completed_at=None, now=None):
    """Stored scan time is not a provider quote/bar time or job completion time.

    Legacy snapshots have no reliable underlying-data watermark. Unknown is
    explicit; a fresh run alone cannot vouch for the freshness of its inputs.
    """
    now = timestamp(now) or dt.datetime.now(UTC)
    scan, data, completed = map(timestamp, (scan_saved_at, market_data_at, scan_completed_at))
    due = last_due_market_scan(now)
    covered = mc.calendar_covered(now.astimezone(mc.ET).date())
    def state(value):
        if value is None or value > now + dt.timedelta(minutes=5):
            return 'unavailable'
        if not covered:
            return 'partial'
        return 'stale' if due and value < due - SCAN_SLOT_EARLY else 'fresh'
    scan_state, data_state = state(scan), state(data)
    overall = ('unavailable' if scan_state == 'unavailable' else 'stale'
               if 'stale' in (scan_state, data_state) else 'fresh'
               if scan_state == data_state == 'fresh' else 'partial')
    return {'state': overall, 'scan_state': scan_state, 'market_data_state': data_state,
            'last_successful_scan_at': scan.isoformat() if scan else None,
            'scan_completed_at': completed.isoformat() if completed else None,
            'market_data_at': data.isoformat() if data else None,
            'expected_scan_at': due.isoformat() if due else None,
            'calendar_covered': covered, 'checked_at': now.isoformat(),
            'timestamp_basis': 'saved_scan',
            'market_data_note': None if data else 'Underlying market-data timestamp unavailable'}


def display_lines(info):
    labels = {'fresh': 'Fresh', 'stale': 'Stale', 'partial': 'Partially available', 'unavailable': 'Unavailable'}
    def fmt(value):
        parsed = timestamp(value)
        return parsed.astimezone(mc.ET).strftime('%b %d, %I:%M %p %Z') if parsed else 'Unavailable'
    return [f"Data freshness: {labels.get(info.get('state'), 'Unavailable')}",
            f"Last successful scan (saved): {fmt(info.get('last_successful_scan_at'))}",
            f"Scan completed: {fmt(info.get('scan_completed_at'))}",
            f"Market data as of: {fmt(info.get('market_data_at'))}"]
