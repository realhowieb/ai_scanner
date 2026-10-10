"""One cached existing health snapshot; no request-time history/Actions queries."""
import datetime as dt
import time

from analytics.operations import summary
from api.today import _cached

TTL_S = 120
STALE_S = 3600
_last_good = None


def load_snapshot():
    from db.system_health import load_latest
    def load():
        global _last_good
        try:
            snapshot = load_latest()
        except Exception:
            snapshot = None
        if snapshot:
            _last_good = (time.monotonic(), snapshot)
            return {'report': snapshot, 'refresh_failed': False}
        if _last_good and time.monotonic() - _last_good[0] <= STALE_S:
            return {'report': _last_good[1], 'refresh_failed': True}
        return {'report': None, 'refresh_failed': True}
    try:
        # Cache failure envelopes too: at most one retry per TTL, not per visit.
        return _cached('operations_health', load, ttl_s=TTL_S)
    except Exception:
        return {'report': None, 'refresh_failed': True}


def get_summary(now=None):
    snapshot = load_snapshot()
    out = summary(snapshot['report'], now=now or dt.datetime.now(dt.timezone.utc))
    out['refresh_failed'] = snapshot['refresh_failed']
    out['stale'] = out['stale'] or out['refresh_failed']
    return out
