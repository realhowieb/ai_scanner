"""One cached existing health snapshot; no request-time history/Actions queries."""
import datetime as dt
import threading
import time
from collections import deque

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


_lock = threading.Lock()
_requests: deque = deque(maxlen=1000)
_providers: deque = deque(maxlen=100)


def record_request(rid: str, method: str, route: str, status: int, ms: float) -> None:
    with _lock:
        _requests.append({"request_id": rid, "method": method, "route": route, "status": status,
                          "ms": round(ms, 1), "at": dt.datetime.now(dt.timezone.utc).isoformat()})


def record_provider(name: str, ok: bool, ms: float) -> None:
    with _lock:
        _providers.append({"name": name, "ok": ok, "ms": round(ms, 1), "at": dt.datetime.now(dt.timezone.utc).isoformat()})


def snapshot(query: str = "", problems_only: bool = False) -> dict:
    with _lock:
        rows, providers = list(reversed(_requests)), list(reversed(_providers))
    matches = [r for r in rows if (not problems_only or r["status"] >= 400 or r["ms"] >= 1000)
               and query.casefold() in f'{r["request_id"]} {r["route"]} {r["status"]}'.casefold()]
    return {"scope": "This API process since its last restart; up to 1,000 requests.", "requests": matches[:100],
            "providers": providers[:20], "total": len(rows),
            "errors": sum(r["status"] >= 500 for r in rows), "slow": sum(r["ms"] >= 1000 for r in rows)}
