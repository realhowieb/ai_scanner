"""Quiet-hour policy shared by the API and standalone real-time worker (stdlib only)."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo


def quiet_at(q: dict, now: dt.datetime | None = None) -> bool:
    if not q.get("enabled"):
        return False
    local = (now or dt.datetime.now(dt.timezone.utc)).astimezone(ZoneInfo(q["timezone"]))
    clock = local.strftime("%H:%M")
    start, end = q["start"], q["end"]
    return start <= clock < end if start < end else (clock >= start or clock < end) if start > end else False
