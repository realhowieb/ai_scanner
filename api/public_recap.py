"""The public weekly recap (/weekly on the web): what HSF's scanner ranked in a finished
week, plus the biggest earnings reports of the week after.

    Ranked most this week   the first Scanner run of each trading day (frozen hsf_scanner
                            rows in signal_outcomes), its top TOP_PER_DAY by HSF Score,
                            grouped by ticker: days ranked, best score, setup type
    Earnings next week      the biggest TEASER_PER_DAY reporters a day (api.earnings_week),
                            with how many more report; scores stay in the app

No returns or results are shown: a single week of picks says nothing about an edge, and
the page must not read as a performance claim.

A week is published once it is over (Saturday 00:00 ET). Its recap is built on the first
request and stored in hsf_public_recaps, so it never changes afterwards; the earnings
part is only added while that "next week" is still ahead. Read-only for visitors.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any, Dict, List, Optional, Sequence

from api.today import TTLCache
from db.engine import schema_once
from utils.swallowed import swallowed

TOP_PER_DAY = 10
RANKED_MIN = 40  # same floor as "ranked" everywhere else (api/share.py)
VERSION = 2      # bump to rebuild stored weeks when the ranking rules change
SHOW = 12
TEASER_PER_DAY = 3
ARCHIVE = 12
WEEK_RE = re.compile(r"^(\d{4})-W(\d{2})$")
_cache = TTLCache(32)

TABLE_SQL = ("CREATE TABLE IF NOT EXISTS hsf_public_recaps (week TEXT PRIMARY KEY, payload TEXT NOT NULL, "
             "built_at TEXT NOT NULL)")


class BadWeek(ValueError):
    pass


def _et():
    from analytics import market_calendar as mc

    return mc.ET


def week_key(day: dt.date) -> str:
    y, w, _ = day.isocalendar()
    return f"{y}-W{w:02d}"


def week_monday(key: str) -> dt.date:
    m = WEEK_RE.match(str(key or ""))
    if not m:
        raise BadWeek("Use a week like 2026-W44.")
    try:
        return dt.date.fromisocalendar(int(m.group(1)), int(m.group(2)), 1)
    except ValueError:
        raise BadWeek("That week doesn't exist.") from None


def latest_week(now: dt.datetime) -> str:
    """The newest finished week: this week from Saturday on, else last week."""
    day = now.astimezone(_et()).date()
    monday = day - dt.timedelta(days=day.weekday())
    return week_key(monday if day.weekday() >= 5 else monday - dt.timedelta(days=7))


def is_published(key: str, now: dt.datetime) -> bool:
    return week_monday(key) <= week_monday(latest_week(now))


def summarize_ranked(rows: Sequence[tuple], tz=None, *, top: int = TOP_PER_DAY, show: int = SHOW) -> Dict[str, Any]:
    """rows: (ticker, fired_at, setup_score, primary_setup). The first run of each ET day,
    its top `top` by score, grouped by ticker (most days ranked, then best score)."""
    tz = tz or _et()
    first: Dict[dt.date, dt.datetime] = {}
    for _t, fired, *_ in rows:
        d = fired.astimezone(tz).date()
        if d not in first or fired < first[d]:
            first[d] = fired
    per_day: Dict[dt.date, List[tuple]] = {}
    for r in rows:
        if r[2] is not None and float(r[2]) >= RANKED_MIN and first.get(r[1].astimezone(tz).date()) == r[1]:
            per_day.setdefault(r[1].astimezone(tz).date(), []).append(r)
    names: Dict[str, Dict[str, Any]] = {}
    for d in sorted(per_day):
        for t, _f, score, setup in sorted(per_day[d], key=lambda x: -float(x[2]))[:top]:
            k = str(t).strip().upper()
            item = names.setdefault(k, {"ticker": k, "days": 0, "best_score": 0, "setup": None,
                                        "first_day": d.isoformat()})
            item["days"] += 1
            if round(float(score)) >= item["best_score"]:
                item["best_score"] = round(float(score))
                item["setup"] = setup or item["setup"]
    ranked = sorted(names.values(), key=lambda x: (-x["days"], -x["best_score"], x["ticker"]))
    return {"scan_days": len(per_day), "ranked_names": len(ranked), "ranked": ranked[:show]}


def load_ranked(monday: dt.date) -> Dict[str, Any]:
    from db.engine import get_neon_conn

    tz = _et()
    start = dt.datetime.combine(monday, dt.time(0, 0), tz)
    end = dt.datetime.combine(monday + dt.timedelta(days=5), dt.time(0, 0), tz)
    conn = get_neon_conn()
    if conn is None:
        return {"scan_days": 0, "ranked_names": 0, "ranked": []}
    try:
        cur = conn.cursor()
        cur.execute("SELECT ticker, fired_at, setup_score, indicators->>'primary_setup' AS setup FROM signal_outcomes "
                    "WHERE source = 'opportunity' AND signal_type = 'hsf_scanner' AND fired_at >= %s AND fired_at < %s",
                    (start, end))
        raw = cur.fetchall() or []
        cur.close()
    finally:
        conn.close()
    rows = [tuple(r.values()) if isinstance(r, dict) else tuple(r) for r in raw]
    return summarize_ranked([(t, f if f.tzinfo else f.replace(tzinfo=dt.timezone.utc), s, setup)
                             for t, f, s, setup in rows], tz)


def earnings_teaser(base: Dict[str, Any]) -> Dict[str, Any]:
    from api import earnings_week as ew

    cut = ew.for_plan(base, per_day=TEASER_PER_DAY, cap=0)
    days = [{"date": d["date"], "total": d["total"],
             "names": [{"ticker": n["ticker"], "time": n["time"]} for n in d["names"]]} for d in cut["days"]]
    return {"start": cut["start"], "end": cut["end"], "days": days}


def build(key: str, now: dt.datetime) -> Dict[str, Any]:
    monday = week_monday(key)
    payload: Dict[str, Any] = {"week": key, "start": monday.isoformat(),
                               "end": (monday + dt.timedelta(days=4)).isoformat(),
                               "built_at": now.astimezone(dt.timezone.utc).isoformat(), "v": VERSION, **load_ranked(monday)}
    try:
        from api import symbol_search

        names = symbol_search.names([r["ticker"] for r in payload["ranked"]])
        for r in payload["ranked"]:
            r["name"] = names.get(r["ticker"])
    except Exception as _ignored:
        swallowed("api.public_recap.build", _ignored)
    payload["earnings"] = None
    today = now.astimezone(_et()).date()
    if today <= monday + dt.timedelta(days=7):  # the week after hasn't started trading yet (Mon counts)
        try:
            from api import earnings_week as ew

            base = ew.build_base(max(today, monday + dt.timedelta(days=5)))
            if any(d["rows"] for d in base["days"]):
                payload["earnings"] = earnings_teaser(base)
        except Exception as _ignored:
            swallowed("api.public_recap.build", _ignored)
    return payload


# ---- stored snapshots -------------------------------------------------------------------------
@schema_once
def _ensure(conn) -> None:
    cur = conn.cursor()
    cur.execute(TABLE_SQL)
    conn.commit()
    cur.close()


def _conn():
    from db.engine import get_neon_conn

    conn = get_neon_conn()
    if conn is not None:
        _ensure(conn)
    return conn


def _stored(key: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT payload FROM hsf_public_recaps WHERE week = %s", (key,))
        row = cur.fetchone()
        cur.close()
    finally:
        conn.close()
    if not row:
        return None
    return json.loads(row["payload"] if isinstance(row, dict) else row[0])


def _store(payload: Dict[str, Any]) -> None:
    conn = _conn()
    if conn is None:
        return
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO hsf_public_recaps (week, payload, built_at) VALUES (%s, %s, %s) "
                    "ON CONFLICT (week) DO UPDATE SET payload = EXCLUDED.payload, built_at = EXCLUDED.built_at",
                    (payload["week"], json.dumps(payload), payload["built_at"]))
        conn.commit()
        cur.close()
    finally:
        conn.close()


def archive() -> List[Dict[str, str]]:
    """Stored weeks, newest first (only weeks that had ranked names are stored)."""
    conn = _conn()
    if conn is None:
        return []
    try:
        cur = conn.cursor()
        cur.execute("SELECT week FROM hsf_public_recaps ORDER BY week DESC LIMIT %s", (ARCHIVE,))
        rows = cur.fetchall() or []
        cur.close()
    finally:
        conn.close()
    out = []
    for r in rows:
        key = r["week"] if isinstance(r, dict) else r[0]
        monday = week_monday(key)
        out.append({"week": key, "start": monday.isoformat(), "end": (monday + dt.timedelta(days=4)).isoformat()})
    return out


def recap(key: Optional[str] = None, *, now: Optional[dt.datetime] = None) -> Optional[Dict[str, Any]]:
    """The recap for `key` (default: the newest finished week), or None when that week
    isn't over yet or had no scans. Raises BadWeek for a malformed key."""
    now = now or dt.datetime.now(dt.timezone.utc)
    key = key or latest_week(now)
    week_monday(key)
    if not is_published(key, now):
        return None

    def load() -> Optional[Dict[str, Any]]:
        try:
            found = _stored(key)
        except Exception:
            found = None
        if found and found.get("v") == VERSION:
            return found
        payload = build(key, now)
        if not payload["ranked"]:
            return None
        try:
            _store(payload)
        except Exception as _ignored:
            swallowed("api.public_recap.recap.load", _ignored)
        return payload

    return _cache.get(("recap", key), load, 600)


def public_index(now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """The latest recap and the archive list, for /weekly."""
    now = now or dt.datetime.now(dt.timezone.utc)
    latest = recap(None, now=now)
    try:
        weeks = _cache.get(("archive",), archive, 600)
    except Exception:
        weeks = []
    if latest and not any(w["week"] == latest["week"] for w in weeks):
        weeks = [{"week": latest["week"], "start": latest["start"], "end": latest["end"]}, *weeks][:ARCHIVE]
    return {"latest": latest, "weeks": weeks}
