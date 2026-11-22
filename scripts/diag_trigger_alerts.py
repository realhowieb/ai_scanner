"""Read-only check of breakout trigger alerts (alert_type 'trigger') on production data,
run through the Diagnostics workflow. Logs are public: user ids are masked.

1. Is the real-time worker alive? Fired price/move/rvol/trigger events per day since 11-17.
2. Every trigger alert: ticker, trigger price, enabled, created, last fired, plan.
3. Every trigger event fired so far.
4. Replay: for each trigger alert and each session since it was created, walk the IEX
   1-minute bars and report the first minute where the close was at/above the trigger
   with today's volume pace >= 1.2x (the worker's rule). A hit with no event = a miss.
Writes nothing.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ET = ZoneInfo("America/New_York")
PACE = 1.2


def mask(user: str) -> str:
    return "u_" + hashlib.sha256(str(user).encode()).hexdigest()[:8]


def alpaca(path: str, params: dict):
    import httpx

    base = (os.getenv("ALPACA_DATA_URL", "").strip() or "https://data.alpaca.markets").rstrip("/")
    resp = httpx.get(f"{base}{path}", params=params, timeout=20.0, headers={
        "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"]})
    resp.raise_for_status()
    return resp.json() or {}


def daily_bars(ticker: str, start: dt.date, end: dt.date):
    out, token = [], None
    while True:
        params = {"symbols": ticker, "timeframe": "1Day", "start": start.isoformat(),
                  "end": end.isoformat(), "limit": 1000, "adjustment": "raw", "feed": "iex"}
        if token:
            params["page_token"] = token
        js = alpaca("/v2/stocks/bars", params)
        out += (js.get("bars") or {}).get(ticker) or []
        token = js.get("next_page_token")
        if not token:
            return out


def minute_bars(ticker: str, day: dt.date):
    start = dt.datetime.combine(day, dt.time(9, 30), ET).astimezone(dt.timezone.utc)
    end = dt.datetime.combine(day, dt.time(16, 0), ET).astimezone(dt.timezone.utc)
    out, token = [], None
    while True:
        params = {"symbols": ticker, "timeframe": "1Min", "start": start.isoformat().replace("+00:00", "Z"),
                  "end": end.isoformat().replace("+00:00", "Z"), "limit": 10000, "adjustment": "raw", "feed": "iex"}
        if token:
            params["page_token"] = token
        js = alpaca("/v2/stocks/bars", params)
        out += (js.get("bars") or {}).get(ticker) or []
        token = js.get("next_page_token")
        if not token:
            return out


def replay(ticker: str, threshold: float, since: dt.datetime):
    today = dt.datetime.now(ET).date()
    first_day = since.astimezone(ET).date()
    days = daily_bars(ticker, first_day - dt.timedelta(days=45), today)
    hits = []
    for i, bar in enumerate(days):
        day = dt.datetime.fromisoformat(bar["t"].replace("Z", "+00:00")).astimezone(ET).date()
        if day < first_day:
            continue
        prior = [float(b.get("v") or 0) for b in days[max(0, i - 20):i]]
        avg = sum(prior) / len(prior) if prior else 0
        hi = float(bar.get("h") or 0)
        info = {"day": str(day), "high": hi, "close": float(bar.get("c") or 0),
                "day_vol_x_avg": round(float(bar.get("v") or 0) / avg, 2) if avg else None}
        if hi < threshold or not avg:
            info["result"] = "never reached trigger" if hi < threshold else "no avg volume"
            hits.append(info)
            continue
        cum, first_cross, first_fire = 0.0, None, None
        for m in minute_bars(ticker, day):
            t = dt.datetime.fromisoformat(m["t"].replace("Z", "+00:00")).astimezone(ET)
            cum += float(m.get("v") or 0)
            if t < dt.datetime.combine(day, dt.time(9, 30), ET) or float(m.get("c") or 0) < threshold:
                continue
            if t >= dt.datetime.combine(day, dt.time(16, 0), ET):
                break
            if first_cross is None:
                first_cross = t
            minutes = (t.hour * 60 + t.minute + 1) - (9 * 60 + 30)
            if minutes < 5:
                continue
            pace = cum / (avg * minutes / 390.0)
            if pace >= PACE and (since.astimezone(ET) <= t):
                first_fire = (t, round(pace, 2), float(m["c"]))
                break
        info["first_close_at_trigger"] = first_cross.strftime("%H:%M") if first_cross else None
        info["result"] = (f"SHOULD FIRE {first_fire[0]:%H:%M} ET at {first_fire[2]:.2f}, pace {first_fire[1]}x"
                          if first_fire else "reached trigger but volume pace < 1.2x (or only before alert created)")
        hits.append(info)
    return hits


def main() -> None:
    import psycopg2

    conn = psycopg2.connect(os.environ["DATABASE_URL"], connect_timeout=15)
    cur = conn.cursor()

    print("== realtime worker activity (events by day/type since 2026-11-17) ==")
    cur.execute("""
        SELECT (e.fired_at AT TIME ZONE 'America/New_York')::date d, COALESCE(a.alert_type,'?'), COUNT(*)
        FROM alert_events e LEFT JOIN user_alerts a ON a.id = e.alert_id
        WHERE e.fired_at >= '2026-11-17' GROUP BY 1,2 ORDER BY 1,2""")
    for row in cur.fetchall():
        print("  ", row[0], row[1], row[2])

    print("\n== alert counts by type (enabled / total) ==")
    cur.execute("SELECT alert_type, COUNT(*) FILTER (WHERE enabled), COUNT(*) FROM user_alerts GROUP BY 1 ORDER BY 1")
    for row in cur.fetchall():
        print("  ", row)

    print("\n== trigger alerts ==")
    cur.execute("""
        SELECT a.id, a.user_id, a.ticker, a.threshold, a.enabled, a.created_at, a.last_fired_at,
               COALESCE(u.tier,'basic'), COALESCE(u.email_verified, FALSE)
        FROM user_alerts a LEFT JOIN users u ON u.username = a.user_id
        WHERE a.alert_type = 'trigger' ORDER BY a.created_at""")
    alerts = cur.fetchall()
    if not alerts:
        print("   none: nobody has created a trigger alert yet")
    for a in alerts:
        print(f"   id={a[0]} user={mask(a[1])} {a[2]} trigger={a[3]} enabled={a[4]} created={a[5]:%Y-%m-%d %H:%M}Z "
              f"last_fired={a[6]} tier={a[7]} verified={a[8]}")

    print("\n== trigger events fired ==")
    cur.execute("""
        SELECT e.fired_at, e.user_id, e.ticker, e.message FROM alert_events e
        LEFT JOIN user_alerts a ON a.id = e.alert_id
        WHERE a.alert_type = 'trigger' OR e.message LIKE 'Trigger alert:%%' ORDER BY e.fired_at""")
    events = cur.fetchall()
    if not events:
        print("   none")
    for e in events:
        print(f"   {e[0]:%Y-%m-%d %H:%M}Z user={mask(e[1])} {e[2]} | {e[3]}")
    conn.close()

    print("\n== replay against IEX minute bars ==")
    for a in alerts:
        try:
            for h in replay(str(a[2]).upper(), float(a[3]), a[5]):
                print(f"   id={a[0]} {a[2]} >= {a[3]}:", h)
        except Exception as exc:
            print(f"   id={a[0]} {a[2]}: replay failed {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
