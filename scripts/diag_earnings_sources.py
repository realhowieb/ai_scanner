"""Read-only probe: which free sources return next earnings dates for names the
bulk calendar misses (foreign filers, renamed tickers). Prints dates only."""
import datetime as dt
import json
import urllib.request

NAMES = ["RACE", "NVO", "ENB", "MODG", "CALY", "TD", "RY", "ASML", "SAP", "TM", "AZN", "SHOP", "NVDA"]
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
      "Accept": "application/json, text/plain, */*"}


def yf_probe():
    print("== yfinance per symbol ==")
    try:
        import yfinance as yf
    except Exception as e:
        print("yfinance import failed", e)
        return
    for s in NAMES:
        try:
            t = yf.Ticker(s)
            cal = t.calendar
            ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
            print(s, "calendar:", ed)
        except Exception as e:
            print(s, "ERR", type(e).__name__, str(e)[:120])


def nasdaq_probe():
    print("== nasdaq.com calendar by day (next 45 days) ==")
    want = set(NAMES)
    found = {}
    day = dt.date.today()
    ok_days = 0
    for _ in range(45):
        day += dt.timedelta(days=1)
        if day.weekday() >= 5:
            continue
        url = f"https://api.nasdaq.com/api/calendar/earnings?date={day.isoformat()}"
        try:
            req = urllib.request.Request(url, headers=UA)
            data = json.loads(urllib.request.urlopen(req, timeout=20).read())
            rows = ((data.get("data") or {}).get("rows")) or []
            ok_days += 1
            for r in rows:
                sym = str(r.get("symbol") or "").upper()
                if sym in want and sym not in found:
                    found[sym] = (day.isoformat(), r.get("time"))
        except Exception as e:
            print(day, "ERR", type(e).__name__, str(e)[:120])
            break
    print("days ok:", ok_days, "found:", found)


if __name__ == "__main__":
    yf_probe()
    nasdaq_probe()
