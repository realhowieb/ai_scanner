"""Read-only: simulate an intraday "early movers watch" against what the scans actually picked.

Same sessions as diag_pickup_lateness.py (the last ~15 sessions of saved market scans).
The watch looks at 5-minute Alpaca bars for the same liquid universe the scan uses
(price >= $1, 20-day average dollar volume >= $5M, both from the scan's own data feed) and
flags each ticker at most once per day, at the first completed 5-minute bar where:

  T1 "+3% on volume": close >= previous close +3% AND today's volume so far is at least
     2x the 20-day average daily volume times the share of the session elapsed.
  T2 "breaks yesterday's high": close > yesterday's high AND close >= previous close +1%
     AND volume pace >= 1.5x as above.

Rules were fixed before the run; nothing is tuned to these results. First bar counted is
9:35-9:40 (so the earliest flag is 9:40 ET). Entry = that bar's close, at the bar's end.

Prints:
  A. Head-to-head on the scan's own pickups: share the watch flagged earlier the same day,
     minutes earlier and % cheaper, and 5-session outcomes from each entry.
  B. Everything the watch flagged vs everything the scans picked: flags per day, same-day
     and 5-session outcomes.
Ticker-days that look like splits (open or any forward day >40% from the prior close) are
dropped. Prints tickers and public market numbers only.
"""
from __future__ import annotations

import datetime as dt
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SESSIONS = 15
OPEN = dt.time(9, 30)
CLOSE = dt.time(16, 0)
MIN_PRICE = 1.0
MIN_DOLLAR_VOL = 5_000_000.0


def med(xs):
    xs = [x for x in xs if x is not None]
    return f"{stats.median(xs):+.1f}%" if xs else "-"


def share_pos(xs):
    xs = [x for x in xs if x is not None]
    return f"{sum(1 for x in xs if x > 0) / len(xs) * 100:.0f}%" if xs else "-"


def pct(a, b):
    try:
        return (float(a) / float(b) - 1.0) * 100.0 if a and b else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def ts(s):
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def main() -> int:
    from analytics.market_calendar import ET
    from data.price_alpaca import fetch_minute_bars_multi, get_alpaca_data_feed
    from db.runs import list_runs
    from ui.market_scans import MARKET_USER, _run_df_uncached, market_runs, tickers_of

    print(f"alpaca feed={get_alpaca_data_feed()}")

    # ---- scan pickups (same rule as the Scanner's "Picked up") ----
    runs = market_runs(list_runs(limit=400, include_snapshots=True, username=MARKET_USER) or [])
    days = []
    for r in runs:
        d = r["created_at"].astimezone(ET).date()
        if d not in days:
            days.append(d)
    keep = sorted(days[:SESSIONS + 1])
    runs = sorted([r for r in runs if r["created_at"].astimezone(ET).date() in set(keep)], key=lambda r: r["created_at"])
    scans = []
    for r in runs:
        try:
            df = _run_df_uncached(int(r["id"]))
        except Exception:  # noqa: BLE001
            continue
        if df is None or df.empty:
            continue
        col = "Ticker" if "Ticker" in df.columns else "Symbol"
        pc = next((c for c in ("Last", "last", "Close") if c in df.columns), None)
        px = {}
        for _, row in df.iterrows():
            v = row.get(pc) if pc else None
            px[str(row[col]).upper()] = float(v) if v is not None and v == v else None
        scans.append((r["created_at"], [t for t in tickers_of(df)], px))
    pickups = []
    for i in range(1, len(scans)):
        at, ticks, px = scans[i]
        prev = set(scans[i - 1][1])
        et = at.astimezone(ET)
        if not (OPEN <= et.time() < CLOSE):
            continue
        for t in ticks:
            if t not in prev and px.get(t):
                pickups.append({"t": t, "at": at, "price": px[t], "day": et.date()})
    print(f"scans={len(scans)} sessions={len(keep)} ({keep[0]}..{keep[-1]}) intraday pickups={len(pickups)}")

    # ---- universe + daily bars ----
    syms = []
    try:
        from data.us_market_universe import build_us_market_universe

        res = build_us_market_universe()
        syms = [str(s).upper() for s in res.get("symbols") or []]
        print(f"universe source={res.get('source')} symbols={len(syms)}")
    except Exception as e:  # noqa: BLE001
        print(f"universe failed {type(e).__name__}")
    if not syms:
        root = Path(__file__).resolve().parents[1]
        for f in ("nasdaq.txt", "other_listed.txt"):
            syms += [ln.strip().upper() for ln in (root / f).read_text().splitlines() if ln.strip()]
        print(f"universe fallback symbols={len(syms)}")
    syms = sorted(set(syms) | {p["t"] for p in pickups})

    start = (keep[0] - dt.timedelta(days=40)).isoformat()
    daily = {}
    for i in range(0, len(syms), 500):
        try:
            got = fetch_minute_bars_multi(syms[i:i + 500], start, timeframe="1Day")
        except Exception as e:  # noqa: BLE001
            print(f"daily chunk {i} failed {type(e).__name__}")
            continue
        for s, bars in got.items():
            daily[s] = [(ts(b["t"]).astimezone(ET).date(), float(b["o"]), float(b["h"]), float(b["l"]), float(b["c"]), float(b["v"])) for b in bars]
    print(f"daily bars for {len(daily)} symbols")

    def ctx(sym, day):
        """(prev_close, prev_high, avg_vol20, avg_dvol20, today_open, fwd list) or None."""
        bars = daily.get(sym) or []
        idx = next((j for j, b in enumerate(bars) if b[0] == day), None)
        if idx is None or idx < 21:
            return None
        hist = bars[idx - 20:idx]
        prev = bars[idx - 1]
        today = bars[idx]
        if abs(today[1] / prev[4] - 1) > 0.4:
            return None
        fwd = bars[idx + 1: idx + 6]
        chain = [today] + fwd
        for a, b in zip(chain, chain[1:]):
            if abs(b[1] / a[4] - 1) > 0.4:
                return None
        return {"pc": prev[4], "ph": prev[2], "av": stats.mean(b[5] for b in hist),
                "adv": stats.mean(b[4] * b[5] for b in hist), "close": today[4], "fwd": fwd}

    # ---- simulate per day ----
    flags = {"T1": [], "T2": []}
    first_flag = {}  # (day, sym) -> {rule: (time, price)}
    pickup_keys = defaultdict(list)
    for p in pickups:
        pickup_keys[p["day"]].append(p)
    for day in keep:
        if day not in pickup_keys and day != keep[-1]:
            pass
        elig = []
        for s in syms:
            c = ctx(s, day)
            if c and c["pc"] >= MIN_PRICE and c["adv"] >= MIN_DOLLAR_VOL:
                elig.append(s)
        need = sorted(set(elig) | {p["t"] for p in pickup_keys.get(day, [])})
        s0 = dt.datetime.combine(day, OPEN, ET).astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        s1 = dt.datetime.combine(day, CLOSE, ET).astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        intra = {}
        for i in range(0, len(need), 200):
            try:
                intra.update(fetch_minute_bars_multi(need[i:i + 200], s0, s1, timeframe="5Min"))
            except Exception as e:  # noqa: BLE001
                print(f"  {day} intraday chunk failed {type(e).__name__}")
        n_day = {"T1": 0, "T2": 0}
        for s in need:
            c = ctx(s, day)
            bars = intra.get(s) or []
            if not c or not bars:
                continue
            cum = 0.0
            hits = {}
            seq = []
            for b in bars:
                t0 = ts(b["t"])
                cum += float(b["v"] or 0)
                end = t0 + dt.timedelta(minutes=5)
                seq.append((end, float(b["c"]), float(b["h"])))
                if t0.astimezone(ET).time() < dt.time(9, 35):
                    continue
                elapsed = (end - dt.datetime.combine(day, OPEN, ET)).total_seconds() / 60.0
                pace = cum / max(1.0, c["av"] * elapsed / 390.0)
                close = float(b["c"])
                if "T1" not in hits and close >= c["pc"] * 1.03 and pace >= 2.0:
                    hits["T1"] = (end, close)
                if "T2" not in hits and close > c["ph"] and close >= c["pc"] * 1.01 and pace >= 1.5:
                    hits["T2"] = (end, close)
            first_flag[(day, s)] = (hits, seq, c)
            if s in elig:
                for rule, (end, close) in hits.items():
                    n_day[rule] += 1
                    flags[rule].append((day, s, end, close))
        print(f"  {day}: eligible={len(elig)} flags T1={n_day['T1']} T2={n_day['T2']} pickups={len(pickup_keys.get(day, []))}")

    def outcome(day, sym, at, price):
        ff = first_flag.get((day, sym))
        if not ff:
            return None
        _, seq, c = ff
        after = [h for (end, _, h) in seq if end > at]
        o = {"day_close": pct(c["close"], price),
             "day_best": pct(max(after), price) if after else None}
        if len(c["fwd"]) >= 5:
            o["c5"] = pct(c["fwd"][-1][4], price)
            o["best5"] = pct(max([b[2] for b in c["fwd"]] + after), price)
        return o

    # ---- A. head-to-head on the scan's pickups ----
    print("\nA. Same stocks the scans picked up (intraday pickups)")
    for rule in ("T1", "T2", "EITHER"):
        rows = []
        for p in pickups:
            ff = first_flag.get((p["day"], p["t"]))
            if not ff:
                continue
            hits = ff[0]
            cand = [hits[r] for r in (("T1", "T2") if rule == "EITHER" else (rule,)) if r in hits]
            hit = min(cand, key=lambda x: x[0]) if cand else None
            rows.append((p, hit))
        valid = [(p, h) for p, h in rows if outcome(p["day"], p["t"], p["at"], p["price"])]
        earlier = [(p, h) for p, h in valid if h and h[0] < p["at"]]
        mins = [(p["at"] - h[0]).total_seconds() / 60 for p, h in earlier]
        cheaper = [pct(h[1], p["price"]) for p, h in earlier]
        print(f"  {rule}: flagged earlier on {len(earlier)}/{len(valid)} pickups ({len(earlier) / max(1, len(valid)) * 100:.0f}%);"
              f" median {stats.median(mins) if mins else 0:.0f} min earlier; watch price vs pickup price {med(cheaper)}")
        o_pick = [outcome(p["day"], p["t"], p["at"], p["price"]) for p, _ in earlier]
        o_watch = [outcome(p["day"], p["t"], h[0], h[1]) for p, h in earlier]
        print(f"     those stocks, 5-session close: from watch entry {med([o.get('c5') for o in o_watch])}"
              f" ({share_pos([o.get('c5') for o in o_watch])} up) vs from scan pickup {med([o.get('c5') for o in o_pick])}"
              f" ({share_pos([o.get('c5') for o in o_pick])} up); same-day close {med([o['day_close'] for o in o_watch])} vs {med([o['day_close'] for o in o_pick])}")

    # ---- B. everything flagged vs everything picked ----
    print("\nB. All watch flags vs all scan pickups (entry -> outcome)")
    print("  group          n   per day | same-day close (up) | best rest of day | 5-session close (up) | 5-session best")

    def line(name, outs, ndays):
        o = [x for x in outs if x]
        c5 = [x.get("c5") for x in o]
        print(f"  {name:<12} {len(o):>5} {len(o) / max(1, ndays):>7.0f} | {med([x['day_close'] for x in o]):>7} ({share_pos([x['day_close'] for x in o]):>4})"
              f"     | {med([x['day_best'] for x in o]):>7}          | {med(c5):>7} ({share_pos(c5):>4})     | {med([x.get('best5') for x in o]):>7}")

    nd = len(keep)
    line("scan pickups", [outcome(p["day"], p["t"], p["at"], p["price"]) for p in pickups], nd)
    for rule in ("T1", "T2"):
        line(f"watch {rule}", [outcome(d, s, at, px) for d, s, at, px in flags[rule]], nd)
        picked_same_day = {(p["day"], p["t"]) for p in pickups}
        line(f" {rule} never picked", [outcome(d, s, at, px) for d, s, at, px in flags[rule] if (d, s) not in picked_same_day], nd)
        early = [f for f in flags[rule] if f[2].astimezone(ET).time() <= dt.time(10, 30)]
        line(f" {rule} by 10:30", [outcome(d, s, at, px) for d, s, at, px in early], nd)

    print("\nflag time of day (ET hour): T1 / T2")
    for h in range(9, 16):
        a = sum(1 for f in flags["T1"] if f[2].astimezone(ET).hour == h)
        b = sum(1 for f in flags["T2"] if f[2].astimezone(ET).hour == h)
        print(f"  {h:02d}:xx {a:>5} / {b:>5}")

    print("\nexamples: pickups the watch caught much earlier (T1)")
    ex = []
    for p in pickups:
        ff = first_flag.get((p["day"], p["t"]))
        if ff and "T1" in ff[0] and ff[0]["T1"][0] < p["at"]:
            ex.append(((p["at"] - ff[0]["T1"][0]).total_seconds() / 60, p, ff[0]["T1"]))
    for mins, p, h in sorted(ex, key=lambda x: -x[0])[:12]:
        print(f"  {p['day']} {p['t']:<6} watch {h[0].astimezone(ET):%H:%M} ${h[1]:.2f} | scan {p['at'].astimezone(ET):%H:%M} ${p['price']:.2f}"
              f" ({pct(h[1], p['price']):+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
