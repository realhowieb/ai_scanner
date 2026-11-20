"""Read-only: how late does HSF pick up a stock relative to its move?

For every scheduled full-market scan of the last ~15 sessions, a "pickup" is a ticker in
that scan that was not in the scan just before it (the same rule the Scanner's "Picked up"
column uses). For each pickup this prints, with Alpaca bars:

  * already_day  - % the stock was up vs the previous session's close at the pickup price
  * already_5d   - % up vs the close 5 sessions earlier
  * share_done   - for intraday pickups, (pickup - prev close) / (day high - prev close):
                   the part of the day's whole up-move that had already happened
  * lag_3pct     - for intraday pickups up 3%+, minutes between the first 5-minute bar
                   that traded 3% above the previous close and the scan that picked it up,
                   and how much cheaper that first +3% price was
  * after        - best high and close over the next 5 sessions vs the pickup price

Prints tickers and public market numbers only.
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


def med(xs):
    xs = [x for x in xs if x is not None]
    return f"{stats.median(xs):+.1f}" if xs else "  -  "


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
    runs = market_runs(list_runs(limit=400, include_snapshots=True, username=MARKET_USER) or [])
    days = []
    for r in runs:
        d = r["created_at"].astimezone(ET).date()
        if d not in days:
            days.append(d)
    keep_days = set(days[:SESSIONS + 1])
    runs = [r for r in runs if r["created_at"].astimezone(ET).date() in keep_days]
    runs = list(reversed(runs))  # oldest first
    print(f"scans={len(runs)} sessions={len(keep_days)} from {min(keep_days)} to {max(keep_days)}")

    frames = {}
    for r in runs:
        try:
            df = _run_df_uncached(int(r["id"]))
        except Exception as e:  # noqa: BLE001
            print(f"  run {r['id']} load failed {type(e).__name__}")
            continue
        if df is None or df.empty:
            continue
        price_col = next((c for c in ("Last", "last", "Close") if c in df.columns), None)
        chg_col = next((c for c in ("PctChange", "chg_pct") if c in df.columns), None)
        rows = {}
        for i, t in enumerate(tickers_of(df)):
            sub = df[df[("Ticker" if "Ticker" in df.columns else "Symbol")].astype(str).str.upper() == t].iloc[0]
            rows[t] = {"rank": i + 1, "last": float(sub[price_col]) if price_col and sub[price_col] == sub[price_col] else None,
                       "chg": float(sub[chg_col]) if chg_col and sub[chg_col] == sub[chg_col] else None}
        frames[int(r["id"])] = (r, rows)

    ordered = [frames[k] for k in sorted(frames, key=lambda k: frames[k][0]["created_at"])]
    pickups = []
    for i in range(1, len(ordered)):
        run, rows = ordered[i]
        prev = ordered[i - 1][1]
        for t, row in rows.items():
            if t not in prev and row["last"]:
                pickups.append({"t": t, "at": run["created_at"], "name": str(run.get("name") or ""), **row})
    print(f"pickups={len(pickups)} (tickers new vs the scan before)")
    print(f"avg scan size={stats.mean(len(r) for _, r in ordered):.0f} rows")
    by_slot = defaultdict(int)
    for _, rows in ordered:
        pass
    for r, _ in ordered:
        by_slot[r["created_at"].astimezone(ET).strftime("%H:%M")[:4] + "x"] += 1
    print("scan save times (ET, 10-min bucket):", dict(sorted(by_slot.items())))

    syms = sorted({p["t"] for p in pickups})
    start = (min(keep_days) - dt.timedelta(days=12)).isoformat()
    daily = {}
    for i in range(0, len(syms), 200):
        try:
            daily.update(fetch_minute_bars_multi(syms[i:i + 200], start, timeframe="1Day"))
        except Exception as e:  # noqa: BLE001
            print(f"daily fetch failed {type(e).__name__}: {e}")
    sessions_of = {}
    for s, bars in daily.items():
        sessions_of[s] = [(ts(b["t"]).astimezone(ET).date(), b) for b in bars]

    intraday_need = defaultdict(set)
    for p in pickups:
        et = p["at"].astimezone(ET)
        if OPEN <= et.time() < CLOSE:
            intraday_need[et.date()].add(p["t"])
    intra = {}
    for d, tick in intraday_need.items():
        s0 = dt.datetime.combine(d, OPEN, ET).astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        s1 = dt.datetime.combine(d, CLOSE, ET).astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        tl = sorted(tick)
        for i in range(0, len(tl), 100):
            try:
                got = fetch_minute_bars_multi(tl[i:i + 100], s0, s1, timeframe="5Min")
            except Exception as e:  # noqa: BLE001
                print(f"intraday fetch {d} failed {type(e).__name__}")
                continue
            for s, bars in got.items():
                intra[(d, s)] = bars

    out = []
    for p in pickups:
        ses = sessions_of.get(p["t"]) or []
        if not ses:
            continue
        et = p["at"].astimezone(ET)
        intraday = OPEN <= et.time() < CLOSE
        # reference session: the last session that had opened by the scan
        idx = None
        for j, (d, _) in enumerate(ses):
            if d < et.date() or (d == et.date() and et.time() >= OPEN):
                idx = j
        if idx is None or idx < 1:
            continue
        d_ref, bar_ref = ses[idx]
        prev_close = ses[idx - 1][1]["c"]
        close_5ago = ses[idx - 5][1]["c"] if idx >= 5 else None
        P = p["last"]
        rec = {**p, "slot": et.strftime("%H:%M"), "intraday": intraday,
               "already_day": pct(P, prev_close), "already_5d": pct(P, close_5ago),
               "gap": pct(bar_ref["o"], prev_close)}
        fwd = ses[idx + 1: idx + 6]
        rest_high = None
        if intraday:
            bars = intra.get((d_ref, p["t"])) or []
            before = [b for b in bars if ts(b["t"]) < p["at"]]
            after = [b for b in bars if ts(b["t"]) >= p["at"]]
            rest_high = max((b["h"] for b in after), default=None)
            day_high = bar_ref["h"]
            if day_high and prev_close and day_high > prev_close:
                rec["share_done"] = max(0.0, min(1.5, (P - prev_close) / (day_high - prev_close)))
            first3 = next((b for b in before if b["h"] >= prev_close * 1.03), None)
            if first3 is not None and rec["already_day"] is not None and rec["already_day"] >= 3:
                rec["lag_min"] = (p["at"] - ts(first3["t"])).total_seconds() / 60.0
                rec["cheaper_at_3pct"] = pct(prev_close * 1.03, P)
            # what a 30-min cadence would have seen: price at the half-hour before the scan
            half = [b for b in before if ts(b["t"]) <= p["at"] - dt.timedelta(minutes=30)]
            if half and rec["already_day"] is not None and rec["already_day"] >= 3:
                rec["px_30m_earlier"] = pct(half[-1]["c"], P)
        else:
            if fwd:
                rec["next_gap"] = pct(fwd[0][1]["o"], P)
        highs = [b["h"] for _, b in fwd] + ([rest_high] if rest_high else [])
        rec["best_after"] = pct(max(highs), P) if highs else None
        rec["close5"] = pct(fwd[-1][1]["c"], P) if len(fwd) >= 5 else None
        rec["complete"] = len(fwd) >= 5
        out.append(rec)

    def block(title, rows):
        if not rows:
            print(f"\n{title}: none")
            return
        print(f"\n{title}: n={len(rows)}")
        print(f"  median already up today {med([r['already_day'] for r in rows])}%  | 5 sessions {med([r['already_5d'] for r in rows])}%"
              f"  | open gap {med([r['gap'] for r in rows])}%")
        sd = [r["share_done"] * 100 for r in rows if r.get("share_done") is not None]
        if sd:
            print(f"  median share of the day's up-move already done at pickup {stats.median(sd):.0f}% (n={len(sd)});"
                  f" >=75% done in {sum(1 for x in sd if x >= 75) / len(sd) * 100:.0f}%")
        lag = [r["lag_min"] for r in rows if r.get("lag_min") is not None]
        if lag:
            print(f"  up 3%+ at pickup: first traded +3% a median {stats.median(lag):.0f} min before the scan"
                  f" (p25 {sorted(lag)[len(lag) // 4]:.0f}, p75 {sorted(lag)[3 * len(lag) // 4]:.0f}, n={len(lag)});"
                  f" +3% price was a median {med([r.get('cheaper_at_3pct') for r in rows])}% vs pickup price")
            print(f"  price 30 min before the scan vs pickup price: median {med([r.get('px_30m_earlier') for r in rows])}%")
        comp = [r for r in rows if r.get("complete")]
        print(f"  after pickup (5 sessions, n={len(comp)}): best high median {med([r['best_after'] for r in comp])}%"
              f"  close median {med([r['close5'] for r in comp])}%  close>pickup {sum(1 for r in comp if (r['close5'] or 0) > 0)}/{len(comp)}")
        if any(r.get("next_gap") is not None for r in rows):
            print(f"  next session open vs pickup median {med([r.get('next_gap') for r in rows])}%")

    block("ALL pickups", out)
    block("Intraday pickups (scan saved 9:30-16:00 ET)", [r for r in out if r["intraday"]])
    block("Premarket pickups (before 9:30 ET)", [r for r in out if r["slot"] < "09:30"])
    block("After-close pickups (16:00+ ET)", [r for r in out if r["slot"] >= "16:00"])
    up = [r for r in out if (r["already_day"] or 0) >= 3]
    block("Pickups already up 3%+ on the day", up)

    print("\nby scan hour (ET): n | already today | 5d | share done | best after | close5")
    hours = defaultdict(list)
    for r in out:
        hours[r["slot"][:2]].append(r)
    for h in sorted(hours):
        rs = hours[h]
        sd = [x["share_done"] * 100 for x in rs if x.get("share_done") is not None]
        comp = [x for x in rs if x.get("complete")]
        print(f"  {h}:xx n={len(rs):>4} {med([x['already_day'] for x in rs]):>6}% {med([x['already_5d'] for x in rs]):>6}%"
              f" {(f'{stats.median(sd):.0f}%' if sd else '-'):>5} {med([x['best_after'] for x in comp]):>6}% {med([x['close5'] for x in comp]):>6}%")

    print("\nby how far it had already run today: n | best after | close5 | close>pickup")
    buckets = [(-1e9, 0, "down"), (0, 3, "0-3%"), (3, 7, "3-7%"), (7, 15, "7-15%"), (15, 1e9, "15%+")]
    for lo, hi, name in buckets:
        rs = [r for r in out if r.get("complete") and r["already_day"] is not None and lo <= r["already_day"] < hi]
        if rs:
            print(f"  {name:>6} n={len(rs):>4} {med([r['best_after'] for r in rs]):>6}% {med([r['close5'] for r in rs]):>6}%"
                  f"  {sum(1 for r in rs if (r['close5'] or 0) > 0) / len(rs) * 100:.0f}%")
    print("\nby 5-session run-up before pickup: n | best after | close5 | close>pickup")
    for lo, hi, name in [(-1e9, 0, "down"), (0, 10, "0-10%"), (10, 25, "10-25%"), (25, 1e9, "25%+")]:
        rs = [r for r in out if r.get("complete") and r["already_5d"] is not None and lo <= r["already_5d"] < hi]
        if rs:
            print(f"  {name:>6} n={len(rs):>4} {med([r['best_after'] for r in rs]):>6}% {med([r['close5'] for r in rs]):>6}%"
                  f"  {sum(1 for r in rs if (r['close5'] or 0) > 0) / len(rs) * 100:.0f}%")

    print("\nexamples: biggest intraday movers at pickup")
    ex = sorted([r for r in out if r["intraday"] and r.get("lag_min") is not None], key=lambda r: -(r["already_day"] or 0))[:15]
    for r in ex:
        print(f"  {r['at'].astimezone(ET):%m-%d %H:%M} {r['t']:<6} #{r['rank']:<3} up {r['already_day']:+.1f}% "
              f"(5d {r['already_5d'] if r['already_5d'] is None else round(r['already_5d'], 1)}%) first +3% {r['lag_min']:.0f} min earlier "
              f"share {r.get('share_done', 0) * 100:.0f}% | after best {r['best_after'] if r['best_after'] is None else round(r['best_after'], 1)}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
