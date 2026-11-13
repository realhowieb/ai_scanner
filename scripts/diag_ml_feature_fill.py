"""Read-only: weekly fill rate of the six core ML v4 inputs, Sep 14 to today.

Features and scan-join status only. Labels/outcomes are never printed or
aggregated. Stops before the reserved forward-test window (Dec 23).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORE = ["trend_10d_pct", "trend_20d_pct", "rvol_20", "dollar_vol_20", "scanner_breakout_score", "scan_gap_pct"]
RESERVED_START = dt.date(2026, 12, 23)


def main():
    from analytics import ml_v3_audit as A
    from analytics import research_dataset as rd
    from db.research_datasets import fetch_opportunity_rows, fetch_scan_records

    start = dt.date(2026, 9, 14)
    end = min(dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1), RESERVED_START)
    lo = dt.datetime(start.year, start.month, start.day, tzinfo=dt.timezone.utc)
    hi = dt.datetime(end.year, end.month, end.day, tzinfo=dt.timezone.utc)
    raw = fetch_opportunity_rows(lo, hi)
    scans = fetch_scan_records({r["ticker"] for r in raw}, lo - rd.MAX_SCAN_LAG - dt.timedelta(hours=1), hi)
    rows = A.build_audit_rows(rd.build_records(raw, scans), raw, unit="signal_day")

    def missing(v):
        x = A.Encoder._num(v)
        return not math.isfinite(x)

    buckets = {
        "train_sep14_oct21": lambda d: d < dt.date(2026, 10, 22),
        "oct22_nov01": lambda d: dt.date(2026, 10, 22) <= d < dt.date(2026, 11, 2),
        "since_nov02": lambda d: d >= dt.date(2026, 11, 2),
    }
    days = sorted({r.entry_day for r in rows})
    last20 = set(days[-20:])
    buckets["last_20_entry_days"] = lambda d: d in last20

    def summarize(group):
        out = {"rows": len(group), "missing_pct": {}, "join_status": dict(Counter(str(r.join_status) for r in group))}
        for f in CORE:
            n = sum(missing(r.features.get(f)) for r in group)
            out["missing_pct"][f] = round(100 * n / len(group), 2) if group else None
        out["all_six_present_pct"] = round(100 * sum(not any(missing(r.features.get(f)) for f in CORE) for r in group) / len(group), 2) if group else None
        return out

    result = {"outcomes_read": False, "window": [start.isoformat(), end.isoformat()], "signal_days": len(rows),
              "buckets": {k: summarize([r for r in rows if fn(r.entry_day)]) for k, fn in buckets.items()},
              "weekly": {}, "daily_since_oct22": {}}
    wk = defaultdict(list)
    for r in rows:
        y, w, _ = r.entry_day.isocalendar()
        wk[f"{y}-W{w:02}"].append(r)
    for k in sorted(wk):
        s = summarize(wk[k])
        result["weekly"][k] = {"rows": s["rows"], "all_six_present_pct": s["all_six_present_pct"],
                               "max_missing_pct": max(s["missing_pct"].values()), "join_status": s["join_status"]}
    for d in days:
        if d >= dt.date(2026, 10, 22):
            s = summarize([r for r in rows if r.entry_day == d])
            result["daily_since_oct22"][d.isoformat()] = {"rows": s["rows"], "all_six_present_pct": s["all_six_present_pct"],
                                                         "missing_pct": {k: v for k, v in s["missing_pct"].items() if v}}
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
