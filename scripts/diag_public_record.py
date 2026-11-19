"""Read-only preview of the public track record, the market condition and the weekly
recap's "last week" block on production data (run through the Diagnostics workflow).

Prints aggregates and the newest picks only; the same numbers the public /results page
will show. Writes nothing.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def pct(v):
    return None if v is None else round(v * 100, 2)


def main() -> None:
    from api import public_record as pr
    from api.market_condition import condition_of, current, labels_by_day

    cond = current()
    print("market condition:", json.dumps({k: cond.get(k) for k in ("available", "label", "why", "as_of")}))
    labels = labels_by_day()
    print(f"condition history: {len(labels)} days; last 10:", json.dumps(dict(sorted(labels.items())[-10:])))

    rows = pr.load_rows()
    print(f"hsf_scanner rows loaded: {len(rows)}")
    out = pr.build(rows, condition_of(labels) if labels else None)
    s = out["summary"]
    print("summary:", json.dumps({**{k: s[k] for k in ("picks", "days", "pending", "unavailable", "first_day", "last_day")},
                                  **{k: pct(s[k]) for k in ("avg_return", "avg_spy", "avg_excess", "median_excess", "beat_rate", "up_rate")},
                                  "excess_ci95_pct": {k: pct(v) for k, v in (s["excess_ci95"] or {}).items()}}))
    print("verdict:", out["verdict"], "-", out["verdict_text"])
    for g in out["by_condition"]:
        print("  condition", g["label"], json.dumps({"picks": g["picks"], "days": g["days"], "avg_excess_pct": pct(g["avg_excess"]),
                                                      "beat_rate_pct": pct(g["beat_rate"])}))
    print("newest 10 picks:")
    for p in out["recent"][:10]:
        print("  ", p["day"], p["ticker"], p["score"], p["setup"], "ret", pct(p["return_5d"]), "spy", pct(p["spy_5d"]), p.get("condition"))

    today = dt.date.today()
    monday = today - dt.timedelta(days=today.weekday())
    for back in (1, 2, 3):
        m = monday - dt.timedelta(days=7 * back)
        w = pr.week_outcomes(m)
        print(f"week of {m}:", "none" if not w else json.dumps(
            {"picks": w["picks"], "pending": w["pending"], "avg_return_pct": pct(w["avg_return"]), "avg_spy_pct": pct(w["avg_spy"]),
             "beat_rate_pct": pct(w["beat_rate"]), "best": [x["ticker"] for x in w["best"]], "worst": [x["ticker"] for x in w["worst"]]}))


if __name__ == "__main__":
    main()
