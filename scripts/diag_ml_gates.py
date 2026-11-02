"""Read-only: per-day breakdown of the three failing ML v4 readiness gates.

For each ET date: opportunity rows, signal type, scan-join status, source_scan_id
stamped, AI Confidence frozen, model version frozen; plus matured signal-days per
entry day (max_entry_day_share). Aggregates only (no tickers). Never writes.
"""
from __future__ import annotations

import datetime as dt
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from analytics import market_calendar as mc
    from analytics import ml_readiness as mr
    from analytics import ml_v3_audit as mv3
    from analytics import research_dataset as rd
    from scripts.ml_readiness import load

    now = dt.datetime.now(dt.timezone.utc)
    data = load(now)
    rows, scans = data["rows"], data["scans"]
    print(f"rows={len(rows)} scans={len(scans)}")
    by_symbol = rd.index_scan_records(scans)

    day = defaultdict(Counter)
    for r in rows:
        ts = rd.to_dt(r["fired_at"])
        d = ts.astimezone(mc.ET).date().isoformat()
        c = day[d]
        c["rows"] += 1
        c["type:" + str(r.get("signal_type"))] += 1
        sid = rd.source_scan_id_of(r)
        c["has_sid"] += int(bool(sid))
        c["has_conf"] += int(r.get("ai_confidence") is not None)
        c["has_ver"] += int(bool(mr.model_version_of(r)))
        _s, info = rd.match_scan_observation(r["ticker"], r["fired_at"], by_symbol, source_scan_id=sid)
        c["join:" + info["status"]] += 1
        if info["status"] != rd.JOIN_MATCHED and sid:
            c["sid_but_unmatched"] += 1
        c["hour_utc:%02d" % ts.hour] += 1

    print("\nPER ET DATE")
    for d in sorted(day):
        c = day[d]
        n = c["rows"]
        joins = {k[5:]: v for k, v in c.items() if k.startswith("join:")}
        types = {k[5:]: v for k, v in c.items() if k.startswith("type:")}
        hours = sorted(k[9:] for k in c if k.startswith("hour_utc:"))
        print(f"{d} rows={n} join={c['join:MATCHED']/n:.0%} conf={c['has_conf']/n:.0%} "
              f"ver={c['has_ver']/n:.0%} sid={c['has_sid']/n:.0%} sid_unmatched={c['sid_but_unmatched']} "
              f"types={types} joins={joins} hours_utc={','.join(hours)}")

    records = rd.build_records(rows, scans)
    sd = mv3.build_audit_rows(records, rows, unit="signal_day")
    mat = [r for r in sd if r.matured(mr.PRIMARY_HORIZON) and r.certified]
    per_entry = Counter(r.entry_day for r in mat)
    per_obs = Counter(r.observed_day for r in mat)
    print(f"\nMATURED signal-days={len(mat)} entry_days={len(per_entry)}")
    for d, k in sorted(per_entry.items()):
        print(f"entry {d}: {k} ({k/len(mat):.1%})")
    print("\nTOP observed days among matured:", per_obs.most_common(5))
    sd_per_day = Counter(r.observed_day for r in sd)
    print("\nALL signal-days per observed day (incl. immature):")
    for d, k in sorted(sd_per_day.items()):
        print(f"obs {d}: {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
