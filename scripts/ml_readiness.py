"""ML v4 data readiness report / monitor, run against production (read-only).

    workflow_dispatch Diagnostics -> script=ml_readiness.py      (full report)
    python scripts/ml_readiness.py --monitor --out DIR           (scheduled check)

Full report (default):
  1. Reads the whole opportunity dataset with the slim readiness projections
     (db.research_datasets.fetch_readiness_rows / fetch_scan_index), the same
     reads /v1/ml/readiness makes.
  2. Asks Alpaca (read-only) for SPY daily closes, so observations can be
     labelled by market regime from prior closes, and for the daily bars of
     tickers whose label came back empty, so "provider had no bars" can be told
     apart from "bars exist now" and the recoverable rows can be counted.
     Nothing is written back.
  3. Diagnoses why the backward scan join misses (how long after the
     observation its own scan record was written).
  4. Writes ml/reports/ml_readiness_report.{md,json} and prints the markdown,
     then a base64 gzip of the JSON between markers (only the log tail of a
     workflow run is readable from the sandbox).

--monitor skips the provider calls, writes ml_readiness_monitor.{md,json} to
--out, prints one ::warning:: line per alert and exits 0 (alerts never fail
the health run). --previous compares against an earlier monitor JSON.

Never trains, scores, writes to the database, or changes production behavior.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import gzip
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "ml" / "reports"
BUNDLE_BEGIN = "=== ML_READINESS_JSON_BEGIN ==="
BUNDLE_END = "=== ML_READINESS_JSON_END ==="


def _log(msg: str) -> None:
    print(f"[ml_readiness {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(now: dt.datetime) -> Dict[str, Any]:
    from analytics import ml_readiness as mr
    from analytics import research_dataset as rd
    from db.research_datasets import fetch_readiness_rows, fetch_scan_index

    s = mr.DATASET_START
    lo = dt.datetime(s.year, s.month, s.day, tzinfo=dt.timezone.utc)
    hi = now + dt.timedelta(minutes=1)
    for attempt in range(3):  # Neon can drop an idle SSL connection; retry the read
        try:
            rows = fetch_readiness_rows(lo, hi)
            scans = fetch_scan_index({r.get("ticker") for r in rows}, lo - rd.MAX_SCAN_LAG - dt.timedelta(hours=1), hi)
            return {"rows": rows, "scans": scans}
        except Exception as e:  # noqa: BLE001
            _log(f"read failed ({type(e).__name__}); retry {attempt + 1}/3")
            time.sleep(5 * (attempt + 1))
    raise SystemExit("database unavailable")


def spy_closes(days: int) -> Optional[Dict[dt.date, float]]:
    try:
        from data.price_alpaca import download_multi_alpaca

        bars = download_multi_alpaca(["SPY"], period=f"{days}d", interval="1d", prepost=False, timeout_s=25.0)
        df = (bars or {}).get("SPY")
        if df is None or df.empty:
            return None
        return {(ts.date() if hasattr(ts, "date") else ts): float(c) for ts, c in df["Close"].dropna().items()}
    except Exception as e:  # noqa: BLE001
        _log(f"SPY closes unavailable: {type(e).__name__}")
        return None


def price_probe(rows: List[Dict[str, Any]], now: dt.datetime) -> Dict[str, Any]:
    """For rows whose label came back empty (or never came), ask whether the
    canonical scorer would produce a label from today's bars. Returns
    {observation id: scorable now}. Read-only."""
    from analytics import ml_readiness as mr
    from analytics.signal_outcomes import complete_session_bars, score_signal

    targets = []
    for r in rows:
        st = mr.maturation_stage(r, now)
        if st["category"] in (mr.MISSING_PRICE_DATA, mr.PREMATURE_LABEL_WRITE, mr.MATURATION_JOB_MISSED):
            targets.append((r, st["category"]))
    if not targets:
        return {"probe": {}, "recoverable": {}, "checked": 0}
    oldest = min(r["fired_at"] for r, _ in targets)
    days = min(1500, (now - oldest).days + 30)
    tickers = sorted({str(r["ticker"]).upper() for r, _ in targets})
    try:
        from data.price_alpaca import download_multi_alpaca

        bars = download_multi_alpaca(tickers, period=f"{days}d", interval="1d", prepost=False, timeout_s=60.0) or {}
    except Exception as e:  # noqa: BLE001
        _log(f"price probe unavailable: {type(e).__name__}")
        return {"probe": None, "recoverable": {}, "checked": 0}
    probe, recoverable = {}, defaultdict(int)
    for r, cat in targets:
        b = bars.get(str(r["ticker"]).upper())
        b = complete_session_bars(b, now) if b is not None else None
        ok = b is not None and not b.empty and score_signal(b, r["fired_at"]) is not None
        probe[int(r["id"])] = ok
        if ok:
            recoverable[cat] += 1
    return {"probe": probe, "recoverable": dict(recoverable), "checked": len(targets), "tickers": len(tickers)}


def join_diagnosis(rows: List[Dict[str, Any]], scans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Why the backward scan join misses: for rows with ONLY_LATER_SCANS, how
    long after the observation was the nearest same-ticker scan written, and how
    far was its scan time from the observation."""
    from analytics import research_dataset as rd

    by_symbol = rd.index_scan_records(scans)
    delays, gaps, same_scan = [], [], 0
    statuses = defaultdict(int)
    for r in rows:
        _scan, info = rd.match_scan_observation(r["ticker"], r["fired_at"], by_symbol)
        statuses[info["status"]] += 1
        if info["status"] != rd.JOIN_FUTURE_ONLY:
            continue
        obs = rd.to_dt(r["fired_at"])
        cands = [c for c in by_symbol.get(str(r["ticker"]).upper(), ()) if c["known_at"] is not None]
        if not cands:
            continue
        near = min(cands, key=lambda c: abs((c["scan_timestamp"] - obs).total_seconds()))
        gap = (near["scan_timestamp"] - obs).total_seconds()
        gaps.append(gap)
        delays.append((near["known_at"] - obs).total_seconds())
        if abs(gap) <= 30 * 60 and near["scan_timestamp"] <= obs:
            same_scan += 1

    def q(v: List[float]) -> Optional[Dict[str, float]]:
        if not v:
            return None
        s = sorted(v)
        return {"min": s[0], "median": statistics.median(s), "p90": s[int(0.9 * (len(s) - 1))], "max": s[-1]}

    return {"join_status": dict(statuses), "only_later_scans": len(delays),
            "scan_started_before_observation_within_30m": same_scan,
            "scan_time_minus_observation_s": q(gaps), "scan_written_after_observation_s": q(delays)}


def main(argv: Optional[List[str]] = None) -> int:
    from analytics import ml_readiness as mr

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--monitor", action="store_true", help="scheduled check: no provider calls, alerts only")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--previous", default=None, help="earlier monitor JSON to compare against")
    args = ap.parse_args(argv)

    now = dt.datetime.now(dt.timezone.utc)
    _log("reading opportunity rows and the scan index (slim projections)")
    data = load(now)
    rows, scans = data["rows"], data["scans"]
    _log(f"rows={len(rows)} scan_index_rows={len(scans)}")
    extra: Dict[str, Any] = {}
    spy, probe = None, None
    if not args.monitor:
        spy = spy_closes(min(1500, (now.date() - mr.DATASET_START).days + 60))
        pr = price_probe(rows, now)
        probe = pr.get("probe")
        extra = {"price_probe": {k: v for k, v in pr.items() if k != "probe"},
                 "join_diagnosis": join_diagnosis(rows, scans)}
    report = mr.build_report(rows, scans, now=now, spy_closes=spy, price_probe=probe)
    report["diagnostics"] = extra
    previous = None
    if args.previous and Path(args.previous).exists():
        try:
            previous = json.loads(Path(args.previous).read_text())
        except (OSError, json.JSONDecodeError):
            previous = None
    alerts = mr.monitor(report, previous)
    report["alerts"] = alerts
    md = mr.render_markdown(report, alerts=alerts)
    if extra:
        md += "\n## Diagnostics\n\n```json\n" + json.dumps(extra, indent=1, default=str) + "\n```\n"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "ml_readiness_monitor" if args.monitor else "ml_readiness_report"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=1, default=str, sort_keys=True))
    (out / f"{stem}.md").write_text(md)
    print(md)
    for a in alerts:
        print(f"::warning title=ML readiness {a['code']}::{a['message']}")
    if not args.monitor:
        blob = base64.b64encode(gzip.compress(json.dumps(report, default=str, sort_keys=True).encode())).decode()
        print(BUNDLE_BEGIN)
        for i in range(0, len(blob), 200):
            print(blob[i:i + 200])
        print(BUNDLE_END)
    _log(f"status={report['status']} recommendation={report['recommendation']} alerts={len(alerts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
