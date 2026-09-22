#!/usr/bin/env python3
"""Run 38A — outcome maturation worker (idempotent, safe to rerun).

Finds captured observations whose outcome horizons have matured, fetches the
appropriate historical price bars, computes outcomes (no lookahead), and attaches
them via the Run 36 outcome schema. Skips already-matured (observation_id,
horizon) pairs (first-write-wins) and records failures.

NOT scheduled by production yet (WORKER READY BUT NOT SCHEDULED). To schedule:
add a daily GitHub Actions job (after market close) with the Alpaca secrets, e.g.
`python -m scripts.mature_observations --min-age-min 75`. It is read-only w.r.t.
scans and only writes to hsf_observation_outcomes.

    python -m scripts.mature_observations [--limit N] [--min-age-min M] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as _dt
from typing import Any, Dict, List

# Intraday horizons the scheduled cadence can support from minute bars.
HORIZON_BARS = {"+5m": 5, "+15m": 15, "+30m": 30, "+60m": 60}


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _to_dt(v: Any):
    try:
        d = v if isinstance(v, _dt.datetime) else _dt.datetime.fromisoformat(
            str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    except Exception:
        return None


def _mature_one(obs: Dict[str, Any], *, min_age_min: int, dry_run: bool) -> Dict[str, int]:
    """Mature a single observation's outcomes. Returns per-obs counters."""
    from analytics.observation_capture import compute_matured_outcomes

    res = {"attached": 0, "skipped": 0, "failed": 0, "pending": 0}
    anchor = _to_dt(obs.get("scan_timestamp") or obs.get("timestamp"))
    if anchor is None:
        return res
    age_min = (_now() - anchor).total_seconds() / 60.0
    if age_min < min_age_min:
        res["pending"] += 1
        return res
    existing = set((obs.get("outcomes") or {}).keys())
    want = [h for h in HORIZON_BARS if h not in existing]
    if not want:
        res["skipped"] += 1
        return res

    symbol = obs.get("symbol")
    start = anchor.date().isoformat()
    try:
        from data.price_alpaca import fetch_minute_bars
        bars = fetch_minute_bars(symbol, start, None) or []
    except Exception:
        res["failed"] += 1
        return res
    # bars at/after the anchor minute; prices_after[0] = signal price
    after = [b for b in bars if (_to_dt(b.get("t")) or anchor) >= anchor]
    closes = [b.get("c") for b in after if b.get("c") is not None]
    if len(closes) < 2:
        res["failed"] += 1
        return res

    eval_times = {h: (anchor + _dt.timedelta(minutes=n)).isoformat()
                  for h, n in HORIZON_BARS.items()}
    outcomes = compute_matured_outcomes(
        obs, prices_after=closes, horizon_bars={h: HORIZON_BARS[h] for h in want},
        evaluation_times=eval_times,
        direction=str((obs.get("scanners") or [{}])[0].get("direction") or "long"))
    if dry_run:
        res["attached"] += len(outcomes)
        return res
    try:
        from db.hsf_observations import save_outcome
        for oc in outcomes:
            if save_outcome(oc):
                res["attached"] += 1
            else:
                res["skipped"] += 1
    except Exception:
        res["failed"] += 1
    return res


def main() -> int:
    from db.hsf_observations import load_recent_observations

    ap = argparse.ArgumentParser(description="Mature captured observation outcomes")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--min-age-min", type=int, default=75)  # 60m horizon + slack
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    observations = load_recent_observations(limit=args.limit) or []
    totals = {"observations": len(observations), "attached": 0, "skipped": 0,
              "failed": 0, "pending": 0}
    for obs in observations:
        r = _mature_one(obs, min_age_min=args.min_age_min, dry_run=args.dry_run)
        for k, v in r.items():
            totals[k] += v
    print(f"[mature_observations] dry_run={args.dry_run} {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
