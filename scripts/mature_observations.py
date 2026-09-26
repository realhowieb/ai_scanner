#!/usr/bin/env python3
"""Run 38A/38B — outcome maturation worker (idempotent, safe to rerun).

Finds captured observations whose intraday horizons (+5m/+15m/+30m/+60m) have
matured, fetches the appropriate future minute bars ONCE per symbol, computes
outcomes (no lookahead), and attaches them via the Run 36 outcome schema. Skips
already-matured (observation_id, horizon) pairs (first-write-wins) and classifies
failures. Produces a machine-readable report artifact.

Safe to run repeatedly (Task 7): each horizon matures independently the moment
its wall-clock elapses; nothing is overwritten. It only writes to
hsf_observation_outcomes and never touches scans, observations, or scanners.

    python -m scripts.mature_observations [--limit N] [--slack-min M] [--dry-run] [--out DIR]

Scheduling: `.github/workflows/mature-observations.yml` (every 30 min during/after
US market hours, workflow_dispatch supported).

Run 51 (maturation hardening): bars are fetched with the multi-symbol Alpaca
endpoint in batches of `batch_size` symbols (one paginated request series per
batch, 10k bars/page shared across symbols) instead of one request per symbol,
through a retrying client (429 → Retry-After/backoff+jitter, bounded). Each
symbol's bars are retrieved once per run and reused by all its observations.
Persistent throttling is reported as RATE_LIMITED (retried next run), never as
PRICE_DATA_UNAVAILABLE, and stops further requests this run. Symbols the
US_MARKET rules exclude (preferred shares, malformed) never reach retrieval.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from analytics.observation_capture import (
    HORIZON_BARS,
    compute_matured_outcomes,
    horizon_eligibility,
)

ROOT = Path(__file__).resolve().parents[1]

# Per-run distinct-symbol cap. Raised 400 -> 2000 after dry-run 36224599421
# measured 4 requests / 400 symbols (0.01 req/symbol, 0 x 429, 55 s job) with
# batched retrieval; 2000 covers the full ready set (~1.6k) in one run.
MAX_SYMBOLS = 2000
# Symbols per multi-symbol request series (URL length stays well under limits).
BATCH_SIZE = 100
# Forward window after a batch's latest anchor. Bars are bar-indexed, so sparse
# symbols need calendar time to accrue 60 bars; 4 days spans a weekend + holiday,
# matching the legacy open-ended (start → now) fetch for all practical purposes.
FORWARD_WINDOW = _dt.timedelta(days=4)
# Retirement: once anchor + FORWARD_WINDOW + grace has passed, the bounded fetch
# window is complete, so a retry cannot change the result (sparse / no-data
# symbols would otherwise be retried forever). The 2-day grace gives every
# observation many attempts first. Non-destructive: nothing is written; pass
# retire_after=None (CLI --retire-after-days 0) to re-attempt for a backfill.
RETIRE_AFTER = FORWARD_WINDOW + _dt.timedelta(days=2)


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _parse(v: Any):
    try:
        d = v if isinstance(v, _dt.datetime) else _dt.datetime.fromisoformat(
            str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    except Exception:
        return None


def _new_report() -> Dict[str, Any]:
    return {
        "schema": "hsf-maturation-1.2",
        "generated_at": _now().isoformat(),
        "observations_scanned": 0,
        "eligible_observations": 0,
        "symbols_with_ready_horizons": 0,
        "symbols_deferred": 0,
        "horizons": {h: {"new": 0, "already": 0, "not_ready": 0, "failed": 0}
                     for h in HORIZON_BARS},
        "failures": {},
        "attached": 0,
        # Run 52 — backlog capacity telemetry (measurement only).
        "backlog": {
            "ready_symbols": 0,          # distinct symbols with >=1 ready horizon
            "ready_observations": 0,     # observations with >=1 ready horizon
            "processed_symbols": 0,      # symbols actually fetched this run (<= cap)
            "deferred_symbols": 0,       # ready symbols not reached (cap overflow)
            "matured_observations": 0,   # NEW outcome rows written this run
            "failed_symbols": 0,         # symbols that failed wholesale (no maturation)
            "oldest_pending_age_min": None,   # age of oldest READY anchor still pending
            "estimated_clearance_runs": None,  # ceil(ready_symbols / cap), snapshot
            "max_symbols": 0,
        },
        # Run 51 — market-data retrieval telemetry. Horizon-level counts
        # (price_data_failures / insufficient_future_bars) mirror `failures`;
        # symbol-level counts separate true missing data from throttling.
        "alpaca_requests": 0,          # HTTP attempts sent (incl. retries)
        "alpaca_429_count": 0,         # HTTP 429 responses received
        "alpaca_retry_count": 0,       # attempts re-sent after 429/5xx/timeout
        "unique_symbols_requested": 0,
        "cache_hits": 0,               # observations served from a symbol's already-retrieved bars
        "cache_misses": 0,             # symbol bar series retrieved from the provider
        "symbols_processed": 0,
        "outcomes_matured": 0,          # (symbols_deferred: see above)
        "price_data_failures": 0,      # horizons: provider answered, no bars (true missing)
        "insufficient_future_bars": 0,  # horizons: bars exist but too few after anchor
        "price_data_unavailable_symbols": 0,
        "rate_limited_symbols": 0,     # symbols skipped because of persistent 429 (retry next run)
        "provider_error_symbols": 0,
        "ineligible": {"symbols": 0, "observations": 0, "reasons": {}},
        # Ready horizons skipped because their bounded window closed long ago.
        "retired": {"observations": 0, "horizons": 0, "symbols": 0,
                    "retire_after_days": None},
        "fetch_mode": None,
    }


def _fail(report: Dict[str, Any], horizon: str | None, reason: str) -> None:
    report["failures"][reason] = report["failures"].get(reason, 0) + 1
    if horizon:
        report["horizons"][horizon]["failed"] += 1


def _default_exclusion_reason(symbol: str):
    from data.us_market_universe import symbol_exclusion_reason
    return symbol_exclusion_reason(symbol)


def _batch_windows(ordered, now, batch_size: int):
    """Yield (symbols, start_iso, end_iso) per batch. `ordered` is oldest-anchor
    first, so batch members share similar start times (little over-fetch)."""
    batch_size = max(1, int(batch_size))
    for i in range(0, len(ordered), batch_size):
        chunk = [(sym, plans, earliest) for sym, (plans, earliest) in ordered[i:i + batch_size]
                 if earliest is not None]
        if not chunk:
            continue
        start = min(e for _s, _p, e in chunk).replace(second=0, microsecond=0)
        latest = max(_parse(a) or e for _s, plans, e in chunk for _o, a, _r in plans)
        end = min(now, latest + FORWARD_WINDOW)
        yield ([sym for sym, _p, _e in chunk],
               start.isoformat().replace("+00:00", "Z"),
               end.isoformat().replace("+00:00", "Z"))


def _retrieve_bars(ordered, report, *, now, fetch_bars, fetch_bars_batch,
                   batch_size: int):
    """Pass 2a — fetch each processed symbol's bars ONCE. Returns
    ({symbol: bars}, {symbol: failure_reason}) where failure_reason is one of
    PROVIDER_ERROR / RATE_LIMITED (wholesale, retried next run)."""
    bars_by_symbol: Dict[str, list] = {}
    errors: Dict[str, str] = {}
    fetchable = [(s, v) for s, v in ordered if v[1] is not None]
    report["unique_symbols_requested"] = len(fetchable)
    if fetch_bars is not None:  # legacy per-symbol injectable (tests / tooling)
        report["fetch_mode"] = "per_symbol"
        for symbol, (_plans, earliest) in fetchable:
            try:
                bars_by_symbol[symbol] = fetch_bars(symbol, earliest.date().isoformat()) or []
            except Exception as exc:
                errors[symbol] = ("RATE_LIMITED" if getattr(exc, "rate_limited", False)
                                  else "PROVIDER_ERROR")
        return bars_by_symbol, errors
    report["fetch_mode"] = "batch"
    throttled = False
    for symbols, start, end in _batch_windows(fetchable, now, batch_size):
        if throttled:  # circuit breaker: don't keep hammering a throttled provider
            errors.update({s: "RATE_LIMITED" for s in symbols})
            continue
        try:
            got = fetch_bars_batch(symbols, start, end) or {}
        except Exception as exc:
            rate_limited = bool(getattr(exc, "rate_limited", False))
            throttled = throttled or rate_limited
            errors.update({s: "RATE_LIMITED" if rate_limited else "PROVIDER_ERROR"
                           for s in symbols})
            continue
        for s in symbols:
            bars_by_symbol[s] = got.get(s) or []
    return bars_by_symbol, errors


def mature_observations(observations, *, now=None, slack_min: int = 15,
                        dry_run: bool = False, fetch_bars=None, save_fn=None,
                        max_symbols: int = MAX_SYMBOLS, fetch_bars_batch=None,
                        request_stats=None, batch_size: int = BATCH_SIZE,
                        exclusion_reason=None,
                        retire_after=RETIRE_AFTER) -> Dict[str, Any]:
    """Pure-ish orchestration (injectable `fetch_bars`/`save_fn` for tests).

    Bounded per run: at most `max_symbols` distinct symbols are fetched, in
    oldest-anchor-first order, so the worker finishes within its CI timeout and
    the backlog drains deterministically across the every-30-min schedule
    (idempotent — deferred symbols mature on a later run). Set max_symbols<=0 for
    no cap (tests).

    Retrieval: `fetch_bars(symbol, start_date)` (legacy, one call per symbol) if
    given, else `fetch_bars_batch(symbols, start_iso, end_iso) -> {symbol: bars}`
    (default: batched Alpaca). `request_stats` (AlpacaRequestStats-like) feeds the
    request/429/retry counters. `exclusion_reason(symbol) -> str|None` drops
    US_MARKET-ineligible symbols before they consume the per-run budget."""
    now = now or _now()
    if not isinstance(now, _dt.datetime):
        now = _parse(now) or _now()
    report = _new_report()

    # Group by symbol so each symbol's future bars are fetched ONCE per run.
    by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in observations or []:
        report["observations_scanned"] += 1
        by_symbol[str(o.get("symbol") or "").upper()].append(o)

    # Ineligible instruments never reach retrieval nor the per-run cap.
    exclusion_reason = exclusion_reason or _default_exclusion_reason
    inel = report["ineligible"]
    for symbol in list(by_symbol):
        try:
            reason = exclusion_reason(symbol)
        except Exception:
            reason = None
        if reason:
            inel["symbols"] += 1
            inel["observations"] += len(by_symbol[symbol])
            inel["reasons"][reason] = inel["reasons"].get(reason, 0) + 1
            del by_symbol[symbol]

    retired = report["retired"]
    if retire_after is not None:
        retired["retire_after_days"] = round(retire_after.total_seconds() / 86400.0, 3)

    # Pass 1 (no I/O): build per-symbol plans + earliest ready anchor.
    symbol_plans: Dict[str, tuple] = {}
    for symbol, obs_list in by_symbol.items():
        plans = []
        symbol_retired = False
        for o in obs_list:
            anchor = o.get("scan_timestamp") or o.get("timestamp")
            elig = horizon_eligibility(anchor, now, (o.get("outcomes") or {}).keys(),
                                       slack_min=slack_min)
            for h, status in elig.items():
                if status in ("already", "not_ready"):
                    report["horizons"][h][status] += 1
            ready = [h for h, s in elig.items() if s == "ready"]
            a = _parse(anchor)
            if ready and retire_after is not None and a is not None and now >= a + retire_after:
                retired["observations"] += 1
                retired["horizons"] += len(ready)
                symbol_retired = True
                continue
            if ready:
                plans.append((o, anchor, ready))
        if not plans and symbol_retired:
            retired["symbols"] += 1  # every ready horizon of this symbol retired
        if plans:
            anchors = [_parse(p[1]) for p in plans if _parse(p[1])]
            symbol_plans[symbol] = (plans, min(anchors) if anchors else None)

    # Oldest-anchor first so the backlog drains and near-complete symbols finish.
    ordered = sorted(symbol_plans.items(),
                     key=lambda kv: (kv[1][1] is None, kv[1][1] or now))
    report["symbols_with_ready_horizons"] = len(ordered)
    ready_symbols_total = len(ordered)
    ready_observations_total = sum(len(v[0]) for _s, v in ordered)
    deferred_symbols = 0
    if max_symbols and max_symbols > 0 and len(ordered) > max_symbols:
        deferred = ordered[max_symbols:]
        deferred_symbols = len(deferred)
        report["symbols_deferred"] = deferred_symbols
        # Oldest observation still pending AFTER this run = oldest deferred anchor.
        oldest_deferred = min((v[1] for _s, v in deferred if v[1] is not None),
                              default=None)
        report["backlog"]["oldest_pending_age_min"] = (
            round((now - oldest_deferred).total_seconds() / 60.0, 1)
            if oldest_deferred is not None else None)
        ordered = ordered[:max_symbols]

    report["backlog"].update({
        "ready_symbols": ready_symbols_total,
        "ready_observations": ready_observations_total,
        "processed_symbols": len(ordered),
        "deferred_symbols": deferred_symbols,
        "max_symbols": int(max_symbols),
        "estimated_clearance_runs": (
            -(-ready_symbols_total // max_symbols) if max_symbols and max_symbols > 0
            else 1),
    })

    # Pass 2a (I/O): retrieve each symbol's bars once (batched, retried, cached).
    if fetch_bars is None and fetch_bars_batch is None:
        fetch_bars_batch, request_stats = _default_fetch_batch_factory(request_stats)
    bars_by_symbol, fetch_errors = _retrieve_bars(
        ordered, report, now=now, fetch_bars=fetch_bars,
        fetch_bars_batch=fetch_bars_batch, batch_size=batch_size)

    # Pass 2b (no provider I/O): mature from the retrieved bars.
    for symbol, (plans, earliest) in ordered:
        report["eligible_observations"] += len(plans)
        if earliest is None:
            _fail(report, None, "INVALID_TIMESTAMP")
            report["backlog"]["failed_symbols"] += 1
            continue
        if symbol in fetch_errors:
            reason = fetch_errors[symbol]
            _fail(report, None, reason)
            report["backlog"]["failed_symbols"] += 1
            report["rate_limited_symbols" if reason == "RATE_LIMITED"
                   else "provider_error_symbols"] += 1
            continue
        bars = bars_by_symbol.get(symbol) or []
        report["cache_misses"] += 1
        if not bars:
            for _o, _a, ready in plans:
                for h in ready:
                    _fail(report, h, "PRICE_DATA_UNAVAILABLE")
            report["backlog"]["failed_symbols"] += 1
            report["price_data_unavailable_symbols"] += 1
            continue
        report["cache_hits"] += len(plans) - 1

        for o, anchor, ready in plans:
            a = _parse(anchor)
            after = [b for b in bars if (_parse(b.get("t")) or a) >= a]
            closes = [b.get("c") for b in after if b.get("c") is not None]
            if len(closes) < 2:
                for h in ready:
                    _fail(report, h, "INSUFFICIENT_FUTURE_BARS")
                continue
            eval_times = {h: (a + _dt.timedelta(minutes=HORIZON_BARS[h])).isoformat()
                          for h in ready}
            direction = str((o.get("scanners") or [{}])[0].get("direction") or "long")
            outcomes = compute_matured_outcomes(
                o, prices_after=closes,
                horizon_bars={h: HORIZON_BARS[h] for h in ready},
                evaluation_times=eval_times, direction=direction)
            produced = {oc["horizon"] for oc in outcomes}
            for h in ready:
                if h not in produced:
                    _fail(report, h, "INSUFFICIENT_FUTURE_BARS")
            if dry_run:
                for oc in outcomes:
                    report["horizons"][oc["horizon"]]["new"] += 1
                    report["attached"] += 1
                continue
            for oc in outcomes:
                try:
                    wrote = (save_fn or _default_save)(oc)
                except Exception:
                    _fail(report, oc["horizon"], "DATABASE_ERROR")
                    continue
                if wrote:
                    report["horizons"][oc["horizon"]]["new"] += 1
                    report["attached"] += 1
                else:
                    report["horizons"][oc["horizon"]]["already"] += 1
    # matured_observations counts NEW outcome rows written (real drain this run);
    # in dry-run it mirrors would-be writes so clearance estimates stay comparable.
    report["backlog"]["matured_observations"] = report["attached"]
    report.update({
        "symbols_processed": report["backlog"]["processed_symbols"],
        "symbols_deferred": report["backlog"]["deferred_symbols"],
        "outcomes_matured": report["attached"],
        "price_data_failures": report["failures"].get("PRICE_DATA_UNAVAILABLE", 0),
        "insufficient_future_bars": report["failures"].get("INSUFFICIENT_FUTURE_BARS", 0),
    })
    if request_stats is not None:
        report.update({
            "alpaca_requests": int(getattr(request_stats, "requests", 0)),
            "alpaca_429_count": int(getattr(request_stats, "rate_limited", 0)),
            "alpaca_retry_count": int(getattr(request_stats, "retries", 0)),
        })
    processed = report["symbols_processed"]
    report["requests_per_processed_symbol"] = (
        round(report["alpaca_requests"] / processed, 3) if processed else None)
    return report


def _default_fetch_batch_factory(stats=None):
    """Batched Alpaca fetcher bound to a shared per-run request-stats object."""
    from data.price_alpaca import AlpacaRequestStats, fetch_minute_bars_multi
    stats = stats if stats is not None else AlpacaRequestStats()

    def _fetch(symbols, start, end):
        return fetch_minute_bars_multi(symbols, start, end, stats=stats)
    return _fetch, stats


def _default_save(outcome) -> bool:
    from db.hsf_observations import save_outcome
    return save_outcome(outcome)


def render_report_text(r: Dict[str, Any]) -> str:
    lines = ["HSF Outcome Maturation", "----------------------",
             f"Eligible observations: {r['eligible_observations']} "
             f"(scanned {r['observations_scanned']})",
             f"Symbols ready: {r.get('symbols_with_ready_horizons', 0)} "
             f"(deferred this run: {r.get('symbols_deferred', 0)})"]
    for h in HORIZON_BARS:
        s = r["horizons"][h]
        lines.append(f"{h}: new={s['new']} already={s['already']} "
                     f"not_ready={s['not_ready']} failed={s['failed']}")
    lines.append(
        f"Retrieval ({r.get('fetch_mode')}): requests={r.get('alpaca_requests')} "
        f"429s={r.get('alpaca_429_count')} retries={r.get('alpaca_retry_count')} "
        f"symbols_requested={r.get('unique_symbols_requested')} "
        f"cache_hits={r.get('cache_hits')} cache_misses={r.get('cache_misses')} "
        f"no_data_symbols={r.get('price_data_unavailable_symbols')} "
        f"rate_limited_symbols={r.get('rate_limited_symbols')} "
        f"ineligible_symbols={(r.get('ineligible') or {}).get('symbols')}")
    ret = r.get("retired") or {}
    lines.append(f"Retired (window closed > {ret.get('retire_after_days')}d): "
                 f"observations={ret.get('observations')} horizons={ret.get('horizons')} "
                 f"symbols={ret.get('symbols')}")
    if r["failures"]:
        lines.append("Failures: " + ", ".join(f"{k}={v}" for k, v in sorted(r["failures"].items())))
    b = r.get("backlog") or {}
    if b:
        lines.append(
            f"Backlog: ready_symbols={b.get('ready_symbols')} "
            f"processed={b.get('processed_symbols')} deferred={b.get('deferred_symbols')} "
            f"matured={b.get('matured_observations')} failed_symbols={b.get('failed_symbols')} "
            f"oldest_pending_min={b.get('oldest_pending_age_min')} "
            f"clearance_runs={b.get('estimated_clearance_runs')}")
    return "\n".join(lines)


def main() -> int:
    from db.hsf_observations import load_recent_observations

    ap = argparse.ArgumentParser(description="Mature captured observation outcomes")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--slack-min", type=int, default=15)
    ap.add_argument("--max-symbols", type=int, default=MAX_SYMBOLS,
                    help="max distinct symbols to fetch this run (bounds runtime; "
                         "backlog drains across the schedule). <=0 = no cap.")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="symbols per multi-symbol Alpaca request series")
    ap.add_argument("--retire-after-days", type=float,
                    default=RETIRE_AFTER.total_seconds() / 86400.0,
                    help="skip ready horizons whose anchor is older than this "
                         "(bounded window closed; retry cannot help). 0 = never retire.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "automation"))
    args = ap.parse_args()

    # attach_outcomes=True (Run 52): matured horizons come back on each record so
    # already-matured work is classified `already` and skipped, not re-fetched.
    observations = load_recent_observations(limit=args.limit,
                                            attach_outcomes=True) or []
    report = mature_observations(observations, slack_min=args.slack_min,
                                 dry_run=args.dry_run, max_symbols=args.max_symbols,
                                 batch_size=args.batch_size,
                                 retire_after=(_dt.timedelta(days=args.retire_after_days)
                                               if args.retire_after_days > 0 else None))
    report["dry_run"] = bool(args.dry_run)

    out_dir = Path(args.out)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "maturation_report.json").write_text(json.dumps(report, indent=2, default=str))
    except Exception:
        pass
    print(render_report_text(report))
    print(f"[mature_observations] attached={report['attached']} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
