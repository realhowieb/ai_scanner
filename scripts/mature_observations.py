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
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from analytics.observation_capture import (
    HORIZON_BARS,
    compute_matured_outcomes,
    horizon_eligibility,
)
from analytics.stair_step_research import (
    OUTCOME_HORIZONS as STAIR_STEP_HORIZONS,
)
from analytics.stair_step_research import (
    compute_stair_step_outcomes,
    horizons_for_observation,
    is_stair_step_observation,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_HORIZONS = {**HORIZON_BARS, **STAIR_STEP_HORIZONS}

# Per-run distinct-symbol cap. Raised 400 -> 2000 after dry-run 36224599421
# measured 4 requests / 400 symbols (0.01 req/symbol, 0 x 429, 55 s job) with
# batched retrieval; 2000 covers the full ready set (~1.6k) in one run. That was
# a DRY run (no writes): real runs at 2000 on 2026-09-28 hit the 20-min CI limit
# because each saved outcome opened a new DB connection. Now one connection per
# run (_RunSaver) plus TIME_BUDGET_MIN keep a run inside its timeout.
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
# Wall-clock budget per scheduled run (CLI default). When it runs out, symbols not
# yet reached are deferred to the next run like cap overflow, so the job finishes
# and reports inside its CI timeout (30 min) instead of being cancelled with no
# report. Measured need: a real 2000-symbol run on a trading day exceeded 20 min
# because every saved outcome opened its own database connection (fixed below).
TIME_BUDGET_MIN = 12.0
# Symbols already fetched when the budget runs out are still saved (their bars are
# in hand), within this extra fraction of the budget: 12 + 6 min + loading stays
# well inside the 30-min CI timeout.
SAVE_GRACE_FRACTION = 0.5
PROGRESS_EVERY_SAVES = 500


def _log(msg: str) -> None:
    print(f"[mature_observations] {msg}", flush=True)


class _RunSaver:
    """Default writer: ONE database connection for the whole run.

    Previously every outcome opened (and closed) its own Neon connection, which
    costs a few hundred ms from CI — ~5k outcomes took well over 20 min. Writes
    are unchanged: the same `save_outcome` INSERT ... ON CONFLICT DO NOTHING with
    its own commit per row (first write wins). `save_outcome` returns False both
    for "already written" and for an error, so after a False the connection is
    checked: if the transaction is aborted or the connection broken, it is rolled
    back / reopened and the row retried once, and a second failure raises so the
    caller records DATABASE_ERROR — an error is never counted as already written."""

    def __init__(self, connect=None):
        self._connect = connect
        self.conn = None
        self.reopened = 0

    def _open(self):
        if self.conn is None:
            if self._connect is None:
                from db.engine import get_neon_conn

                self._connect = get_neon_conn
            self.conn = self._connect()
        return self.conn

    @staticmethod
    def _usable(conn) -> bool:
        if getattr(conn, "closed", False) or getattr(conn, "broken", False):
            return False
        status = getattr(getattr(conn, "info", None), "transaction_status", None)
        return getattr(status, "name", "") != "INERROR"

    def _reset(self) -> None:
        conn, self.conn = self.conn, None
        try:
            if conn is not None and not getattr(conn, "closed", False) and not getattr(conn, "broken", False):
                conn.rollback()
                self.conn = conn
                return
        except Exception:
            pass
        self.close_quietly(conn)
        self.reopened += 1

    @staticmethod
    def close_quietly(conn) -> None:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    def __call__(self, outcome) -> bool:
        from db.hsf_observations import save_outcome

        for _attempt in (1, 2):
            conn = self._open()
            if conn is None:  # no Neon configured: keep the old per-row path (SQLite fallback)
                return save_outcome(outcome)
            if save_outcome(outcome, conn=conn):
                return True
            if self._usable(conn):
                return False  # genuinely already written
            self._reset()
        raise RuntimeError("outcome write failed after reconnect")

    def save_many(self, outcomes) -> list:
        """P2-41: one transaction (one commit) for a symbol's outcomes. Raises on
        any error after resetting the connection; the caller then saves row by
        row through __call__, so an error is never counted as already written."""
        from db.hsf_observations import save_outcomes_batch

        conn = self._open()
        if conn is None:
            raise RuntimeError("no Neon connection; use the per-row path")
        try:
            return save_outcomes_batch(list(outcomes), conn=conn)
        except Exception:
            self._reset()
            raise

    def close(self) -> None:
        self.close_quietly(self.conn)
        self.conn = None


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
                     for h in REPORT_HORIZONS},
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


def _horizons_for(observation: Dict[str, Any]) -> Dict[str, int]:
    """Return the observation's declared horizons without changing defaults."""
    stair_step_horizons = horizons_for_observation(observation)
    return stair_step_horizons or HORIZON_BARS


def _t(trace, o, horizon, status) -> None:
    """Record one scheduling decision (status only, never outcome values)."""
    if trace is not None:
        trace.append({"observation_id": str(o.get("observation_id") or ""),
                      "symbol": str(o.get("symbol") or "").upper(),
                      "horizon": horizon, "status": status})


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
                   batch_size: int, out_of_time=None, budget_deferred=None):
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
    batches = list(_batch_windows(fetchable, now, batch_size))
    for i, (symbols, start, end) in enumerate(batches, 1):
        if out_of_time is not None and out_of_time():
            # Time budget spent: the rest wait for the next run (not failures).
            for later_symbols, _start, _end in batches[i - 1:]:
                budget_deferred.update(later_symbols)
            report["unique_symbols_requested"] -= len(budget_deferred)
            report["time_budget"]["hit_during"] = "fetch"
            _log(f"time budget reached before fetch batch {i}/{len(batches)}; "
                 f"deferring {len(budget_deferred)} symbols to the next run")
            break
        _log(f"fetch batch {i}/{len(batches)}: {len(symbols)} symbols")
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
                        retire_after=RETIRE_AFTER, trace=None,
                        time_budget_s=None, clock=None) -> Dict[str, Any]:
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
    US_MARKET-ineligible symbols before they consume the per-run budget.

    `trace` (Run 58, observability only): if a list is given, one
    {observation_id, symbol, horizon, status} entry is appended per decision
    (ALREADY, NOT_READY, RETIRED, INELIGIBLE, DEFERRED, MATURED, ALREADY_WRITTEN,
    or the failure reason). It never contains outcome values and never changes
    the report or any write.

    `time_budget_s` (optional): wall-clock seconds for retrieval + saving; when
    spent, symbols not yet reached are DEFERRED to the next run (reported in
    `time_budget` and the backlog), never failed. None = no budget (tests/tools).
    `clock` is injectable for tests (default time.monotonic)."""
    clock = clock or time.monotonic
    started = clock()

    def out_of_time(grace: float = 0.0) -> bool:
        return (time_budget_s is not None
                and clock() - started >= time_budget_s * (1.0 + grace))

    now = now or _now()
    if not isinstance(now, _dt.datetime):
        now = _parse(now) or _now()
    report = _new_report()
    report["time_budget"] = {"seconds": time_budget_s, "hit_during": None,
                             "deferred_symbols": 0}

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
            for o in by_symbol[symbol]:
                _t(trace, o, "*", "INELIGIBLE")
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
            elig = horizon_eligibility(
                anchor,
                now,
                (o.get("outcomes") or {}).keys(),
                horizon_bars=_horizons_for(o),
                slack_min=slack_min,
            )
            for h, status in elig.items():
                if status in ("already", "not_ready"):
                    report["horizons"][h][status] += 1
                    _t(trace, o, h, status.upper())
            ready = [h for h, s in elig.items() if s == "ready"]
            a = _parse(anchor)
            if ready and retire_after is not None and a is not None and now >= a + retire_after:
                retired["observations"] += 1
                retired["horizons"] += len(ready)
                for h in ready:
                    _t(trace, o, h, "RETIRED")
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
        for _s, (plans, _e) in deferred:
            for o, _a, ready in plans:
                for h in ready:
                    _t(trace, o, h, "DEFERRED")
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
    budget_deferred: set = set()
    bars_by_symbol, fetch_errors = _retrieve_bars(
        ordered, report, now=now, fetch_bars=fetch_bars,
        fetch_bars_batch=fetch_bars_batch, batch_size=batch_size,
        out_of_time=out_of_time, budget_deferred=budget_deferred)

    run_saver = None
    if save_fn is None and not dry_run:
        run_saver = save_fn = _RunSaver()
    try:
        _mature_from_bars(ordered, report, bars_by_symbol, fetch_errors, budget_deferred,
                          save_fn=save_fn, dry_run=dry_run, trace=trace,
                          out_of_time=out_of_time)
    finally:
        if run_saver is not None:
            report["db_reconnects"] = run_saver.reopened
            run_saver.close()
    _apply_budget_deferral(report, ordered, budget_deferred, now, trace)
    return _finish_report(report, request_stats)


def _mature_from_bars(ordered, report, bars_by_symbol, fetch_errors, budget_deferred, *,
                      save_fn, dry_run, trace, out_of_time):
    """Pass 2b (no provider I/O): mature from the retrieved bars."""
    saved = 0
    for symbol, (plans, earliest) in ordered:
        if symbol in budget_deferred:
            continue
        if out_of_time(SAVE_GRACE_FRACTION):
            report["time_budget"]["hit_during"] = report["time_budget"]["hit_during"] or "save"
            remaining = [s for s, _v in ordered if s not in budget_deferred]
            start = remaining.index(symbol)
            budget_deferred.update(remaining[start:])
            _log(f"time budget reached while saving; deferring {len(remaining) - start} symbols")
            break
        report["eligible_observations"] += len(plans)
        if earliest is None:
            _fail(report, None, "INVALID_TIMESTAMP")
            report["backlog"]["failed_symbols"] += 1
            for o, _a, ready in plans:
                for h in ready:
                    _t(trace, o, h, "INVALID_TIMESTAMP")
            continue
        if symbol in fetch_errors:
            reason = fetch_errors[symbol]
            _fail(report, None, reason)
            for o, _a, ready in plans:
                for h in ready:
                    _t(trace, o, h, reason)
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
                    _t(trace, _o, h, "PRICE_DATA_UNAVAILABLE")
            report["backlog"]["failed_symbols"] += 1
            report["price_data_unavailable_symbols"] += 1
            continue
        report["cache_hits"] += len(plans) - 1

        pending = []  # (observation, outcome) for this symbol, saved together below
        for o, anchor, ready in plans:
            a = _parse(anchor)
            if is_stair_step_observation(o):
                outcomes = compute_stair_step_outcomes(o, bars, horizons=ready)
            else:
                after = [b for b in bars if (_parse(b.get("t")) or a) >= a]
                closes = [b.get("c") for b in after if b.get("c") is not None]
                if len(closes) < 2:
                    for h in ready:
                        _fail(report, h, "INSUFFICIENT_FUTURE_BARS")
                        _t(trace, o, h, "INSUFFICIENT_FUTURE_BARS")
                    continue
                horizon_bars = _horizons_for(o)
                eval_times = {
                    h: (a + _dt.timedelta(minutes=horizon_bars[h])).isoformat()
                    for h in ready
                }
                direction = str((o.get("scanners") or [{}])[0].get("direction") or "long")
                outcomes = compute_matured_outcomes(
                    o,
                    prices_after=closes,
                    horizon_bars={h: horizon_bars[h] for h in ready},
                    evaluation_times=eval_times,
                    direction=direction,
                )
            produced = {oc["horizon"] for oc in outcomes}
            for h in ready:
                if h not in produced:
                    _fail(report, h, "INSUFFICIENT_FUTURE_BARS")
                    _t(trace, o, h, "INSUFFICIENT_FUTURE_BARS")
            if dry_run:
                for oc in outcomes:
                    report["horizons"][oc["horizon"]]["new"] += 1
                    report["attached"] += 1
                    _t(trace, o, oc["horizon"], "MATURED")
                continue
            pending.extend((o, oc) for oc in outcomes)
        saved = _save_symbol_outcomes(pending, report, trace, save_fn, saved)


def _save_symbol_outcomes(pending, report, trace, save_fn, saved: int) -> int:
    """Save one symbol's outcomes and record each result.

    P2-41: a saver with `save_many` writes them in one transaction (one commit
    instead of one per row). If that batch fails it was rolled back as a whole,
    so every row is retried one at a time and classified individually. Savers
    without `save_many` (tests, SQLite fallback) keep the per-row path."""
    if not pending:
        return saved
    results = None
    save_many = getattr(save_fn, "save_many", None)
    if save_many is not None:
        try:
            results = list(save_many([oc for _o, oc in pending]))
        except Exception:
            results = None
    for i, (o, oc) in enumerate(pending):
        if results is not None and i < len(results):
            wrote = bool(results[i])
        else:
            try:
                wrote = (save_fn or _default_save)(oc)
            except Exception:
                _fail(report, oc["horizon"], "DATABASE_ERROR")
                _t(trace, o, oc["horizon"], "DATABASE_ERROR")
                continue
        saved += 1
        if saved % PROGRESS_EVERY_SAVES == 0:
            _log(f"saved {saved} outcomes")
        if wrote:
            report["horizons"][oc["horizon"]]["new"] += 1
            report["attached"] += 1
            _t(trace, o, oc["horizon"], "MATURED")
        else:
            report["horizons"][oc["horizon"]]["already"] += 1
            _t(trace, o, oc["horizon"], "ALREADY_WRITTEN")
    return saved


def _apply_budget_deferral(report, ordered, budget_deferred, now, trace) -> None:
    """Symbols the time budget didn't reach count as deferred (like cap overflow)."""
    if not budget_deferred:
        return
    report["time_budget"]["deferred_symbols"] = len(budget_deferred)
    b = report["backlog"]
    b["processed_symbols"] -= len(budget_deferred)
    b["deferred_symbols"] += len(budget_deferred)
    report["symbols_deferred"] = b["deferred_symbols"]
    anchors = [v[1] for s, v in ordered if s in budget_deferred and v[1] is not None]
    if anchors:
        age = round((now - min(anchors)).total_seconds() / 60.0, 1)
        prev = b.get("oldest_pending_age_min")
        b["oldest_pending_age_min"] = age if prev is None else max(prev, age)
    for s, (plans, _e) in ordered:
        if s in budget_deferred:
            for o, _a, ready in plans:
                for h in ready:
                    _t(trace, o, h, "DEFERRED")


def _finish_report(report, request_stats):
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
    for h in REPORT_HORIZONS:
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
    ap.add_argument("--time-budget-min", type=float, default=TIME_BUDGET_MIN,
                    help="stop starting new work after this many minutes; the rest "
                         "is deferred to the next run. <=0 = no budget.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "automation"))
    args = ap.parse_args()
    t0 = time.monotonic()

    # attach_outcomes=True (Run 52): matured horizons come back on each record so
    # already-matured work is classified `already` and skipped, not re-fetched.
    observations = load_recent_observations(limit=args.limit,
                                            attach_outcomes=True) or []
    _log(f"loaded {len(observations)} observations in {time.monotonic() - t0:.0f}s")
    budget = None
    if args.time_budget_min > 0:
        budget = max(0.0, args.time_budget_min * 60.0 - (time.monotonic() - t0))
    report = mature_observations(observations, slack_min=args.slack_min, time_budget_s=budget,
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
