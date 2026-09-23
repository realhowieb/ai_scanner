"""Run scheduled market scans without the Streamlit UI."""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "artifacts" / "scheduled_scan_summary.json"
COVERAGE_DIR = ROOT / "artifacts" / "automation"


_SCAN_LOCK = ROOT / "artifacts" / "cron_scan.lock"


def _acquire_scan_lock(max_age_s: int = 1800) -> bool:
    """Lightweight single-host overlap protection (Run 45, Task 9). Returns True
    when the lock is acquired. A stale lock older than `max_age_s` is reclaimed
    (a crashed run never blocks forever). Best-effort; never raises."""
    try:
        import time as _t
        _SCAN_LOCK.parent.mkdir(parents=True, exist_ok=True)
        if _SCAN_LOCK.exists():
            age = _t.time() - _SCAN_LOCK.stat().st_mtime
            if age < max_age_s:
                return False
        _SCAN_LOCK.write_text(str(dt.datetime.now(dt.timezone.utc).isoformat()))
        return True
    except Exception:
        return True  # fail-open: never let lock IO prevent a scheduled scan


def _release_scan_lock() -> None:
    try:
        if _SCAN_LOCK.exists():
            _SCAN_LOCK.unlink()
    except Exception:
        pass


def _append_perf_history(record: dict, *, keep: int = 60) -> None:
    """Append a per-run performance record to a rolling JSONL history for
    repeatability comparison (Run 45). Best-effort; never raises."""
    try:
        path = ROOT / "artifacts" / "automation" / "perf_history.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        if path.exists():
            lines = path.read_text().splitlines()[-(keep - 1):]
        lines.append(json.dumps(record, default=str))
        path.write_text("\n".join(lines) + "\n")
    except Exception:
        pass


def _write_coverage_artifact(universe: str, report: dict) -> None:
    """Persist the coverage/health report as an artifact (Run 37). Best-effort;
    mirrors the automation export's artifacts/automation location so the existing
    scheduled-scans workflow uploads it. Never raises."""
    try:
        COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(c for c in str(universe).lower() if c.isalnum() or c in "-_") or "scan"
        (COVERAGE_DIR / f"coverage_{safe}.json").write_text(
            json.dumps(report, indent=2, default=str))
    except Exception:
        pass

# Report best-effort task failures to Sentry when configured (no-op otherwise),
# so silently-swallowed cron problems still get counted somewhere.
try:
    from ui.monitoring import capture as _capture
except Exception:  # pragma: no cover - fallback when monitoring is unavailable
    def _capture(exc: BaseException) -> None:
        pass


@dataclass
class ScanRunSummary:
    universe: str
    ok: bool
    row_count: int = 0
    duration_sec: float = 0.0
    error: str | None = None
    automation_export: dict | None = None


def _read_symbols(path: Path) -> list[str]:
    if not path.exists():
        return []
    symbols: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        symbol = line.strip().upper()
        if symbol and not symbol.startswith("#"):
            symbols.append(symbol)
    return symbols


def _dedupe(symbols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for symbol in symbols:
        if symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


def _load_universe_result(universe: str) -> tuple[list[str], dict]:
    """Return (symbols, metadata) for a universe. US_MARKET uses the canonical
    live/cached Alpaca provider (no 2,000 cap); legacy universes read the static
    files. Metadata records the source so a scheduled run can prove what it scanned.
    """
    key = universe.strip().upper()
    if key == "US_MARKET":
        from data.us_market_universe import build_us_market_universe
        res = build_us_market_universe()
        meta = {"universe_name": "US_MARKET", "universe_source": res.get("source"),
                "universe_generated_at": res.get("generated_at"),
                "universe_symbol_count": res.get("symbol_count"),
                "is_fallback": res.get("is_fallback", False),
                "fallback_reason": res.get("fallback_reason"),
                "provider_assets": res.get("provider_assets"),
                "exclusions": res.get("exclusions")}
        return list(res.get("symbols") or []), meta
    sp500 = _read_symbols(ROOT / "sp500.txt")
    nasdaq = _read_symbols(ROOT / "nasdaq.txt")
    if key == "SP500":
        return sp500, {"universe_name": "SP500", "universe_source": "static"}
    if key == "NASDAQ":
        limit = int(os.getenv("CRON_NASDAQ_LIMIT", "2000"))
        return nasdaq[:limit], {"universe_name": "NASDAQ", "universe_source": "static"}
    if key == "COMBO":
        limit = int(os.getenv("CRON_NASDAQ_LIMIT", "2000"))
        return _dedupe([*sp500, *nasdaq[:limit]]), {"universe_name": "COMBO",
                                                    "universe_source": "static"}
    raise ValueError(f"Unknown universe: {universe}")


def _load_universe(universe: str) -> list[str]:
    """Back-compat: symbols only."""
    return _load_universe_result(universe)[0]


def _results_to_json(results) -> str:
    if hasattr(results, "to_json"):
        return results.to_json(orient="records", date_format="iso")
    return json.dumps(results, default=str)


def _write_summary(summary: dict, path: Path | None = None) -> None:
    """Write a scheduled scan summary artifact for CI/deployment review."""
    target = path or Path(os.getenv("CRON_SUMMARY_PATH", str(SUMMARY_PATH)))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _market_closed_today(now_utc: dt.datetime) -> bool:
    """True only when Alpaca's calendar confirms today is NOT a trading day
    (a weekday market holiday). Fails safe: on missing creds or any API error it
    returns False, so a real trading day is never skipped by mistake."""
    try:
        import requests

        from data.alpaca_config import get_alpaca_config, get_alpaca_headers

        cfg = get_alpaca_config()
        headers = get_alpaca_headers()
        if not cfg or not headers:
            return False
        day = now_utc.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        base = (cfg.get("base_url") or "https://paper-api.alpaca.markets").rstrip("/")
        r = requests.get(
            f"{base}/v2/calendar", params={"start": day, "end": day},
            headers=headers, timeout=8,
        )
        if r.status_code != 200:
            return False
        return len(r.json() or []) == 0        # empty calendar → market closed today
    except Exception:
        return False


def _skip_reason(now_utc: dt.datetime | None = None) -> str | None:
    """Return a reason to skip scheduled scans, or None when scans may run."""
    # CRON_FORCE=1 bypasses the weekend/premarket/holiday skip — used by the
    # manual "Run workflow" trigger so the pipeline can be tested any time.
    if os.getenv("CRON_FORCE", "").strip() == "1":
        return None

    if now_utc is None:
        now_utc = dt.datetime.now(dt.timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=dt.timezone.utc)

    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return "Weekend detected - skipping scans."
    if now_et.hour < 6:
        return "Too early (premarket) - skipping scans."
    if _market_closed_today(now_utc):
        return "Market holiday - skipping scans."
    return None


def _resolve_session(now_utc: dt.datetime | None = None) -> str:
    """Resolve the scan session: explicit CRON_SESSION, or inferred from ET time.

    'auto' (the scheduled default) maps ET time to a session so one external
    cron job covers every slot: before 9:30 -> premarket, 16:00 or later ->
    postmarket, otherwise regular.
    """
    explicit = os.getenv("CRON_SESSION", "auto").strip().lower()
    if explicit in ("regular", "premarket", "postmarket"):
        return explicit

    if now_utc is None:
        now_utc = dt.datetime.now(dt.timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=dt.timezone.utc)
    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
    minutes = now_et.hour * 60 + now_et.minute
    if minutes < 9 * 60 + 30:
        return "premarket"
    if minutes >= 16 * 60:
        return "postmarket"
    return "regular"


def _run_session_scan(session: str) -> "ScanRunSummary":
    """Run the session-labeled premarket/postmarket scan (scan.pre_post)."""
    started = time.time()
    try:
        from scan.pre_post import run_postmarket_headless, run_premarket_headless

        rc = run_premarket_headless() if session == "premarket" else run_postmarket_headless()
        ok = rc == 0
        return ScanRunSummary(
            universe=session.upper(),
            ok=ok,
            duration_sec=time.time() - started,
            error=None if ok else f"{session} scan returned {rc}",
        )
    except Exception as e:
        _capture(e)
        return ScanRunSummary(
            universe=session.upper(),
            ok=False,
            duration_sec=time.time() - started,
            error=f"{type(e).__name__}: {e}",
        )


def _configured_universes() -> list[str]:
    """Return the universes configured for scheduled scans.

    Environment variable:
    - CRON_UNIVERSES=US_MARKET (default) — the canonical full U.S. market.
    - CRON_UNIVERSES=SP500,NASDAQ,COMBO — legacy universes, still selectable.

    Run 44: the scheduled default is now US_MARKET (whole-market coverage), not
    SP500,NASDAQ,COMBO (which scanned overlapping subsets three times). An explicit
    CRON_UNIVERSES still overrides. Blank entries are ignored.
    """
    raw = os.getenv("CRON_UNIVERSES", "US_MARKET")
    universes = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return universes or ["US_MARKET"]


def run_and_save(
    universe: str,
    username: str = "cron",
    *,
    premarket: bool = False,
    afterhours: bool = False,
    unusual_volume: bool = False,
    min_gap: float | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    top_n: int | None = None,
    profile: str | None = None,
    save_snapshot: bool = False,
) -> ScanRunSummary:
    """Run one universe scan and save results to the configured database.

    save_snapshot=True also promotes the results to the day's daily snapshot
    (idempotent per universe/day) — the track record and morning digest read
    is_snapshot rows, and the cron is the only reliable daily producer.
    """
    from db.runs import save_daily_snapshot, save_run
    from scan.engine import run_breakout_scan

    print(f"\n=== Running {universe} scan @ {dt.datetime.now(dt.timezone.utc).isoformat()} ===")
    try:
        universe_started = time.perf_counter()
        tickers, universe_meta = _load_universe_result(universe)
        universe_load_sec = time.perf_counter() - universe_started
        # Fail safely: a full-market universe with no live AND no cached source
        # must NOT fall back to a partial list or claim coverage.
        if universe_meta.get("universe_source") == "none":
            raise RuntimeError(
                f"{universe}: universe provider unavailable and no cached "
                f"universe — scheduled scan aborted (no partial substitution).")
        if universe_meta.get("is_fallback"):
            print(f"⚠️ {universe}: {universe_meta.get('fallback_reason')} "
                  f"(source={universe_meta.get('universe_source')})")
        print(f"Universe {universe}: {len(tickers)} symbols "
              f"(source={universe_meta.get('universe_source')}, "
              f"load={universe_load_sec:.1f}s)")
        if not tickers:
            raise RuntimeError(f"No tickers loaded for {universe}")

        # Drop delisted / non-tradable names (e.g. EA) from the regular scan too;
        # the headless pre/post path filters inside run_headless_pipeline, but the
        # engine path does not. Fails open (never empties the universe).
        _pre_tradable = len(tickers)
        try:
            from data.tradability import filter_tradable_tickers

            tickers = filter_tradable_tickers(tickers)
        except Exception as _e:
            _capture(_e)
        dropped_untradable = _pre_tradable - len(tickers)

        scan_started_at = dt.datetime.now(dt.timezone.utc)
        started = time.perf_counter()
        coverage_sink: dict = {}
        results = run_breakout_scan(
            tickers,
            coverage_sink=coverage_sink,
            premarket=premarket,
            afterhours=afterhours,
            unusual_volume=unusual_volume,
            min_gap=min_gap if min_gap is not None else float(os.getenv("CRON_MIN_GAP", "0")),
            min_price=min_price if min_price is not None else float(os.getenv("CRON_MIN_PRICE", "1")),
            max_price=max_price if max_price is not None else float(os.getenv("CRON_MAX_PRICE", "1000")),
            top_n=top_n if top_n is not None else int(os.getenv("CRON_TOP_N", "100")),
            # Liquidity floor (20-day-avg dollar volume) to keep illiquid
            # micro-cap/pump names out of scheduled scans + alerts. Default $5M.
            min_dollar_vol=float(os.getenv("CRON_MIN_DOLLAR_VOL", "5000000")),
            profile=profile or os.getenv("CRON_PROFILE", "regular"),
            diagnostics=False,
            use_cache=True,
        )
        duration = time.perf_counter() - started
        completed_at = dt.datetime.now(dt.timezone.utc)
        row_count = len(results)

        run_name = f"{universe} | {row_count} results | {duration:.1f}s"
        print(f"Scan completed: {run_name}")

        # Coverage & data-health report (Run 37). Best-effort; never affects the
        # scan/save outcome. Generated even for throttled/partial runs so a
        # degraded scan is observable rather than silent.
        coverage_report_obj = None
        try:
            from analytics.coverage import (
                build_coverage_funnel,
                classify_health,
                coverage_report,
            )

            eligible = len(tickers)
            expected = eligible + int(dropped_untradable)
            price_success = int(coverage_sink.get("price_success", 0))
            funnel = build_coverage_funnel(
                universe_version=universe,
                expected=expected,
                eligible=eligible,
                attempted=int(coverage_sink.get("attempted", eligible)),
                price_success=price_success,
                skipped=coverage_sink.get("skipped") or [],
                results=row_count,
                scan_started_at=scan_started_at,
                scan_completed_at=completed_at,
                market_session=_resolve_session(),
                duration_sec=duration,
            )
            health = classify_health(funnel)
            coverage_report_obj = coverage_report(funnel, health)
            # Attach universe provenance + performance telemetry so an automated
            # review can verify the whole market was actually attempted.
            coverage_report_obj["universe"] = {
                **universe_meta,
                "universe_load_sec": round(universe_load_sec, 2),
                "eligible_symbol_count": eligible,
                "attempted_symbol_count": int(coverage_sink.get("attempted", eligible)),
                "successfully_priced_count": price_success,
                "skipped_symbol_count": max(0, int(coverage_sink.get("attempted", eligible)) - price_success),
                "result_count": row_count,
                "scan_duration_sec": round(duration, 2),
                "symbols_per_sec": round(eligible / duration, 1) if duration else None,
                "coverage_percentage": funnel.get("coverage_pct"),
            }
            print(coverage_report_obj["text"])
            _write_coverage_artifact(universe, coverage_report_obj)
        except Exception as e:
            print(f"[coverage] report failed for {universe}: {e}")
            _capture(e)

        # Guard: a large universe returning almost nothing means the price fetch
        # was throttled/failed. Don't overwrite a good snapshot with garbage —
        # skip the save and report the run as failed so it's visible.
        min_save_rows = int(os.getenv("CRON_MIN_SAVE_ROWS", "10"))
        if len(tickers) >= 200 and row_count < min_save_rows:
            msg = (
                f"{universe}: only {row_count} rows from {len(tickers)} tickers "
                f"(< {min_save_rows}); likely throttled — snapshot NOT saved."
            )
            print(f"⚠️ {msg}")
            return ScanRunSummary(
                universe=universe,
                ok=False,
                row_count=row_count,
                duration_sec=duration,
                error=f"SkippedSave: {msg}",
            )

        results_json = _results_to_json(results)
        save_run(
            name=run_name,
            label=universe,
            username=username,
            row_count=row_count,
            duration_sec=duration,
            results_json=results_json,
            is_snapshot=False,
            allow_sqlite_fallback=False,
        )
        print(f"Saved {row_count} rows for {universe}.")

        # Canonical observation capture (Run 38A). SIDE-EFFECT ONLY: observes the
        # completed scan and persists per-scanner canonical observations for
        # future research. Never changes results/ranking; fully non-fatal — any
        # failure is logged and the scan continues unchanged. Kill switch:
        # HSF_OBSERVATION_CAPTURE=0. Dry-run: HSF_OBSERVATION_CAPTURE_DRYRUN=1.
        if os.getenv("HSF_OBSERVATION_CAPTURE", "1").strip() != "0":
            try:
                from analytics.observation_capture import (
                    capture_scan_observations,
                    render_capture_text,
                )

                rows = results.to_dict("records") if hasattr(results, "to_dict") else list(results)
                cap = capture_scan_observations(
                    rows,
                    universe=universe,
                    scan_timestamp=scan_started_at,
                    session=_resolve_session(),
                    universe_version=universe,
                    scan_id=scan_started_at.isoformat(),
                    coverage=coverage_report_obj,
                    dry_run=os.getenv("HSF_OBSERVATION_CAPTURE_DRYRUN", "0").strip() == "1",
                )
                print(render_capture_text(cap))
                _write_coverage_artifact(f"{universe}_capture", cap)
            except Exception as e:
                print(f"[observation_capture] failed for {universe}: {e}")
                _capture(e)

        automation_export = None
        try:
            from integrations.automation_export import publish_scan_results

            automation_export = publish_scan_results(
                results,
                universe=universe,
                scan_type="scheduled",
                market_session=_resolve_session(),
                started_at_utc=scan_started_at,
                completed_at_utc=completed_at,
                duration_seconds=duration,
                symbols_requested=len(tickers),
                symbols_processed=coverage_sink.get("price_success"),
                symbols_skipped=len(coverage_sink.get("skipped") or []) or None,
                dropped_untradable=int(dropped_untradable),
                retention_days=int(os.getenv("AUTOMATION_HISTORY_DAYS", "30")),
            )
            print(
                "[automation_export] latest_scan.json published: "
                f"candidates={automation_export.get('candidate_count')} "
                f"warnings={automation_export.get('warning_count')} "
                f"path={automation_export.get('latest_path')}"
            )
        except Exception as e:
            # Export is an integration layer; a failed export should be visible
            # without turning a successful scan into a new outage mode.
            automation_export = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"[automation_export] publish failed for {universe}: {e}")
            _capture(e)

        # Snapshot safety (Run 45, Task 4): only a HEALTHY scan may be promoted as
        # the canonical daily snapshot. A DEGRADED/FAILED full-market scan keeps
        # its artifact + diagnostics (failure evidence preserved) but must NOT
        # overwrite a known-good snapshot. Health is unavailable only when the
        # coverage report failed; then we preserve the prior (row-count-guarded)
        # behavior so legacy universes are unaffected.
        snapshot_promoted = None
        snapshot_suppression_reason = None
        if save_snapshot:
            promote = True
            if coverage_report_obj is not None:
                try:
                    from analytics.scan_reliability import snapshot_decision
                    hstate = (coverage_report_obj.get("health") or {}).get("state")
                    cov_pct = (coverage_report_obj.get("funnel") or {}).get("coverage_pct")
                    decision = snapshot_decision(hstate, cov_pct)
                    promote = decision["promote"]
                    snapshot_suppression_reason = decision["reason"]
                except Exception as e:
                    _capture(e)
            if promote:
                try:
                    save_daily_snapshot(
                        universe, results_json, username=username,
                        row_count=row_count, duration_sec=duration,
                    )
                    snapshot_promoted = True
                    print(f"Saved daily snapshot for {universe}.")
                except Exception as e:
                    snapshot_promoted = False
                    snapshot_suppression_reason = f"snapshot_save_error: {type(e).__name__}"
                    print(f"[cron] snapshot save failed for {universe}: {e}")
            else:
                snapshot_promoted = False
                print(f"⚠️ {universe}: snapshot NOT promoted — {snapshot_suppression_reason} "
                      f"(artifact + diagnostics retained).")

        # Human-readable performance summary + attach reliability telemetry to the
        # coverage artifact (Run 45, Tasks 1/10/11).
        try:
            from analytics.scan_reliability import build_performance_record, render_run_summary
            uni = (coverage_report_obj or {}).get("universe", {}) if coverage_report_obj else {}
            perf = build_performance_record(
                run_id=os.getenv("GITHUB_RUN_ID") or scan_started_at.isoformat(),
                started_at=scan_started_at, completed_at=completed_at,
                market_session=_resolve_session(), universe=universe,
                universe_source=universe_meta.get("universe_source"),
                provider_asset_count=universe_meta.get("provider_assets"),
                eligible_symbol_count=uni.get("eligible_symbol_count") or len(tickers),
                attempted_symbol_count=int(coverage_sink.get("attempted", len(tickers))),
                priced_symbol_count=int(coverage_sink.get("price_success", 0)),
                skipped_symbol_count=len(coverage_sink.get("skipped") or []),
                candidate_count=row_count,
                coverage_percentage=(coverage_report_obj or {}).get("funnel", {}).get("coverage_pct"),
                coverage_health=(coverage_report_obj or {}).get("health", {}).get("state"),
                timings={"total_runtime_seconds": round(duration, 1),
                         "universe_load_seconds": round(universe_load_sec, 2)},
                batch_size=int(os.getenv("CRON_BATCH_SIZE") or 0) or None,
                skipped=coverage_sink.get("skipped") or [],
                snapshot_promoted=snapshot_promoted,
                snapshot_suppression_reason=snapshot_suppression_reason,
            )
            print(render_run_summary(perf))
            if coverage_report_obj is not None:
                coverage_report_obj["performance"] = perf
                _write_coverage_artifact(universe, coverage_report_obj)
            _append_perf_history(perf)
        except Exception as e:
            print(f"[reliability] performance record failed: {e}")
            _capture(e)
        return ScanRunSummary(
            universe=universe,
            ok=True,
            row_count=row_count,
            duration_sec=duration,
            automation_export=automation_export,
        )

    except Exception as e:
        print(f"ERROR running {universe} scan: {e}")
        try:
            from ui.monitoring import capture

            capture(e)
        except Exception:
            pass
        return ScanRunSummary(universe=universe, ok=False, error=f"{type(e).__name__}: {e}")


def _print_provider_status() -> None:
    """Log which price provider will be used, to diagnose Alpaca config."""
    # Resolve through the shared config so the diagnostic reflects exactly what
    # the download code will see (env-first, then guarded secrets).
    from data.alpaca_config import alpaca_secret

    key = alpaca_secret("ALPACA_API_KEY_ID")
    secret = alpaca_secret("ALPACA_API_SECRET_KEY")
    print(
        f"Price provider — ALPACA_API_KEY_ID={'set(…' + key[-4:] + ')' if key else 'MISSING'}, "
        f"ALPACA_API_SECRET_KEY={'set' if secret else 'MISSING'}"
    )
    try:
        from data.price_alpaca import get_alpaca_config
        cfg = get_alpaca_config()
        print(f"Active provider: {'Alpaca' if cfg else 'yfinance (Alpaca not configured)'}")
    except Exception as e:
        print(f"Active provider: unknown (config check failed: {e})")


def _purge_old_login_attempts() -> None:
    """Delete login_attempts rows older than 24 hours (throttled once/day).

    Migrated from the retired top-level scheduler.py APScheduler path, which was
    shadowed by the scheduler/ package and never actually ran — so this purge
    had never executed and the table grew unbounded.
    """
    from db.earnings import mark_earnings_refreshed_today, should_refresh_earnings_today

    key = "cron_login_purge"
    if not should_refresh_earnings_today(key):
        return
    from db.engine import get_neon_conn

    conn = get_neon_conn()
    if conn is None:
        return
    cur = conn.cursor()
    cur.execute("DELETE FROM login_attempts WHERE attempted_at < NOW() - INTERVAL '24 hours'")
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    mark_earnings_refreshed_today(key)
    print(f"[maintenance] purged {deleted} stale login_attempts row(s)")


def _prune_old_runs() -> None:
    """Delete old non-snapshot runs (throttled once/day).

    Every scan stores a full results_json blob; without retention the runs
    table grows unbounded and every metadata query slowly degrades. Snapshots
    are kept forever (track record / history need them); plain runs are pruned
    after CRON_RUNS_RETENTION_DAYS (default 90, 0 disables).
    """
    days = int(os.getenv("CRON_RUNS_RETENTION_DAYS", "90") or "90")
    if days <= 0:
        return
    from db.earnings import mark_earnings_refreshed_today, should_refresh_earnings_today

    key = "cron_runs_prune"
    if not should_refresh_earnings_today(key):
        return
    from db.engine import get_neon_conn

    conn = get_neon_conn()
    if conn is None:
        return
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM runs WHERE is_snapshot = FALSE "
        "AND created_at < NOW() - make_interval(days => %s)",
        (days,),
    )
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    mark_earnings_refreshed_today(key)
    print(f"[maintenance] pruned {deleted} run(s) older than {days}d")


def _refresh_track_record() -> None:
    """Recompute + persist the signal track record once per day (throttled).

    CRON_FORCE=1 (manual run) bypasses the daily throttle for on-demand testing.
    """
    from db.earnings import mark_earnings_refreshed_today, should_refresh_earnings_today

    key = "cron_track_record"
    forced = os.getenv("CRON_FORCE", "").strip() == "1"
    if not forced and not should_refresh_earnings_today(key):
        print("[track_record] already computed today; skipping")
        return

    from analytics.track_record import compute_track_record
    from db.track_record import save_track_record

    any_saved = False
    # Multiple horizons so we can see which holding period the signal actually
    # wins at: 1d (day-trade), 5d (swing), 20d (position). Longer horizons need
    # older snapshots, so they populate later as history accumulates.
    for horizon in (1, 5, 20):
        by_ranking = compute_track_record(horizon_days=horizon)
        if not by_ranking:
            print(f"[track_record] horizon={horizon}: insufficient history")
            continue
        for ranking, summary in by_ranking.items():
            saved = save_track_record(
                horizon_days=summary["horizon_days"],
                avg_return=summary["avg_return"],
                median_return=summary["median_return"],
                win_rate=summary["win_rate"],
                sample_size=summary["sample_size"],
                runs_used=summary["runs_used"],
                benchmark=summary.get("benchmark"),
                top_n=summary.get("top_n"),
                ranking=ranking,
            )
            any_saved = any_saved or saved
            if horizon == 5 and summary.get("daily"):
                try:
                    from db.track_record import save_daily_excess

                    save_daily_excess(ranking, 5, summary["daily"])
                except Exception as e:
                    print(f"[track_record] daily save failed: {e}")
            print(
                f"[track_record] h={horizon} {ranking}: "
                f"excess_vs_{summary.get('benchmark')}={summary['avg_return']:+.2%} "
                f"beat={summary['win_rate']:.0%} n={summary['sample_size']}"
            )

    if any_saved:
        mark_earnings_refreshed_today(key)


def _refresh_signal_leaderboard() -> None:
    """Recompute + persist the signal leaderboard once per day (throttled).

    Ranks the whole slate of scanner signals by benchmark-excess forward return
    (Strategy Lab). CRON_FORCE=1 bypasses the daily throttle. Reuses the same
    eligible-snapshot machinery as the track record.
    """
    from db.earnings import mark_earnings_refreshed_today, should_refresh_earnings_today

    key = "cron_signal_leaderboard"
    forced = os.getenv("CRON_FORCE", "").strip() == "1"
    if not forced and not should_refresh_earnings_today(key):
        print("[leaderboard] already computed today; skipping")
        return

    from analytics.signal_leaderboard import compute_signal_leaderboard
    from db.signal_leaderboard import save_leaderboard

    any_saved = False
    # Both entry conventions: 'close' (enter at signal-day close) and 'open'
    # (enter at signal-day open — the realistic fill for an early signal). Users
    # compare them in the Strategy Lab.
    for entry_mode in ("close", "open"):
        for horizon in (1, 5, 20):
            rows = compute_signal_leaderboard(horizon_days=horizon, entry_mode=entry_mode)
            if not rows:
                print(f"[leaderboard] {entry_mode} h={horizon}: insufficient history")
                continue
            saved = save_leaderboard(horizon, rows, entry_mode=entry_mode)
            any_saved = any_saved or bool(saved)
            best = rows[0]
            print(
                f"[leaderboard] {entry_mode} h={horizon}: {saved} signals; "
                f"best={best['signal']} excess={best['avg_excess']:+.2%} "
                f"beat={best['win_rate']:.0%} n={best['sample_size']}"
            )

    if any_saved:
        mark_earnings_refreshed_today(key)


def _refresh_earnings() -> None:
    """Once-per-day earnings-calendar refresh over the full universe (FMP/Finnhub).

    Throttled via the earnings refresh log so it runs on only one scheduled scan
    per day. Disabled when CRON_EARNINGS_REFRESH=0.
    """
    if os.getenv("CRON_EARNINGS_REFRESH", "1").strip() != "1":
        return
    from db.earnings import (
        mark_earnings_refreshed_today,
        populate_earnings_calendar,
        should_refresh_earnings_today,
    )

    key = "cron_earnings"
    # Manual "Run workflow" (CRON_FORCE=1) bypasses the daily throttle so the
    # refresh can be tested on demand; scheduled runs respect once-per-day.
    forced = os.getenv("CRON_FORCE", "").strip() == "1"
    if not forced and not should_refresh_earnings_today(key):
        print("[earnings] already refreshed today; skipping")
        return

    universe = _dedupe([*_load_universe("SP500"), *_load_universe("NASDAQ")])
    print(f"[earnings] refreshing {len(universe)} symbols (FMP -> Finnhub, bulk)")
    result = populate_earnings_calendar(universe, use_yf_fallback=False, sleep_s=0)
    found = sum(1 for info in result.values() if info.earnings_date is not None)
    print(f"[earnings] cron refresh complete: {found} dated of {len(universe)}")
    # Only mark as refreshed when we actually got dates. If all sources returned
    # nothing (keys missing, rate-limited, provider outage), leave today unmarked
    # so the next scheduled scan retries instead of waiting until tomorrow.
    if found > 0:
        mark_earnings_refreshed_today(key)
    else:
        print("[earnings] 0 dates found — not marking refreshed; will retry next scan")


def main():
    print("=== cron_runner started ===")
    try:
        from ui.monitoring import init_sentry

        init_sentry("cron")
    except Exception:
        pass
    _print_provider_status()
    started_at = dt.datetime.now(dt.timezone.utc)

    skip_reason = _skip_reason()
    if skip_reason:
        print(skip_reason)
        _write_summary(
            {
                "started_at": started_at.isoformat(),
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "ok": True,
                "skipped": True,
                "skip_reason": skip_reason,
                "runs": [],
            }
        )
        return

    # --- Run the scans ---
    # Session routing: premarket/postmarket slots run the session-labeled scan
    # (scan.pre_post) instead of the full universe sweep; the regular slots keep
    # the standard universes + daily snapshot. Alerts/digest/etc. below run for
    # every session (each is throttled or cheap).
    # Overlap protection (Run 45): if a prior full-market scan is still running,
    # this invocation exits safely rather than launching a second concurrent
    # whole-market sweep. The reason is recorded, never hidden.
    if not _acquire_scan_lock():
        print("⚠️ Another scheduled scan is still running — skipping (overlapping_run).")
        _write_summary({
            "started_at": started_at.isoformat(),
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "ok": True, "skipped": True, "skip_reason": "OVERLAPPING_RUN", "runs": [],
        })
        return

    session = _resolve_session()
    print(f"Session: {session}")
    try:
        if session in ("premarket", "postmarket"):
            runs = [_run_session_scan(session)]
        else:
            universes = _configured_universes()
            print(f"Configured universes: {', '.join(universes)}")
            # Regular slots also produce the day's snapshot (per-universe, idempotent).
            runs = [run_and_save(universe, save_snapshot=True) for universe in universes]
    finally:
        _release_scan_lock()

    # Evaluate per-user alerts against the fresh snapshot (best-effort; never
    # let alert failures fail the scan run).
    try:
        from scheduler.alert_runner import run_alerts

        run_alerts()
    except Exception as e:
        print(f"[cron] alert evaluation failed: {e}")
        _capture(e)

    # Refresh the earnings calendar once per day from FMP -> Finnhub (bulk; no
    # per-symbol yfinance over the full universe). Best-effort, throttled so it
    # runs on only one of the day's scheduled scans.
    try:
        _refresh_earnings()
    except Exception as e:
        print(f"[cron] earnings refresh failed: {e}")
        _capture(e)

    # Recompute the signal track record once per day (forward returns of past
    # snapshot candidates). Best-effort; never fail the scan run.
    try:
        _refresh_track_record()
    except Exception as e:
        print(f"[cron] track record refresh failed: {e}")
        _capture(e)

    # Recompute the signal leaderboard once per day (Strategy Lab: which signal
    # actually predicted forward moves). Best-effort; never fail the scan run.
    try:
        _refresh_signal_leaderboard()
    except Exception as e:
        print(f"[cron] signal leaderboard refresh failed: {e}")
        _capture(e)

    # Log + settle 15-min Kalshi BTC outcomes (training data for the model).
    # Best-effort; the dedicated btc_outcome_logger workflow runs this on a
    # 15-min cadence, this just captures extra windows at scan time.
    try:
        from analytics.btc_outcome_logger import run_btc_outcome_logger

        run_btc_outcome_logger()
    except Exception as e:
        print(f"[cron] btc outcome logger failed: {e}")
        _capture(e)

    # Score fired alerts against what happened next (per-alert scorecards).
    # Best-effort; the unscored-events query is naturally incremental.
    try:
        from analytics.alert_outcomes import score_pending_outcomes

        scored = score_pending_outcomes()
        if scored:
            print(f"[alert_outcomes] scored {scored} fired alert(s)")
    except Exception as e:
        print(f"[cron] alert outcome scoring failed: {e}")
        _capture(e)

    # Freeze the latest snapshot's HSF opportunities for calibration, so
    # signal-time records are collected by the deliberate cron pipeline rather
    # than only when a user opens the Market Brief. Idempotent per snapshot.
    try:
        from analytics.opportunity_freeze import freeze_latest_opportunities

        frozen = freeze_latest_opportunities()
        if frozen:
            print(f"[opportunity_freeze] froze {frozen} HSF opportunity(ies)")
    except Exception as e:
        print(f"[cron] opportunity freeze failed: {e}")
        _capture(e)

    # HSF intelligence alerts: detect opportunity state-changes between the two
    # latest snapshots and notify subscribed users. Owned by the background
    # pipeline (never a page render); fails independently.
    try:
        from analytics.alert_evaluation import run_intelligence_alert_evaluation

        m = run_intelligence_alert_evaluation()
        if m.get("events_detected"):
            print(f"[intelligence_alerts] events={m['events_detected']} "
                  f"delivered={m['delivered']} deduped={m['deduped']} failed={m['failed']}")
    except Exception as e:
        print(f"[cron] intelligence alert evaluation failed: {e}")
        _capture(e)

    # HSF intelligence alert QUALITY maturation: for each past alert, find the
    # first comparable subsequent HSF snapshot at/after each horizon and classify
    # follow-through (measurement only — never price, never tunes alert behavior).
    # Idempotent; owned by the background pipeline (never a page render).
    try:
        from analytics.alert_quality import mature_alert_outcomes

        matured = mature_alert_outcomes()
        if matured:
            print(f"[alert_quality] matured {matured} alert outcome(s)")
    except Exception as e:
        print(f"[cron] alert quality maturation failed: {e}")
        _capture(e)

    # HSF opportunity OUTCOME intelligence maturation: for every eligible frozen
    # opportunity (not only alerted ones), classify subsequent HSF state at each
    # horizon (HSF-state persistence only — never price, never tunes anything).
    # Idempotent, first-observation immutable; background-only.
    try:
        from analytics.opportunity_outcomes import mature_opportunity_outcomes

        om = mature_opportunity_outcomes()
        if om.get("matured"):
            print(f"[opportunity_outcomes] matured={om['matured']} "
                  f"pending={om['pending']} unavailable={om['unavailable']}")
    except Exception as e:
        print(f"[cron] opportunity outcome maturation failed: {e}")
        _capture(e)

    # Settle immutable fired-signal rows with 1/3/5D return plus MFE/MAE.
    try:
        from analytics.signal_outcomes import score_pending_signal_outcomes

        scored = score_pending_signal_outcomes()
        if scored:
            print(f"[signal_outcomes] scored {scored} fired signal(s)")
    except Exception as e:
        print(f"[cron] signal outcome scoring failed: {e}")
        _capture(e)

    # Postmarket slots additionally send the evening wrap (throttled once/day).
    if session == "postmarket":
        try:
            from scheduler.evening_wrap import run_evening_wrap

            run_evening_wrap(force=os.getenv("CRON_FORCE", "").strip() == "1")
        except Exception as e:
            print(f"[cron] evening wrap failed: {e}")
            _capture(e)

    # Send the Pro+ morning digest once per day (throttled to the first scan run
    # of the day). Best-effort; never let email failures fail the scan run.
    try:
        from scheduler.morning_digest import run_morning_digest

        # A manual forced workflow run (CRON_FORCE=1) bypasses the daily throttle
        # so admins can test the digest on demand; scheduled runs send once/day.
        run_morning_digest(force=os.getenv("CRON_FORCE", "").strip() == "1")
    except Exception as e:
        print(f"[cron] morning digest failed: {e}")
        _capture(e)

    # Nightly-equivalent maintenance (throttled once/day). Best-effort.
    try:
        _purge_old_login_attempts()
    except Exception as e:
        print(f"[cron] login purge failed: {e}")
        _capture(e)
    try:
        _prune_old_runs()
    except Exception as e:
        print(f"[cron] runs prune failed: {e}")
        _capture(e)

    ok = all(run.ok for run in runs)
    _write_summary(
        {
            "started_at": started_at.isoformat(),
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "ok": ok,
            "skipped": False,
            "session": session,
            "runs": [asdict(run) for run in runs],
            "total_rows": sum(run.row_count for run in runs),
        }
    )

    if not ok:
        raise SystemExit(1)

    print("=== cron_runner complete ===")


if __name__ == "__main__":
    main()
