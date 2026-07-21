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


def _load_universe(universe: str) -> list[str]:
    key = universe.strip().upper()
    sp500 = _read_symbols(ROOT / "sp500.txt")
    nasdaq = _read_symbols(ROOT / "nasdaq.txt")

    if key == "SP500":
        return sp500
    if key == "NASDAQ":
        limit = int(os.getenv("CRON_NASDAQ_LIMIT", "2000"))
        return nasdaq[:limit]
    if key == "COMBO":
        limit = int(os.getenv("CRON_NASDAQ_LIMIT", "2000"))
        return _dedupe([*sp500, *nasdaq[:limit]])
    raise ValueError(f"Unknown universe: {universe}")


def _results_to_json(results) -> str:
    if hasattr(results, "to_json"):
        return results.to_json(orient="records", date_format="iso")
    return json.dumps(results, default=str)


def _write_summary(summary: dict, path: Path | None = None) -> None:
    """Write a scheduled scan summary artifact for CI/deployment review."""
    target = path or Path(os.getenv("CRON_SUMMARY_PATH", str(SUMMARY_PATH)))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _skip_reason(now_utc: dt.datetime | None = None) -> str | None:
    """Return a reason to skip scheduled scans, or None when scans may run."""
    # CRON_FORCE=1 bypasses the weekend/premarket skip — used by the manual
    # "Run workflow" trigger so the pipeline can be tested any time.
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
    - CRON_UNIVERSES=SP500,NASDAQ,COMBO

    Blank entries are ignored. If the variable is not set, preserve the
    existing default behavior.
    """
    raw = os.getenv("CRON_UNIVERSES", "SP500,NASDAQ,COMBO")
    universes = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return universes or ["SP500", "NASDAQ", "COMBO"]


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
        tickers = _load_universe(universe)
        if not tickers:
            raise RuntimeError(f"No tickers loaded for {universe}")

        started = time.perf_counter()
        results = run_breakout_scan(
            tickers,
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
        row_count = len(results)

        run_name = f"{universe} | {row_count} results | {duration:.1f}s"
        print(f"Scan completed: {run_name}")

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

        # Promote to the day's snapshot so the track record + digest have a
        # reliable daily is_snapshot row (idempotent per universe/day).
        if save_snapshot:
            try:
                save_daily_snapshot(
                    universe, results_json, username=username,
                    row_count=row_count, duration_sec=duration,
                )
                print(f"Saved daily snapshot for {universe}.")
            except Exception as e:
                print(f"[cron] snapshot save failed for {universe}: {e}")
        return ScanRunSummary(
            universe=universe,
            ok=True,
            row_count=row_count,
            duration_sec=duration,
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
    session = _resolve_session()
    print(f"Session: {session}")
    if session in ("premarket", "postmarket"):
        runs = [_run_session_scan(session)]
    else:
        universes = _configured_universes()
        print(f"Configured universes: {', '.join(universes)}")
        # Regular slots also produce the day's snapshot (per-universe, idempotent).
        runs = [run_and_save(universe, save_snapshot=True) for universe in universes]

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
