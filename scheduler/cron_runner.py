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
) -> ScanRunSummary:
    """Run one universe scan and save results to the configured database."""
    from db.runs import save_run
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

        save_run(
            name=run_name,
            label=universe,
            username=username,
            row_count=row_count,
            duration_sec=duration,
            results_json=_results_to_json(results),
            is_snapshot=False,
            allow_sqlite_fallback=False,
        )
        print(f"Saved {row_count} rows for {universe}.")
        return ScanRunSummary(
            universe=universe,
            ok=True,
            row_count=row_count,
            duration_sec=duration,
        )

    except Exception as e:
        print(f"ERROR running {universe} scan: {e}")
        return ScanRunSummary(universe=universe, ok=False, error=f"{type(e).__name__}: {e}")


def _print_provider_status() -> None:
    """Log which price provider will be used, to diagnose Alpaca config."""
    key = os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET_KEY")
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


def main():
    print("=== cron_runner started ===")
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
    universes = _configured_universes()
    print(f"Configured universes: {', '.join(universes)}")

    runs = [run_and_save(universe) for universe in universes]

    ok = all(run.ok for run in runs)
    _write_summary(
        {
            "started_at": started_at.isoformat(),
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "ok": ok,
            "skipped": False,
            "runs": [asdict(run) for run in runs],
            "total_rows": sum(run.row_count for run in runs),
        }
    )

    if not ok:
        raise SystemExit(1)

    print("=== cron_runner complete ===")


if __name__ == "__main__":
    main()
