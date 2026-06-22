"""Run scheduled market scans without the Streamlit UI."""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


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
) -> bool:
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
        return True

    except Exception as e:
        print(f"ERROR running {universe} scan: {e}")
        return False


def main():
    print("=== cron_runner started ===")

    # Optional: block weekends
    today = dt.datetime.now(dt.timezone.utc).weekday()  # Monday=0, Sunday=6
    if today >= 5:  # Saturday/Sunday
        print("Weekend detected — skipping scans.")
        return

    # Optional: block early premarket
    now_et_hour = dt.datetime.now(dt.timezone.utc).hour - 5  # convert UTC to ET, roughly.
    if now_et_hour < 6:
        print("Too early (premarket) — skipping scans.")
        return

    # --- Run the scans ---
    results = [
        run_and_save("SP500"),
        run_and_save("NASDAQ"),
        run_and_save("COMBO"),
    ]
    if not all(results):
        raise SystemExit(1)

    print("=== cron_runner complete ===")


if __name__ == "__main__":
    main()
