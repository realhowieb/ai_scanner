"""Diagnostic: how much memory each of hsf-api's big caches holds against real data.

hsf-api (512 MB) climbs ~70 MB over a trading day and never gives it back. This replays
the work the API does over a day, in one process, and prints how much Python memory each
step keeps (tracemalloc) and the process RSS after it. Read-only: it only loads data the
API would load. Prints sizes and counts only, never tickers, users or rows (the
Diagnostics workflow's logs are public).

Run via the Diagnostics workflow: script = diag_api_memory.py
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_api_deps() -> None:
    try:
        import fastapi  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--prefer-binary", "-r",
                        str(ROOT / "api" / "requirements.txt")], check=True)


def rss_mb() -> float:
    try:
        with open("/proc/self/statm", "rb") as f:
            return round(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1048576, 1)
    except OSError:
        return -1.0


TRACE = os.environ.get("HSF_DIAG_NO_TRACE") != "1"


def traced_mb() -> float:
    if not TRACE:
        return 0.0
    return round(tracemalloc.get_traced_memory()[0] / 1048576, 1)


def cache_table() -> list:
    from api.today import TTLCache

    out = []
    with TTLCache._instances_lock:
        caches = list(TTLCache._instances)
    for c in caches:
        with c._lock:
            kinds: dict = {}
            for k in c._items:
                kind = k[0] if isinstance(k, tuple) and k else k
                kinds[str(kind)[:24]] = kinds.get(str(kind)[:24], 0) + 1
        out.append((c.name, len(c._items), c.max_entries, kinds))
    return out


def step(name: str, fn) -> None:
    gc.collect()
    if TRACE:
        tracemalloc.reset_peak()
    before_t = traced_mb()
    started = time.monotonic()
    err = ""
    try:
        fn()
    except Exception as e:  # keep going; report the type only
        err = f"  ERROR {type(e).__name__}"
    peak = round(tracemalloc.get_traced_memory()[1] / 1048576, 1) if TRACE else 0.0
    gc.collect()
    after_t, after_r = traced_mb(), rss_mb()
    print(f"{name:<44} kept {after_t - before_t:+8.1f} MB  peak +{peak - before_t:7.1f}  (python total {after_t:7.1f})  "
          f"rss {after_r:7.1f}  {time.monotonic() - started:6.1f}s{err}", flush=True)


def main() -> None:
    _ensure_api_deps()
    os.environ.setdefault("RENDER", "")  # no background threads
    if TRACE:
        tracemalloc.start(1)
    print("=" * 110)
    print("hsf-api cache memory replay (MB kept after each step, after gc)" + ("" if TRACE else " -- RSS only, no tracing"))
    print("=" * 110)
    step("import api.main", lambda: __import__("api.main"))

    from api import history, market, scans, today

    runs: list = []

    def load_runs() -> None:
        runs.extend(today.market_runs())
        print(f"    market runs listed: {len(runs)}")

    step("market_runs() (60 runs with snapshots)", load_runs)
    def import_brief_libs() -> None:
        import ml_prebreakout  # noqa: F401
        import ui.market_brief  # noqa: F401
        import ui.opportunities  # noqa: F401

    step("import Brief modules (ML, scipy...)", import_brief_libs)

    def picks_only() -> None:
        from scheduler.morning_digest import _prebreakout_picks
        from ui.market_brief import _brief_scan_df

        _prebreakout_picks(_brief_scan_df())

    step("PreBreakout picks alone (scan + ML score)", picks_only)
    step("Market Brief (_brief_core)", market._brief_core)
    step("Day Trader movers", lambda: market.day_trader("movers"))
    for i in (2, 3):
        today._cache.clear_key("brief")
        step(f"Market Brief rebuild #{i}", market._brief_core)
    for i in (2, 3):
        for k in [k for k in list(today._cache._items) if isinstance(k, tuple) and k[0] == "dt_rows"]:
            today._cache.clear_key(k)
        today._cache.clear_key(("dt_source", "movers"))
        step(f"Day Trader movers rebuild #{i}", lambda: market.day_trader("movers"))
    step("warm_stock_pages", scans.warm_stock_pages)
    step("track_record", history.track_record)
    for n in (1, 4, 8):
        ids = [int(r["id"]) for r in runs[:n] if r.get("id") is not None]
        step(f"run_df for latest {n} runs", lambda ids=ids: [today.run_df(i) for i in ids])
    ids24 = [int(r["id"]) for r in runs[:scans.SCORE_HISTORY_RUNS] if r.get("id") is not None]
    step("score history (24 runs)", lambda: [scans._run_ticker_scores(i) for i in ids24])

    def outcomes_summary() -> None:
        from api import outcomes

        outcomes.summary({}, 5, "pct")

    step("Outcome Intelligence dataset + summary", outcomes_summary)

    print("-" * 110)
    for name, n, cap, kinds in cache_table():
        if n:
            print(f"cache {name:<18} {n:4d}/{cap:<4d} {kinds}")
    print("-" * 110)
    snap = tracemalloc.take_snapshot() if TRACE else None
    print("Top 15 allocation sites still held:")
    for s in (snap.statistics("filename")[:15] if snap else []):
        fn = s.traceback[0].filename
        fn = fn.split("site-packages/")[-1] if "site-packages/" in fn else os.path.relpath(fn, ROOT)
        print(f"  {s.size / 1048576:8.1f} MB  {s.count:9d} blocks  {fn}")

    def clear_all() -> None:
        today.clear_cache()
        scans.score_cache.clear()
        scans.spark_cache.clear()
        scans.quote_cache.clear()

    step("clear every cache", clear_all)
    print(f"python memory left after clearing: {traced_mb()} MB, rss {rss_mb()} MB "
          "(the gap is allocator memory the process keeps)")


if __name__ == "__main__":
    main()
    if TRACE:  # tracing inflates RSS; replay once more untraced for true RSS figures
        sys.stdout.flush()
        subprocess.run([sys.executable, __file__], env={**os.environ, "HSF_DIAG_NO_TRACE": "1"})
        sys.stdout.flush()
        print("\n### same, with numeric libraries limited to one thread")
        sys.stdout.flush()
        one = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        subprocess.run([sys.executable, __file__], env={**os.environ, "HSF_DIAG_NO_TRACE": "1", **one})
