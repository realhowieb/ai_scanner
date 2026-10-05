#!/usr/bin/env python3
"""P2-35 step 1: replay one session's 1-minute bars through the Stair-stepper
thresholds and report how often and how steadily names qualify.

Read-only: reads the latest market scan (for the default symbol list) and
Alpaca 1-minute bars; writes only the report files. Changes no defaults.

    python scripts/stair_stepper_threshold_check.py                 # last completed session
    python scripts/stair_stepper_threshold_check.py --date 2026-10-05 --symbols NVDA,AMD
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics import market_calendar as mc  # noqa: E402
from analytics.stair_step import DEFAULT_WINDOW, WINDOW_OPTIONS  # noqa: E402
from analytics.stair_step_threshold_check import build_report, render_markdown  # noqa: E402

MAX_SYMBOLS = 100


def last_completed_session(now: dt.datetime) -> dt.date:
    day = now.astimezone(mc.ET).date()
    if mc.is_trading_day(day) and now >= mc.session_bounds_utc(day)[1]:
        return day
    return mc.previous_trading_day(day)


def default_symbols(top_n: int) -> List[str]:
    """Top HSF setups of the latest market scan, ranked as in the Scanner."""
    from db.runs import list_runs, load_run_results
    from ui.app_runtime import normalize_results_to_df
    from ui.market_scans import MARKET_USER, market_runs, top_setups

    runs = market_runs(list_runs(limit=25, include_snapshots=True, username=MARKET_USER) or [])
    if not runs:
        return []
    df = normalize_results_to_df(load_run_results(int(runs[0]["id"])))
    return [o["ticker"] for o in top_setups(df, n=top_n)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--date", help="ET session date (YYYY-MM-DD); default: last completed session")
    p.add_argument("--symbols", default="", help="Comma-separated; default: top setups of the latest scan")
    p.add_argument("--top", type=int, default=40, help="Symbols from the latest scan when --symbols is empty")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW, choices=WINDOW_OPTIONS)
    p.add_argument("--out", type=Path, default=Path("artifacts/automation/stair_stepper"))
    args = p.parse_args()

    day = dt.date.fromisoformat(args.date) if args.date else last_completed_session(dt.datetime.now(dt.timezone.utc))
    if not mc.is_trading_day(day):
        print(f"{day} is not a trading day.")
        return 1
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or default_symbols(args.top)
    symbols = list(dict.fromkeys(symbols))[:MAX_SYMBOLS]
    if not symbols:
        print("No symbols: pass --symbols or make sure a market scan exists.")
        return 1

    from data.price_alpaca import fetch_minute_bars_multi, get_alpaca_data_feed

    open_utc, close_utc = mc.session_bounds_utc(day)
    bars = fetch_minute_bars_multi(symbols, open_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                   close_utc.strftime("%Y-%m-%dT%H:%M:%SZ"))
    report = build_report({s: bars.get(s, []) for s in symbols}, (open_utc, close_utc), window=args.window)
    report["data_feed"] = get_alpaca_data_feed()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stair_stepper_threshold_check.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    md = render_markdown(report) + f"\nData feed: {report['data_feed']}.\n"
    (args.out / "stair_stepper_threshold_check.md").write_text(md, encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
