"""
cron_runner.py
Run scheduled scans (SP500, NASDAQ, Combo) without Streamlit.
Designed for cron jobs, GitHub Actions, Railway CRON, or VPS crontab.
"""

import os
import datetime as dt

# --- Import your existing scan engine & DB functions ---
from scans.engine import run_breakout_scan
from db.runs import save_run


def run_and_save(universe: str, username: str = "cron"):
    """Run a scan for a single universe and save to Neon."""
    print(f"\n=== Running {universe} scan @ {dt.datetime.utcnow()} ===")
    try:
        results, duration = run_breakout_scan(universe=universe)
        row_count = len(results)

        run_name = f"{universe} | {row_count} results | {duration:.1f}s"
        print(f"Scan completed: {run_name}")

        save_run(
            name=run_name,
            label=universe,
            username=username,
            row_count=row_count,
            duration_sec=duration,
            results_json=results,
            is_snapshot=False,
        )
        print(f"Saved {row_count} rows to Neon for {universe}.")

    except Exception as e:
        print(f"ERROR running {universe} scan: {e}")


def main():
    print("=== cron_runner started ===")

    # Optional: block weekends
    today = dt.datetime.utcnow().weekday()  # Monday=0, Sunday=6
    if today >= 5:  # Saturday/Sunday
        print("Weekend detected — skipping scans.")
        return

    # Optional: block early premarket
    now_et_hour = dt.datetime.utcnow().hour - 5  # convert UTC → ET (rough)
    if now_et_hour < 6:
        print("Too early (premarket) — skipping scans.")
        return

    # --- Run the scans ---
    run_and_save("SP500")
    run_and_save("NASDAQ")
    run_and_save("COMBO")

    print("=== cron_runner complete ===")


if __name__ == "__main__":
    main()