"""Score every upcoming earnings reporter once per scheduled scan (Premium auto-score).

Runs after the market scan in scheduler/cron_runner.py: every name reporting from
today through the next 7 days (ET) is scored with the Earnings tab's "Score all"
scan (screens off) and the rows are saved to db.earnings_autoscore. The API serves
them to Premium on the Earnings tab's "All reporters" view, and the earnings
email sections reuse a fresh pass instead of scoring the same names again.

Best-effort: never raises. HSF_EARNINGS_AUTOSCORE=0 turns it off.
"""
from __future__ import annotations

import datetime as dt
import os
import time
from typing import Any, Dict, List, Optional

WINDOW_DAYS = 7
MAX_SYMBOLS = 400   # a heavy earnings week; nearest days are kept first


def enabled() -> bool:
    return os.getenv("HSF_EARNINGS_AUTOSCORE", "1").strip().lower() not in ("0", "false", "no", "off")


def et_today(now: Optional[dt.datetime] = None) -> dt.date:
    from zoneinfo import ZoneInfo

    return (now or dt.datetime.now(dt.timezone.utc)).astimezone(ZoneInfo("America/New_York")).date()


def window_tickers(start: dt.date, end: dt.date, limit: int = MAX_SYMBOLS) -> List[str]:
    """Reporters between start and end (inclusive), soonest first, at most `limit`."""
    from scheduler.earnings_email import reporters

    rows = sorted(reporters(start, end), key=lambda r: (r["date"], r["ticker"]))
    return [r["ticker"] for r in rows][:limit]


def scan_setups(tickers: List[str], *, premarket: bool = False) -> List[Dict[str, Any]]:
    """Scanner rows (api.scans._scan_row shape, strict JSON) for `tickers` from one scan
    with the screens off, best score first. Raises on scan failure."""
    from api.scans import _scan_row, json_safe
    from scan.engine import run_breakout_scan
    from scan.execution import run_manual_scan_execution
    from ui.market_scans import top_setups
    from ui.scan_providers import apply_alpaca_extended_prices, sanitize_universe_symbols

    tickers = sanitize_universe_symbols(list(dict.fromkeys(t for t in tickers if t)))
    if not tickers:
        return []
    df = run_manual_scan_execution(
        runner=run_breakout_scan, tickers=tickers, premarket=premarket, afterhours=False,
        unusual_volume=False, min_gap=0.0, min_price=0.0, max_price=1_000_000.0,
        top_n=len(tickers), profile="regular", apply_gap_filter=False, diagnostics=False,
        extended_price_transform=apply_alpaca_extended_prices, min_dollar_vol=0.0,
    )
    rows, seen = [], set()
    for o in top_setups(df, n=100_000):
        row = _scan_row(o)
        key = str(row.get("ticker") or "").strip().upper()
        if key and key not in seen:
            seen.add(key)
            rows.append(row)
    return json_safe(rows)


def run_earnings_autoscore(session: str = "regular", *, now: Optional[dt.datetime] = None) -> Optional[int]:
    """Score this week's reporters and save the pass. Returns its id, or None."""
    if not enabled():
        print("[earnings_autoscore] off (HSF_EARNINGS_AUTOSCORE=0)")
        return None
    try:
        start = et_today(now)
        end = start + dt.timedelta(days=WINDOW_DAYS)
        tickers = window_tickers(start, end)
        if not tickers:
            print("[earnings_autoscore] no reporters in the window; nothing to score")
            return None
        started = time.perf_counter()
        setups = scan_setups(tickers, premarket=session == "premarket")
        duration = round(time.perf_counter() - started, 1)
        from db.earnings_autoscore import save

        new_id = save(setups, session=session, window_start=start, window_end=end,
                      symbols=len(tickers), duration_s=duration)
        print(f"[earnings_autoscore] scored {len(setups)} of {len(tickers)} reporters "
              f"({start}..{end}, {session}) in {duration}s; saved id={new_id}")
        return new_id
    except Exception as e:
        print(f"[earnings_autoscore] failed: {type(e).__name__}")
        try:
            import sentry_sdk

            sentry_sdk.capture_exception(e)
        except Exception:
            pass
        return None


def fresh_scores(max_age_s: int = 45 * 60, *, premarket: bool = False) -> Optional[Dict[str, Dict[str, Any]]]:
    """{ticker key: row} from a pass saved in the last max_age_s for the same session
    kind, or None (then the caller scores for itself)."""
    try:
        from db.earnings_autoscore import latest_stamp, load_setups

        stamp = latest_stamp(max_age_s)
        if not stamp or (stamp.get("session") == "premarket") != premarket:
            return None
        out: Dict[str, Dict[str, Any]] = {}
        for row in load_setups(int(stamp["id"])):
            key = "".join(ch for ch in str(row.get("ticker") or "").upper() if ch.isalnum())
            if key and key not in out:
                out[key] = row
        return out or None
    except Exception:
        return None
