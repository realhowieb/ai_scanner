"""Score every watchlist stock the market scan didn't rank, once per scheduled scan
(Premium auto-score on the Scanner's Watchlist tab and the Watchlists page).

Runs after the market scan in scheduler/cron_runner.py: the tickers saved on any
watchlist, minus the latest scan's ranked setups, are scored with the same screen-free
scan as the earnings auto-score and saved to db.watchlist_autoscore.

Best-effort: never raises. HSF_WATCHLIST_AUTOSCORE=0 turns it off.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional, Set

MAX_SYMBOLS = 400   # most-watched first when there are more


def enabled() -> bool:
    return os.getenv("HSF_WATCHLIST_AUTOSCORE", "1").strip().lower() not in ("0", "false", "no", "off")


def ranked_tickers() -> Set[str]:
    """Tickers among the latest market scan's ranked setups (they already have a score)."""
    from api.scans import market_runs, run_opportunities

    runs = market_runs()
    if not runs:
        return set()
    return {str(o.get("ticker") or "").strip().upper() for o in run_opportunities(int(runs[0]["id"]))}


def tickers_to_score(limit: int = MAX_SYMBOLS) -> List[str]:
    from db.watchlist_autoscore import watched_tickers

    ranked = ranked_tickers()
    # Ask for the cap plus the ranked count, so ranked names don't eat into the cap.
    return [t for t in watched_tickers(limit + len(ranked)) if t not in ranked][:limit]


def run_watchlist_autoscore(session: str = "regular") -> Optional[int]:
    """Score unranked watchlist stocks and save the pass. Returns its id, or None."""
    if not enabled():
        print("[watchlist_autoscore] off (HSF_WATCHLIST_AUTOSCORE=0)")
        return None
    try:
        tickers = tickers_to_score()
        if not tickers:
            print("[watchlist_autoscore] no unranked watchlist stocks; nothing to score")
            return None
        from scheduler.earnings_autoscore import scan_setups

        started = time.perf_counter()
        setups = scan_setups(tickers, premarket=session == "premarket")
        duration = round(time.perf_counter() - started, 1)
        from db.watchlist_autoscore import save

        new_id = save(setups, session=session, symbols=len(tickers), duration_s=duration)
        print(f"[watchlist_autoscore] scored {len(setups)} of {len(tickers)} unranked watchlist stocks "
              f"({session}) in {duration}s; saved id={new_id}")
        return new_id
    except Exception as e:
        print(f"[watchlist_autoscore] failed: {type(e).__name__}")
        try:
            import sentry_sdk

            sentry_sdk.capture_exception(e)
        except Exception:
            pass
        return None
