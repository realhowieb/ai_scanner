"""Market pages for API clients: earnings (P1-68), Market Brief (P1-69) and
the Day Trader snapshot (P1-70). Each reuses the web page's data code; plan
gates are applied by the routes in api.main.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Sequence

MAX_EARNINGS_DAYS = 30


def earnings(days: int, tickers: Optional[Sequence[str]] = None,
             today: Optional[dt.date] = None) -> List[Dict[str, Any]]:
    """Upcoming earnings in the next `days` days (DB only, like the web's
    'Earnings this week' panel), optionally only for `tickers`."""
    from db.earnings import fetch_earnings_this_week

    today = today or dt.datetime.now(dt.timezone.utc).date()
    wanted = {t.strip().upper().replace(".", "-") for t in tickers or [] if t.strip()}
    out = []
    for r in fetch_earnings_this_week(days_ahead=int(days)) or []:
        sym = str(r.get("symbol") or "").strip().upper()
        if not sym or (wanted and sym not in wanted and sym.replace("-", ".") not in wanted):
            continue
        d = r.get("earnings_date")
        if isinstance(d, dt.datetime):
            d = d.date()
        out.append({"ticker": sym, "earnings_date": d.isoformat() if isinstance(d, dt.date) else None,
                    "days_until": (d - today).days if isinstance(d, dt.date) else None,
                    "time": (str(r.get("earnings_time")).strip().lower() or None) if r.get("earnings_time") else None})
    out.sort(key=lambda x: (x["days_until"] if x["days_until"] is not None else 10_000, x["ticker"]))
    return out
