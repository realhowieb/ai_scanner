"""Market pages for API clients: earnings (P1-68), Market Brief (P1-69) and
the Day Trader snapshot (P1-70). Each reuses the web page's data code; plan
gates are applied by the routes in api.main.
"""
from __future__ import annotations

import datetime as dt
import re
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


# ---- Market Brief (P1-69) ------------------------------------------------------------------------
BRIEF_TTL_S = 300   # the web caches the brief for 5 minutes too


def _brief_core() -> Optional[Dict[str, Any]]:
    """The user-independent brief (ui.market_brief._compute_brief, the web's builder)
    plus opportunities compared with the previous snapshot, read-only: the API never
    writes opportunity snapshots or research rows (the web page does that)."""
    from api.today import _cached

    def load():
        from ui.market_brief import _base_ticker, _compute_brief, _market_phase

        data = _compute_brief()
        if not data:
            return None
        compared: List[Dict[str, Any]] = []
        has_previous = False
        try:
            from ui.opportunities import build_opportunities, compare_opportunities

            opps = build_opportunities(data, base_ticker=_base_ticker, top_n=5)
            previous = None
            try:
                from db.opportunity_snapshots import load_previous_opportunity_snapshot

                prev = load_previous_opportunity_snapshot(data.get("snapshot_time"), context="market_brief")
                previous = prev.get("opportunities") if prev else None
            except Exception:
                previous = None
            has_previous = previous is not None
            compared = compare_opportunities(opps, previous)
        except Exception:
            compared = []
        return {"data": data, "compared": compared, "has_previous": has_previous, "phase": _market_phase()}

    return _cached("brief", load, ttl_s=BRIEF_TTL_S)


_EARN_FLAG = re.compile(r"^\s*(\S+)\s*(?:⚠️\s*E(\d+)d)?\s*$")


def _split_flag(label: Any) -> tuple:
    """'MSFT ⚠️E3d' (the digest's earnings flag) -> ('MSFT', 3); 'MSFT' -> ('MSFT', None)."""
    m = _EARN_FLAG.match(str(label or ""))
    if not m:
        return str(label or "").strip(), None
    return m.group(1), int(m.group(2)) if m.group(2) else None


def brief(entitlements: Dict[str, bool]) -> Dict[str, Any]:
    from ui.entitlement_view import redact_prebreakout_rows

    core = _brief_core()
    if not core:
        return {"available": False, "snapshot_time": None}
    d = core["data"]
    early = bool(entitlements.get("can_early_breakout"))
    picks = []
    for p in (d.get("picks") or []) if early else []:
        ticker, edays = _split_flag(p.get("symbol"))
        picks.append({**{k: v for k, v in p.items() if k != "symbol"}, "ticker": ticker, "earnings_days": edays})
    gappers = []
    for g in d.get("gappers") or []:
        ticker, edays = _split_flag(g.get("ticker"))
        gappers.append({"ticker": ticker, "last": g.get("last"), "chg_pct": g.get("chg_pct"),
                        "gap_pct": g.get("gap_pct"), "earnings_days": edays})
    breadth = d.get("breadth")
    return {
        "available": True,
        "snapshot_time": d.get("snapshot_time"),
        "phase": core["phase"],
        "market": [{"label": lbl, "last": last, "chg_pct": chg} for lbl, last, chg in d.get("market_close") or []],
        "breadth": {"advancers": breadth[0], "decliners": breadth[1]} if breadth else None,
        "sectors": [{"sector": s, "chg_pct": c} for s, c in d.get("sectors") or []],
        "opportunities": redact_prebreakout_rows(core["compared"], allowed=early),
        "has_previous_snapshot": core["has_previous"],
        "gappers": gappers,
        "gainers": [{"ticker": t, "chg_pct": c} for t, c in d.get("gainers") or []],
        "losers": [{"ticker": t, "chg_pct": c} for t, c in d.get("losers") or []],
        "golden_crosses": list(d.get("golden") or []),
        "top_breakout_scores": [{"ticker": t, "score": s} for t, s in d.get("top_setups") or []],
        "prebreakout_picks": picks,
        "prebreakout_locked": not early,
        "earnings_today": list(d.get("earnings_today") or []),
    }
