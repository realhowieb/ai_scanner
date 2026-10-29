"""The signed-out stock page behind a shared link (/share/{ticker} on the web).

Shows only what a link preview needs: name, price and day change, and, when the stock is
in the latest scan's ranked setups, its HSF Score and setup type with the first reason.
Everything else (signals, risks, lifecycle, history) stays behind sign-up and is only
counted. Unknown symbols are refused so the page can't be used to fill the stock cache
with junk.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

RANKED_MIN = 40  # same floor the web uses for "ranked" (web/src/components/ui.tsx RANKED_MIN)


class UnknownTicker(Exception):
    pass


def public_stock(ticker: str) -> Dict[str, Any]:
    from api import symbol_search
    from api.scans import stock_detail

    t = ticker.strip().upper()
    try:
        known = symbol_search.lookup(t)
        directory_up = bool(symbol_search._rows())
    except Exception:
        known, directory_up = None, False
    if known is None and directory_up:
        raise UnknownTicker(t)
    d = stock_detail(t, {})
    if known is None and not d.get("in_latest_scan"):
        raise UnknownTicker(t)
    score: Optional[float] = d.get("hsf_score")
    ranked = bool(d.get("has_setup") and d.get("in_latest_scan") and score is not None and score >= RANKED_MIN)
    reasons = list(d.get("reasons") or [])
    hidden = len(reasons[1:]) + len(d.get("signals") or []) + len(d.get("risks") or []) if ranked else 0
    price, change = d.get("price"), d.get("change_pct")
    if price is None:
        # Not in the scan (or no scan price): fall back to the latest trade, as Watchlists do.
        try:
            from api.scans import latest_quotes

            q = (latest_quotes([t]).get("quotes") or {}).get(t) or {}
            price, change = q.get("last"), q.get("chg_pct")
        except Exception:
            pass
    return {
        "ticker": t,
        "name": (known or {}).get("name"),
        "price": price,
        "change_pct": change,
        "scan_at": d.get("scan_at"),
        "ranked": ranked,
        "hsf_score": round(float(score)) if ranked and score is not None else None,
        "setup": d.get("primary_setup") if ranked else None,
        "reason": reasons[0] if ranked and reasons else None,
        "hidden_insights": hidden,
    }


PREVIEW_COUNT = 3


def public_top_setups(count: int = PREVIEW_COUNT) -> Dict[str, Any]:
    """The landing page's live preview: the latest scan's top few ranked setups (the same
    ticker, score and setup a shared link already shows), and how many more ranked setups
    a free account sees. Signals, risks, reasons and the rest of the list need an account."""
    from api.scans import latest_scan, max_results_for

    data = latest_scan({}, "basic", limit=count, offset=0, min_score=RANKED_MIN)
    rows = [{"ticker": r["ticker"], "hsf_score": round(float(r["score"])) if r.get("score") is not None else None,
             "setup": r.get("primary_setup"), "change_pct": r.get("chg_pct")}
            for r in data.get("setups") or [] if r.get("ticker")]
    ranked = int(data.get("total") or 0)
    return {"scan_at": data.get("scan_at"), "stale": bool(data.get("stale")), "setups": rows,
            "ranked_total": ranked, "free_rows": min(ranked, max_results_for("basic"))}
