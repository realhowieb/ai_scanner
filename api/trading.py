"""Journal, trade plans and Alpaca paper trading for API clients (P1-72).

Same rules as the web:
  * Journal (db.trades): every plan can read its journal; logging, closing and
    deleting trades come from the trade plan, which is Pro on the web (ui.results
    locks it for Free), so writes are Pro+.
  * Trade plan (ui.trade_plan.build_trade_plan): Pro+.
  * Paper trading (Premium, can_paper_trade): each user connects their OWN Alpaca
    paper keys; they are validated against the paper endpoint and stored
    encrypted (db.paper_trading / db.secret_box, needs APP_ENCRYPTION_KEY) and
    never returned. Every order needs an explicit confirm=true (the web's
    confirmation form) and is a whole-share market buy, imported into the journal.

Extra guard for the API: trading calls are refused unless the trading endpoint
(data.alpaca_trading._base_url) is Alpaca's paper host, so a misconfigured
ALPACA_BASE_URL can never route a customer's order to a live account.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.paper_orders import (
    OrderConflict,
    PaperRejected,
    PaperUnavailable,
    reconcile_pending,
    require_paper_endpoint,
)

PAPER_HOST = "paper-api.alpaca.markets"
MAX_ORDER_QTY = 100_000


class TradeClosed(ValueError):
    """The journal trade is already closed (409)."""


# ---- journal --------------------------------------------------------------------------------------
def _quotes(tickers: List[str]) -> Dict[str, float]:
    try:
        from market_data import get_latest_quotes

        quotes = get_latest_quotes(sorted(set(tickers))) or {}
        return {s: float(q["last"]) for s, q in quotes.items() if isinstance(q, dict) and q.get("last") is not None}
    except Exception:
        return {}


def journal(username: str, limit: int = 200, offset: int = 0) -> Dict[str, Any]:
    """Trades (open first), open ones marked to live quotes, plus closed-trade stats."""
    from db.trades import journal_stats, list_trades

    _reconcile(username)
    page = list_trades(username, limit=limit + 1, offset=offset)
    trades = page[:limit]
    live = _quotes([t["ticker"] for t in trades if not t.get("closed_at")])
    out = []
    for t in trades:
        entry = float(t.get("entry_price") or 0)
        shares = int(t.get("shares") or 0)
        is_open = not t.get("closed_at")
        mark = live.get(t["ticker"]) if is_open else (float(t["exit_price"]) if t.get("exit_price") is not None else None)
        pnl = (mark - entry) * shares if mark is not None and entry else None
        pct = (mark - entry) / entry * 100.0 if mark is not None and entry else None
        out.append({**t, "open": is_open, "mark": mark, "pnl": pnl, "pnl_pct": pct})
    stats = journal_stats(username)
    open_trades = [t for t in out if t.get("open")]
    closed = [t for t in out if not t.get("open")]
    return {"trades": out, "stats": stats, "limit": limit, "offset": offset, "has_more": len(page) > limit,
            "summary": {"open": len(open_trades), "closed": len(closed),
                        "win_rate": (100.0 * stats["wins"] / stats["closed"]) if stats and stats.get("closed") else None,
                        "avg_return_pct": (stats or {}).get("avg_return_pct"),
                        "open_pnl": sum(float(t.get("pnl") or 0) for t in open_trades if t.get("pnl") is not None),
                        "total_pnl": sum(float(t.get("pnl") or 0) for t in out if t.get("pnl") is not None)}}


def _owned_trade(username: str, trade_id: int) -> Optional[Dict[str, Any]]:
    from db.trades import get_trade

    return get_trade(username, trade_id)


def _entry_score(ticker: str) -> Optional[float]:
    """The ticker's HSF Score in the latest market scan (None when it isn't a ranked setup),
    stored with the trade so journal insights can group results by score at entry."""
    try:
        from api.scans import watchlist_scan_state

        row = watchlist_scan_state([ticker], {})["rows"].get(ticker.upper()) or {}
        score = row.get("score")
        return float(score) if score is not None else None
    except Exception:
        return None


def log(username: str, ticker: str, entry_price: float, shares: int) -> None:
    from db.trades import log_trade

    log_trade(username, ticker, float(entry_price), int(shares), source="api", hsf_score=_entry_score(ticker))


def close(username: str, trade_id: int, exit_price: float) -> bool:
    from db.trades import close_trade

    t = _owned_trade(username, trade_id)
    if t is None:
        return False
    if t.get("closed_at"):
        raise TradeClosed("This trade is already closed.")
    close_trade(int(trade_id), username, float(exit_price))
    return True


def delete(username: str, trade_id: int) -> bool:
    from db.trades import delete_trade

    if _owned_trade(username, trade_id) is None:
        return False
    delete_trade(int(trade_id), username)
    return True


# ---- journal insights -----------------------------------------------------------------------------
SCORE_BANDS = (("80+", 80.0), ("60-79", 60.0), ("Under 60", float("-inf")))
HOLD_BANDS = (("Same day", 1), ("1-5 days", 5), ("6-20 days", 20), ("Over 20 days", None))  # upper bound, days
SOURCE_LABELS = {"api": "Logged by hand", "scan": "From a scan", "paper": "Paper trading", "manual": "Logged by hand"}


def _ts(v: Any):
    import datetime as dt

    if isinstance(v, dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    try:
        t = dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _score_band(t: Dict[str, Any]) -> str:
    s = t.get("hsf_score")
    if s is None:
        return "Not recorded"
    return next(label for label, floor in SCORE_BANDS if float(s) >= floor)


def _hold_band(days: Optional[float]) -> str:
    if days is None:
        return "Unknown"
    for label, top in HOLD_BANDS:
        if top is None or (days < top if label == "Same day" else days <= top):
            return label
    return HOLD_BANDS[-1][0]


def _group(rows: List[Dict[str, Any]], key, order: List[str]) -> List[Dict[str, Any]]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(key(r), []).append(r)
    out = []
    for label in [o for o in order if o in by] + sorted(k for k in by if k not in order):
        g = by[label]
        wins = sum(1 for r in g if r["return_pct"] > 0)
        out.append({"label": label, "count": len(g), "wins": wins, "win_rate": round(100.0 * wins / len(g), 1),
                    "avg_return_pct": round(sum(r["return_pct"] for r in g) / len(g), 2),
                    "pnl": round(sum(r["pnl"] for r in g), 2)})
    return out


def insights(username: str) -> Dict[str, Any]:
    """Closed-trade analytics: win/loss profile, a running P&L curve, and results by score at
    entry, by holding time and by where the trade came from. Closed trades only (open
    ones have no result yet); a trade without a recorded score reads 'Not recorded'."""
    from db.trades import closed_trades

    rows = []
    for t in closed_trades(username):
        entry, exit_, shares = float(t.get("entry_price") or 0), t.get("exit_price"), int(t.get("shares") or 0)
        if not entry or exit_ is None:
            continue
        exit_ = float(exit_)
        opened, closed = _ts(t.get("entered_at")), _ts(t.get("closed_at"))
        hold = (closed - opened).total_seconds() / 86400.0 if opened and closed else None
        rows.append({"ticker": t.get("ticker"), "pnl": (exit_ - entry) * shares,
                     "return_pct": (exit_ - entry) / entry * 100.0, "closed_at": closed.isoformat() if closed else None,
                     "hold_days": hold, "source": str(t.get("source") or "api"), "hsf_score": t.get("hsf_score")})
    out: Dict[str, Any] = {"closed": len(rows), "summary": None, "curve": [], "by_score": [], "by_hold": [],
                           "by_source": []}
    if not rows:
        return out
    wins = [r for r in rows if r["return_pct"] > 0]
    losses = [r for r in rows if r["return_pct"] < 0]
    gross_win = sum(r["pnl"] for r in rows if r["pnl"] > 0)
    gross_loss = -sum(r["pnl"] for r in rows if r["pnl"] < 0)
    holds = [r["hold_days"] for r in rows if r["hold_days"] is not None]
    best = max(rows, key=lambda r: r["return_pct"])
    worst = min(rows, key=lambda r: r["return_pct"])
    out["summary"] = {
        "wins": len(wins), "losses": len(losses), "win_rate": round(100.0 * len(wins) / len(rows), 1),
        "avg_return_pct": round(sum(r["return_pct"] for r in rows) / len(rows), 2),
        "avg_win_pct": round(sum(r["return_pct"] for r in wins) / len(wins), 2) if wins else None,
        "avg_loss_pct": round(sum(r["return_pct"] for r in losses) / len(losses), 2) if losses else None,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "total_pnl": round(sum(r["pnl"] for r in rows), 2),
        "avg_hold_days": round(sum(holds) / len(holds), 1) if holds else None,
        "best": {"ticker": best["ticker"], "return_pct": round(best["return_pct"], 2)},
        "worst": {"ticker": worst["ticker"], "return_pct": round(worst["return_pct"], 2)},
    }
    running = 0.0
    for r in rows:
        running += r["pnl"]
        out["curve"].append({"closed_at": r["closed_at"], "ticker": r["ticker"], "pnl": round(r["pnl"], 2),
                             "cumulative_pnl": round(running, 2)})
    out["by_score"] = _group(rows, _score_band, [b[0] for b in SCORE_BANDS] + ["Not recorded"])
    out["by_hold"] = _group(rows, lambda r: _hold_band(r["hold_days"]), [b[0] for b in HOLD_BANDS] + ["Unknown"])
    out["by_source"] = _group(rows, lambda r: SOURCE_LABELS.get(r["source"], r["source"].title()),
                              ["Logged by hand", "From a scan", "Paper trading"])
    return out


# ---- trade plan -----------------------------------------------------------------------------------
def latest_row(ticker: str) -> Optional[Dict[str, Any]]:
    """The ticker's row in the latest market scan (the web plans from scan rows)."""
    from api.today import market_runs, run_df

    runs = market_runs()
    df = run_df(int(runs[0]["id"])) if runs else None
    if df is None or "Ticker" not in getattr(df, "columns", []):
        return None
    rows = df[df["Ticker"].astype(str).str.strip().str.upper() == ticker]
    return None if rows.empty else rows.iloc[0].to_dict()


def plan(ticker: str, account_size: float, risk_pct: float) -> Optional[Dict[str, Any]]:
    from ui.trade_plan import build_trade_plan

    row = latest_row(ticker)
    if row is None:
        return None
    p = build_trade_plan(row, account_size=account_size, risk_pct=risk_pct)
    return None if p is None else {"ticker": ticker, **p}


# ---- paper trading --------------------------------------------------------------------------------
_require_paper_endpoint = require_paper_endpoint


def _require_encryption() -> None:
    from db.secret_box import encryption_available

    if not encryption_available():
        raise PaperUnavailable("Paper trading is unavailable right now (secure key storage isn't configured).")


def paper_status(username: str) -> Dict[str, Any]:
    from db.paper_trading import account_meta, get_paper_account

    meta = account_meta(username)
    if not meta:
        return {"connected": False}
    out: Dict[str, Any] = {"connected": True, "connected_at": meta.get("connected_at"), "account": None}
    acct = get_paper_account(username)
    if acct:
        _require_paper_endpoint()
        from data.alpaca_trading import get_account

        summary = get_account(acct["api_key"], acct["api_secret"])
        if summary:   # never include the account number
            out["account"] = {k: summary.get(k) for k in ("status", "buying_power", "cash")}
    return out


def connect(username: str, api_key: str, api_secret: str) -> Dict[str, Any]:
    _require_encryption()
    _require_paper_endpoint()
    from data.alpaca_trading import get_account
    from db.paper_trading import save_paper_account

    summary = get_account(api_key.strip(), api_secret.strip())
    if not summary:
        raise PaperRejected("Could not validate those keys against the Alpaca paper endpoint. "
                            "Check they're paper keys (not live).")
    if not save_paper_account(username, api_key.strip(), api_secret.strip()):
        raise PaperUnavailable("Validated, but the keys couldn't be stored securely. Try again.")
    return paper_status(username)


def disconnect(username: str) -> None:
    from db.paper_trading import delete_paper_account

    delete_paper_account(username)


def activity(username: str, limit: int = 25) -> Dict[str, Any]:
    """Live positions and the order feed (orders synced into the durable feed, as on the web)."""
    from db.paper_events import list_events, sync_orders
    from db.paper_trading import get_paper_account

    _reconcile(username)
    acct = get_paper_account(username)
    if not acct:
        orders = list_events(username, limit)
        return {"connected": False, "positions": [], "orders": orders,
                "summary": {"positions": 0, "orders": len(orders), "market_value": None, "unrealized_pl": None}}
    _require_paper_endpoint()
    from data.alpaca_trading import get_orders, get_positions

    positions = get_positions(acct["api_key"], acct["api_secret"])
    orders = get_orders(acct["api_key"], acct["api_secret"], status="all", limit=limit)
    if orders:
        try:
            sync_orders(username, orders)
        except Exception:
            pass
    pos = positions or []
    orders_out = list_events(username, limit)
    def n(v: Any) -> Optional[float]:
        try:
            return None if v in (None, "") else float(v)
        except Exception:
            return None
    return {"connected": True, "positions": pos, "positions_available": positions is not None,
            "orders": orders_out,
            "summary": {"positions": len(pos) if positions is not None else None, "orders": len(orders_out),
                        "market_value": sum(n(p.get("market_value")) or 0 for p in pos) if positions is not None else None,
                        "unrealized_pl": sum(n(p.get("unrealized_pl")) or 0 for p in pos) if positions is not None else None}}


def order(username: str, ticker: str, qty: int, idempotency_key: str) -> Dict[str, Any]:
    from services.paper_orders import submit

    _require_paper_endpoint()
    try:
        row = latest_row(ticker) or {}
        p = plan(ticker, 10_000.0, 1.0) or {}
    except Exception:
        row, p = {}, {}  # a scan outage must not prevent recovering an accepted order
    def number(value):
        import math

        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None
    context = {"stop_price": number(p.get("stop")), "target_price": number((p.get("targets") or [None])[0]),
               "breakout_score": number(row.get("BreakoutScore")), "ai_confidence": number(row.get("AI Confidence"))}
    return submit(username, ticker, qty, idempotency_key, context)


def _reconcile(username):
    import logging

    try:
        reconcile_pending(username)
    except Exception:
        logging.getLogger(__name__).warning("paper_reconciliation_unavailable")
