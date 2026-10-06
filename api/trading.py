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
from urllib.parse import urlparse

PAPER_HOST = "paper-api.alpaca.markets"
MAX_ORDER_QTY = 100_000


class PaperUnavailable(RuntimeError):
    """Paper trading can't run on this server (503)."""


class PaperRejected(ValueError):
    """Alpaca or the request rejected the order / keys (400)."""


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


def journal(username: str) -> Dict[str, Any]:
    """Trades (open first), open ones marked to live quotes, plus closed-trade stats."""
    from db.trades import journal_stats, list_trades

    trades = list_trades(username, limit=200)
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
    return {"trades": out, "stats": journal_stats(username)}


def _owned_trade(username: str, trade_id: int) -> Optional[Dict[str, Any]]:
    from db.trades import list_trades

    return next((t for t in list_trades(username, limit=1000) if int(t["id"]) == int(trade_id)), None)


def log(username: str, ticker: str, entry_price: float, shares: int) -> None:
    from db.trades import log_trade

    log_trade(username, ticker, float(entry_price), int(shares), source="api")


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
def _require_paper_endpoint() -> None:
    from data.alpaca_trading import _base_url

    if (urlparse(_base_url()).hostname or "").lower() != PAPER_HOST:
        raise PaperUnavailable("Paper trading is unavailable right now.")


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

    acct = get_paper_account(username)
    if not acct:
        return {"connected": False, "positions": [], "orders": list_events(username, limit)}
    _require_paper_endpoint()
    from data.alpaca_trading import get_orders, get_positions

    positions = get_positions(acct["api_key"], acct["api_secret"])
    orders = get_orders(acct["api_key"], acct["api_secret"], status="all", limit=limit)
    if orders:
        try:
            sync_orders(username, orders)
        except Exception:
            pass
    return {"connected": True, "positions": positions or [], "positions_available": positions is not None,
            "orders": list_events(username, limit)}


def order(username: str, ticker: str, qty: int) -> Dict[str, Any]:
    """Whole-share market BUY to the user's paper account, imported into the journal
    with the plan's stop/target and the scan's score, like the web."""
    from data.alpaca_trading import submit_market_order
    from db.paper_trading import get_paper_account
    from db.trades import log_trade

    _require_paper_endpoint()
    acct = get_paper_account(username)
    if not acct:
        raise PaperRejected("Connect your Alpaca paper account first (POST /v1/paper/account).")
    res = submit_market_order(acct["api_key"], acct["api_secret"], ticker, int(qty), side="buy")
    if not res.get("ok"):
        raise PaperRejected(f"Order rejected: {str(res.get('error') or 'unknown error')[:200]}")
    row = latest_row(ticker) or {}
    p = plan(ticker, 10_000.0, 1.0) or {}
    filled = res.get("filled_avg_price")
    entry = float(filled) if filled not in (None, "") else float(row.get("Last") or 0.0)
    try:
        log_trade(username, ticker, entry, int(qty), source="paper", alpaca_order_id=res.get("order_id"),
                  stop_price=p.get("stop"), target_price=(p.get("targets") or [None])[0],
                  breakout_score=row.get("BreakoutScore"), ai_confidence=row.get("AI Confidence"))
    except Exception:
        pass
    return {"order_id": res.get("order_id"), "status": res.get("status") or "submitted", "ticker": ticker,
            "qty": int(qty), "filled_avg_price": filled}
