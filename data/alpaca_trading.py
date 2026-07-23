"""Alpaca paper-trading REST client (per-user keys).

Thin wrapper over the /v2 trading API against the paper endpoint. Callers pass
the *user's own* paper key/secret (loaded + decrypted from db.paper_trading), so
orders and positions belong to that user. Best-effort: network/HTTP failures are
returned as structured errors, never raised.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]

_TIMEOUT = 12.0
_DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"


def _base_url() -> str:
    try:
        from data.alpaca_config import DEFAULT_BASE_URL, alpaca_secret

        return (alpaca_secret("ALPACA_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")
    except Exception:
        return _DEFAULT_BASE_URL


def _headers(api_key: str, api_secret: str) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "Content-Type": "application/json",
    }


def get_account(api_key: str, api_secret: str) -> Optional[Dict[str, Any]]:
    """Validate the keys and return a small account summary, or None if invalid."""
    if requests is None or not api_key or not api_secret:
        return None
    try:
        r = requests.get(
            f"{_base_url()}/v2/account", headers=_headers(api_key, api_secret), timeout=_TIMEOUT
        )
        if r.status_code != 200:
            return None
        d = r.json() or {}
        return {
            "status": d.get("status"),
            "buying_power": d.get("buying_power"),
            "cash": d.get("cash"),
            "account_number": d.get("account_number"),
        }
    except Exception:
        return None


def get_positions(api_key: str, api_secret: str) -> Optional[list]:
    """Open positions as a list of dicts, or None on failure.

    Each dict: {symbol, qty, avg_entry_price, current_price, market_value,
    unrealized_pl, unrealized_plpc}. Best-effort; never raises.
    """
    if requests is None or not api_key or not api_secret:
        return None
    try:
        r = requests.get(
            f"{_base_url()}/v2/positions",
            headers=_headers(api_key, api_secret), timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        data = r.json() or []
    except Exception:
        return None
    out = []
    for p in data:
        if not isinstance(p, dict):
            continue
        out.append({
            "symbol": p.get("symbol"),
            "qty": p.get("qty"),
            "avg_entry_price": p.get("avg_entry_price"),
            "current_price": p.get("current_price"),
            "market_value": p.get("market_value"),
            "unrealized_pl": p.get("unrealized_pl"),
            "unrealized_plpc": p.get("unrealized_plpc"),
        })
    return out


def get_orders(api_key: str, api_secret: str, status: str = "all", limit: int = 25) -> Optional[list]:
    """Recent orders (newest first) as a list of dicts, or None on failure.

    Each dict: {id, symbol, side, qty, filled_qty, type, status,
    filled_avg_price, submitted_at, filled_at}. Best-effort; never raises.
    """
    if requests is None or not api_key or not api_secret:
        return None
    try:
        r = requests.get(
            f"{_base_url()}/v2/orders",
            headers=_headers(api_key, api_secret),
            params={"status": status, "limit": int(limit), "direction": "desc"},
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        data = r.json() or []
    except Exception:
        return None
    out = []
    for o in data:
        if not isinstance(o, dict):
            continue
        out.append({
            "id": o.get("id"),
            "symbol": o.get("symbol"),
            "side": o.get("side"),
            "qty": o.get("qty"),
            "filled_qty": o.get("filled_qty"),
            "type": o.get("type"),
            "status": o.get("status"),
            "filled_avg_price": o.get("filled_avg_price"),
            "submitted_at": o.get("submitted_at"),
            "filled_at": o.get("filled_at"),
        })
    return out


def submit_market_order(
    api_key: str, api_secret: str, symbol: str, qty: int, side: str = "buy"
) -> Dict[str, Any]:
    """Submit a whole-share market DAY order.

    Returns {"ok": True, "order_id", "status", "filled_avg_price", ...} on
    success, or {"ok": False, "error": <message>} on failure. Never raises.
    """
    if requests is None:
        return {"ok": False, "error": "HTTP client unavailable"}
    if not api_key or not api_secret:
        return {"ok": False, "error": "No connected paper account."}
    try:
        q = int(qty)
    except (TypeError, ValueError):
        q = 0
    if q <= 0:
        return {"ok": False, "error": "Quantity must be a positive whole number."}
    side = str(side or "buy").lower()
    if side not in ("buy", "sell"):
        return {"ok": False, "error": f"Invalid side: {side!r}"}

    payload = {
        "symbol": str(symbol or "").upper(),
        "qty": str(q),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    try:
        r = requests.post(
            f"{_base_url()}/v2/orders",
            headers=_headers(api_key, api_secret),
            json=payload,
            timeout=_TIMEOUT,
        )
    except Exception as e:
        return {"ok": False, "error": f"Network error: {type(e).__name__}"}
    if r.status_code not in (200, 201):
        try:
            msg = (r.json() or {}).get("message") or r.text
        except Exception:
            msg = r.text
        return {"ok": False, "error": f"Alpaca {r.status_code}: {msg}"}
    try:
        d = r.json() or {}
    except Exception:
        return {"ok": False, "error": "Unexpected response from Alpaca."}
    return {
        "ok": True,
        "order_id": d.get("id"),
        "status": d.get("status"),
        "filled_avg_price": d.get("filled_avg_price"),
        "symbol": d.get("symbol"),
        "qty": d.get("qty"),
    }
