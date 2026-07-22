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
