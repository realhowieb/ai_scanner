"""Recoverable, idempotent paper orders shared by the API and classic app."""

import hashlib
import logging
import math
import re
import time
from urllib.parse import urlparse

from data import alpaca_trading as broker
from db import paper_orders as store
from db.trades import reconcile_paper_fill

log = logging.getLogger(__name__)


class PaperUnavailable(RuntimeError):
    pass


class PaperRejected(ValueError):
    pass


class OrderConflict(ValueError):
    pass


def require_paper_endpoint():
    if (urlparse(broker._base_url()).hostname or "").lower() != "paper-api.alpaca.markets":
        raise PaperUnavailable("Paper trading is unavailable right now.")


def account_ref(account):
    return hashlib.sha256(account["api_key"].encode()).hexdigest()


def client_id(user, key):
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", key or ""):
        raise PaperRejected("A valid order confirmation ID is required.")
    return "hsf-" + hashlib.sha256(f"{user.strip().lower()}\0{key}".encode()).hexdigest()[:40]


def _lookup(account, cid, *, timeout_s=None):
    if timeout_s is not None:
        return broker.get_order_by_client_id(account["api_key"], account["api_secret"], cid, timeout_s=timeout_s)
    return broker.get_order_by_client_id(account["api_key"], account["api_secret"], cid)


def _confirmed(result, intent):
    try:
        return (
            bool(result.get("order_id"))
            and result.get("client_order_id") == intent["client_order_id"]
            and result.get("symbol", "").upper() == intent["ticker"]
            and float(result.get("qty")) == intent["qty"]
            and result.get("side") == "buy"
        )
    except (TypeError, ValueError, AttributeError):
        return False


def _record(user, intent, result):
    if not _confirmed(result, intent):
        raise PaperUnavailable("The order response couldn't be confirmed. Retry the same order confirmation.")
    _fill_quantity(result, intent["qty"])
    cid = intent["client_order_id"]
    # Persist the broker response before attempting the independent journal write.
    saved = store.save(user, cid, state="submitted", result=result)
    if saved["state"] == "synced":
        return _snapshot(saved)
    if saved["state"] == "rejected":
        raise PaperRejected(saved.get("error") or "Order rejected.")
    result = saved["result"]
    synced = False
    try:
        filled = float(result.get("filled_qty") or 0)
        price = float(result.get("filled_avg_price") or 0)
        if (
            math.isfinite(filled)
            and math.isfinite(price)
            and price > 0
            and 0 < filled <= intent["qty"]
            and filled.is_integer()
        ):
            reconcile_paper_fill(user, result["order_id"], intent["ticker"], price, int(filled), intent["context"])
            synced = True
        terminal = result.get("status") in {"filled", "canceled", "expired", "rejected"}
        if terminal and (synced or filled == 0):
            saved = store.save(user, cid, state="synced", result=result)
            if saved["state"] == "synced":
                return _snapshot(saved)
    except Exception:
        # The durable submitted intent remains eligible for reconciliation.
        log.warning("paper_journal_sync_pending", exc_info=False)
        synced = False
    return {
        "order_id": result["order_id"],
        "status": result.get("status") or "submitted",
        "ticker": intent["ticker"],
        "qty": intent["qty"],
        "filled_avg_price": result.get("filled_avg_price"),
        "journal_synced": synced,
    }


def submit(user, ticker, qty, key, context=None):
    require_paper_endpoint()
    from db.paper_trading import get_paper_account

    user, ticker = user.strip().lower(), ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,9}", ticker) or type(qty) is not int or not 1 <= qty <= 100_000:
        raise PaperRejected("Choose a ticker and a positive whole-share quantity up to 100,000.")
    account = get_paper_account(user)
    if not account:
        raise PaperRejected("Connect your Alpaca paper account first.")
    cid = client_id(user, key)
    try:
        intent = store.reserve(user, cid, account_ref(account), ticker, qty, context or {})
        if not intent:
            raise RuntimeError("Missing order intent")
        if (intent["ticker"], intent["qty"]) != (ticker, qty):
            raise OrderConflict("This confirmation belongs to a different order or paper account.")
        if intent["account_ref"] != account_ref(account):
            # A key rotation may still refer to the same Alpaca account. Only a
            # confirmed existing order is recoverable with different credentials;
            # never submit this old confirmation against another account.
            recovered = _lookup(account, cid)
            if recovered.get("ok") and _confirmed(recovered, intent):
                return _record(user, intent, recovered)
            if not recovered.get("ok") and not recovered.get("not_found"):
                raise PaperUnavailable(
                    "Couldn't confirm the order after the credential change. Retry the same confirmation."
                )
            raise OrderConflict(
                "This confirmation belongs to a different paper account. Check its original order first."
            )
        if intent["state"] == "rejected":
            raise PaperRejected(intent.get("error") or "Order rejected. Review a new order to try again.")
        if intent.get("state") == "synced" and intent.get("result"):
            return _snapshot(intent)
        result = _lookup(account, cid)
        if not result.get("ok") and intent.get("result"):
            return _record(user, intent, intent["result"])
        if not result.get("ok"):
            if not result.get("not_found"):
                raise PaperUnavailable("Couldn't confirm this order's status. Retry the same confirmation.")
            result = broker.submit_market_order(
                account["api_key"], account["api_secret"], ticker, qty, side="buy", client_order_id=cid
            )
            if not result.get("ok"):
                # Handles lost responses and a concurrent request winning the same ID.
                found = _lookup(account, cid)
                if found.get("ok"):
                    result = found
                elif found.get("not_found") and result.get("status_code") in {400, 401, 403, 422}:
                    message = "Order rejected. Check your paper account and review a new order."
                    store.save(user, cid, state="rejected", error=message)
                    raise PaperRejected(message)
                else:
                    raise PaperUnavailable(
                        "Order status is uncertain. Retry this confirmation; do not create another order."
                    )
        return _record(user, intent, result)
    except (PaperRejected, PaperUnavailable, OrderConflict):
        raise
    except Exception as exc:
        raise PaperUnavailable("Order tracking is temporarily unavailable. Retry the same confirmation.") from exc


def _fill_quantity(result, qty):
    try:
        filled = float(result.get("filled_qty") or 0)
    except (TypeError, ValueError):
        filled = math.nan
    if not math.isfinite(filled) or not 0 <= filled <= qty or not filled.is_integer():
        raise PaperUnavailable("The broker's fill quantity could not be confirmed. Your confirmation is saved.")
    return filled


def _snapshot(intent, broker_available=None):
    result = intent.get("result") or {}
    filled = _fill_quantity(result, intent["qty"])
    return {"state": intent["state"], "order_id": result.get("order_id"),
            "status": result.get("status") or intent["state"], "ticker": intent["ticker"],
            "qty": intent["qty"], "filled_qty": filled, "filled_avg_price": result.get("filled_avg_price"),
            "journal_synced": intent["state"] == "synced" and filled > 0,
            "broker_available": broker_available, "updated_at": intent.get("updated_at")}


def status(user, key):
    """Recover an owned confirmation with lookups only, never a broker submission."""
    from db.paper_trading import get_paper_account

    user = user.strip().lower()
    cid = client_id(user, key)
    try:
        intent = store.get(user, cid)
        if intent is None:
            return None
        # A completed receipt must not recreate a journal row the user deleted.
        if intent["state"] in {"synced", "rejected"}:
            return _snapshot(intent)
        require_paper_endpoint()
        account = get_paper_account(user)
        if not account:
            return _snapshot(intent, False)
        result = _lookup(account, cid)
        if not result.get("ok") or not _confirmed(result, intent):
            return _snapshot(intent, False)
        receipt = _record(user, intent, result)
        latest = store.get(user, cid)
        return {**_snapshot(latest, True), "journal_synced": receipt["journal_synced"]}
    except PaperUnavailable:
        raise
    except Exception as exc:
        raise PaperUnavailable("Order tracking is temporarily unavailable. Your confirmation is saved.") from exc


def reconcile_pending(user):
    """Bounded recovery on journal/activity refresh; never submits an order."""
    require_paper_endpoint()
    from db.paper_trading import get_paper_account

    user = user.strip().lower()
    account = get_paper_account(user)
    if not account:
        return
    try:
        intents = store.pending(user, account_ref(account), limit=5)
        deadline = time.monotonic() + 3
        for intent in intents:
            if time.monotonic() >= deadline:
                break
            try:
                result = _lookup(account, intent["client_order_id"], timeout_s=deadline - time.monotonic())
                if result.get("ok"):
                    _record(user, intent, result)
            except Exception:
                log.warning("paper_order_reconciliation_item_pending", exc_info=False)
            finally:
                # Touch the existing state atomically; a stale batch cannot reset it.
                try:
                    store.save(user, intent["client_order_id"], state="pending")
                except Exception:
                    log.warning("paper_order_reconciliation_touch_failed", exc_info=False)
    except Exception:
        log.warning("paper_order_reconciliation_pending", exc_info=False)
