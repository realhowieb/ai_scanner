"""Optional in-process billing on the paid API service.

External mode preserves the existing service during rollout and rollback.
Only server-authenticated account handlers may call session_for_account.
"""
from __future__ import annotations

import os

from fastapi import HTTPException


def embedded_enabled() -> bool:
    mode = os.getenv("BILLING_MODE", "external").strip().lower()
    if mode not in {"external", "embedded"}:
        raise RuntimeError("BILLING_MODE must be external or embedded")
    return mode == "embedded"


def register(app) -> None:
    if not embedded_enabled():
        return
    from billing_service import main as billing

    missing = billing._required_env_missing(
        "STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET", "STRIPE_PRICE_PRO",
        "STRIPE_PRICE_PREMIUM", "APP_SUCCESS_URL", "APP_CANCEL_URL", "DATABASE_URL",
    )
    if missing:
        raise RuntimeError(f"Embedded billing configuration missing: {', '.join(missing)}")
    try:
        grace = int(os.getenv("BILLING_PAST_DUE_GRACE_DAYS", "3"))
        if not 0 <= grace <= 30:
            raise ValueError
    except ValueError as exc:
        raise RuntimeError("BILLING_PAST_DUE_GRACE_DAYS must be between 0 and 30") from exc
    # Mounting does not run the subapp lifespan. The API already owns the alert
    # worker; do not run the standalone billing service's startup hook here.
    app.mount("/billing", billing.app)


def session_for_account(username: str, payload: dict, *, portal: bool = False) -> dict:
    from api.account import AccountError, BillingUnavailable
    from billing_service import main as billing
    from config import APP_BASE_URL

    body = dict(payload)
    # The web app uses its own HttpOnly session. Streamlit restore tokens are
    # neither needed nor consumed by Next.js and must not go in these URLs.
    base = APP_BASE_URL.rstrip("/")
    body["success_url"] = f"{base}/billing/return?checkout=success"
    body["return_url"] = f"{base}/billing/return?portal=return"
    try:
        handler = billing.portal_for_account if portal else billing.checkout_for_account
        result = handler(username, body)
    except HTTPException as exc:
        if exc.status_code in {400, 404}:
            if portal:
                raise AccountError("No subscription to manage yet. Choose a plan first.", status=404) from exc
            raise AccountError(str(exc.detail), status=exc.status_code) from exc
        if exc.status_code == 409:
            raise AccountError(str(exc.detail), status=409) from exc
        raise BillingUnavailable("embedded billing unavailable") from exc
    except Exception as exc:
        raise BillingUnavailable("billing session unavailable") from exc
    url = result.get("portal_url") or result.get("checkout_url")
    if not url:
        raise BillingUnavailable("no billing session URL")
    return {"url": url, "mode": "portal" if result.get("portal_url") else "checkout"}


def sync_account(username: str) -> dict:
    """Recover a delayed webhook from Stripe state, never from browser claims."""
    from api.account import AccountError, BillingUnavailable

    if not embedded_enabled():
        raise BillingUnavailable("billing sync requires embedded mode")
    from billing_service import main as billing

    try:
        return billing.sync_account_subscription(username)
    except HTTPException as exc:
        if exc.status_code == 409:
            raise AccountError(str(exc.detail), status=409) from exc
        raise BillingUnavailable("subscription sync unavailable") from exc
    except Exception as exc:
        raise BillingUnavailable("subscription sync unavailable") from exc


def status_for_account(username: str) -> dict:
    from api.account import BillingUnavailable

    if not embedded_enabled():
        return {"status": "external", "sync_state": "unavailable", "message": "Use Manage subscription to view billing details."}
    from billing_service import main as billing

    try:
        return billing.billing_status(username)
    except Exception as exc:
        raise BillingUnavailable("subscription status unavailable") from exc


def cancel_checkout(username: str) -> dict:
    from api.account import AccountError, BillingUnavailable

    if not embedded_enabled():
        raise BillingUnavailable("embedded billing is required")
    from billing_service import main as billing

    try:
        return billing.billing_engine().cancel_checkout(username)
    except HTTPException as exc:
        if exc.status_code == 409:
            raise AccountError(str(exc.detail), status=409) from exc
        raise BillingUnavailable("checkout status unavailable") from exc
    except Exception as exc:
        raise BillingUnavailable("checkout status unavailable") from exc


def diagnostics() -> dict:
    if not embedded_enabled():
        return {"mode": "external", "prices_configured": {}, "prices_valid": {},
                "webhook_configured": False, "webhook_check_available": False}
    from billing_service import main as billing

    preflight = billing._billing_preflight()
    prices = preflight["prices"]
    hooks = preflight["webhooks"]
    return {"mode": preflight["stripe_mode"],
            "prices_configured": {name: bool(value.get("id_set")) for name, value in prices.items()},
            "prices_valid": {name: bool(value.get("id_set")) and not value.get("problems") for name, value in prices.items()},
            "webhook_configured": bool(hooks["checked"] and not hooks["problems"]),
            "webhook_check_available": hooks["checked"], **billing.billing_engine().store.diagnostics()}
