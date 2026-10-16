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
    # Mounting does not run the subapp lifespan. The API already owns the alert
    # worker; do not run the standalone billing service's startup hook here.
    app.mount("/billing", billing.app)


def session_for_account(username: str, payload: dict, *, portal: bool = False) -> dict:
    from api.account import AccountError, BillingUnavailable
    from billing_service import main as billing
    from services.billing import _build_return_urls

    body = dict(payload)
    success, return_url = _build_return_urls(username)
    if success:
        body["success_url"] = success
    if return_url:
        body["return_url"] = return_url
    try:
        handler = billing.portal_for_account if portal else billing.checkout_for_account
        result = handler(username, body)
    except HTTPException as exc:
        if exc.status_code in {400, 404}:
            if portal:
                raise AccountError("No subscription to manage yet. Choose a plan first.", status=404) from exc
            raise AccountError(str(exc.detail), status=exc.status_code) from exc
        raise BillingUnavailable("embedded billing unavailable") from exc
    url = result.get("portal_url") or result.get("checkout_url")
    if not url:
        raise BillingUnavailable("no billing session URL")
    return {"url": url, "mode": "portal" if result.get("portal_url") else "checkout"}
