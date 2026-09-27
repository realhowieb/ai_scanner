"""Shared Stripe checkout helper.

Creates a checkout (or portal) session via the billing service, embedding an
rt restore token in the success/return URLs so the user is restored after the
Stripe round-trip without relying on browser cookies.

Run 83: each rt is a separate single-use, 2-hour restore token (ui/auth_tokens),
not a reusable login session, and every billing-service call carries a
single-use billing token proving which signed-in account is asking.
"""
from __future__ import annotations

import json
import urllib.request


def _build_return_urls(username: str) -> tuple[str | None, str | None]:
    """Checkout-success and portal-return URLs, each with its own single-use
    restore token. Returns (success_url, portal_url); None where unavailable."""
    try:
        from config import APP_BASE_URL
        from ui.auth_tokens import issue_token

        base = (APP_BASE_URL or "").rstrip("/")
        success_rt = issue_token(username, "restore")
        portal_rt = issue_token(username, "restore")
        return (
            f"{base}/?checkout=success&rt={success_rt}" if success_rt else None,
            f"{base}/?portal=return&rt={portal_rt}" if portal_rt else None,
        )
    except Exception:
        return None, None


def billing_auth_headers(username: str) -> dict[str, str] | None:
    """Header proving the signed-in account to the billing service (single use).
    None when a token can't be issued; callers must then not call the service."""
    from ui.auth_tokens import issue_token

    token = issue_token(username, "billing")
    return {"X-HSF-Auth": token} if token else None


def create_checkout_url(email: str, plan: str) -> tuple[str | None, str | None]:
    """Create a Stripe checkout/portal session. Returns (url, error)."""
    if not email or plan not in {"pro", "premium"}:
        return None, "Invalid plan or missing account."
    try:
        from config import BILLING_API_BASE
        base = (BILLING_API_BASE or "").rstrip("/")
        if not base:
            return None, "BILLING_API_BASE not configured."

        auth = billing_auth_headers(email)
        if not auth:
            return None, "Couldn't verify your account right now. Please try again in a moment."
        success_url, return_url = _build_return_urls(email)
        body: dict[str, str] = {"email": email, "plan": plan}
        if success_url:
            body["success_url"] = success_url
        if return_url:
            body["return_url"] = return_url

        req = urllib.request.Request(
            f"{base}/create-checkout-session",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **auth},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        url = data.get("checkout_url") or data.get("url") or data.get("portal_url")
        return (url, None) if url else (None, f"Unexpected billing response: {data}")
    except Exception as e:
        return None, str(e)
