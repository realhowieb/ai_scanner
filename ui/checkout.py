"""Shared Stripe checkout helper.

Creates a checkout (or portal) session via the billing service, embedding an
rt session token in the success/return URLs so the user is restored after the
Stripe round-trip without relying on browser cookies.
"""
from __future__ import annotations

import json
import urllib.request


def _build_return_urls(username: str) -> tuple[str | None, str | None]:
    """Mint a session for the user; embed its id in checkout success + portal
    return URLs. Returns (success_url, portal_url)."""
    try:
        from config import APP_BASE_URL
        from ui.auth_sessions import create_session
        sid = create_session(username)
        if not sid:
            return None, None
        base = (APP_BASE_URL or "").rstrip("/")
        return (
            f"{base}/?checkout=success&rt={sid}",
            f"{base}/?portal=return&rt={sid}",
        )
    except Exception:
        return None, None


def create_checkout_url(email: str, plan: str) -> tuple[str | None, str | None]:
    """Create a Stripe checkout/portal session. Returns (url, error)."""
    if not email or plan not in {"pro", "premium"}:
        return None, "Invalid plan or missing account."
    try:
        from config import BILLING_API_BASE
        base = (BILLING_API_BASE or "").rstrip("/")
        if not base:
            return None, "BILLING_API_BASE not configured."

        success_url, return_url = _build_return_urls(email)
        body: dict[str, str] = {"email": email, "plan": plan}
        if success_url:
            body["success_url"] = success_url
        if return_url:
            body["return_url"] = return_url

        req = urllib.request.Request(
            f"{base}/create-checkout-session",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        url = data.get("checkout_url") or data.get("url") or data.get("portal_url")
        return (url, None) if url else (None, f"Unexpected billing response: {data}")
    except Exception as e:
        return None, str(e)
