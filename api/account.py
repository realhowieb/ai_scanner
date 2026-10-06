"""Account endpoints for the API (P1-59 account step): sign-up, email
verification, password reset and change, email preferences, billing links.

Each flow calls the same functions the Streamlit pages use (db.users,
db.password_reset, db.email_verification, ui.password_policy, ui.email_utils,
ui.checkout, ui.auth_sessions), so rules, emails, tokens and limits match the
web app. Emailed links point at the web app's pages (APP_BASE_URL), which
already handle them; the API's confirm endpoints accept the same tokens for a
future frontend.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from api.store import DatabaseUnavailable

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")  # same as the web app's forms
USERNAME_MAX = 40


class AccountError(ValueError):
    """A user-facing problem with the request (400/409 decided by the route)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _hash_password(password: str) -> str:
    import bcrypt

    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _link(path: str, token: str) -> str:
    from config import APP_BASE_URL

    return f"{(APP_BASE_URL or '').rstrip('/')}/{path}?token={token}"


def _check_password(password: str, *, email: Optional[str], username: Optional[str] = None) -> None:
    from ui.password_policy import password_problem

    problem = password_problem(password, email=email, username=username)
    if problem:
        raise AccountError(problem)


def send_verification(email: str) -> bool:
    """Issue a verification token and email the link. False if not sent."""
    try:
        from db.email_verification import create_verification_token
        from ui.email_utils import send_verification_email

        token = create_verification_token(email)
        return bool(token) and bool(send_verification_email(to_address=email, verify_url=_link("verify_email", token)))
    except Exception:
        return False


# ---- sign-up -------------------------------------------------------------------------------------
def signup(email: str, password: str, username: str, accept_terms: bool) -> Dict[str, Any]:
    """Create a Free account with the web form's rules. Returns {email, verification_sent}."""
    from db import users

    email = (email or "").strip().lower()
    username = (username or "").strip()
    if not username:
        raise AccountError("Please choose a username.")
    if "@" in username or " " in username or len(username) > USERNAME_MAX:
        raise AccountError("Username cannot contain spaces or '@'.")
    if not EMAIL_RE.match(email):
        raise AccountError("Please enter a valid email address.")
    _check_password(password, email=email, username=username)
    if not accept_terms:
        raise AccountError("Please confirm the usage agreement to continue.")
    try:
        taken = users.get_user_by_username(email) is not None or \
            users.find_username_by_display_name(username) is not None
    except Exception as e:
        raise DatabaseUnavailable("database unavailable") from e
    if taken:
        raise AccountError("That email or username is already taken.", status=409)
    try:
        users.create_user_account(email=email, password_hash=_hash_password(password),
                                  tier="basic", full_name=username)
    except ValueError as e:  # created in between by someone else
        raise AccountError("That email or username is already taken.", status=409) from e
    except RuntimeError as e:
        raise DatabaseUnavailable("database unavailable") from e
    return {"email": email, "verification_sent": send_verification(email)}


# ---- email verification --------------------------------------------------------------------------
def verify_email(token: str) -> str:
    from db.email_verification import consume_verification_token

    user = consume_verification_token((token or "").strip()) if token and len(token) <= 128 else None
    if not user:
        raise AccountError("This verification link is invalid or has expired.")
    return str(user).strip().lower()


def is_verified(email: str) -> bool:
    try:
        from db.email_verification import is_email_verified

        return bool(is_email_verified(email))
    except Exception:
        return True  # same fail-open rule as the web app's gate


# ---- password reset -------------------------------------------------------------------------------
def request_password_reset(email: str) -> None:
    """Email a reset link when the account exists and is active. Says nothing either
    way (no account enumeration); db.password_reset limits 3 requests per hour."""
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email):
        raise AccountError("Please enter a valid email address.")
    try:
        from api.store import get_account

        account = get_account(email)
    except DatabaseUnavailable:
        raise
    if not account or account.get("is_active") is False:
        return
    try:
        from config import RESET_TOKEN_TTL_MINUTES
        from db.password_reset import create_reset_token
        from ui.email_utils import send_password_reset_email

        token = create_reset_token(str(account["username"]).strip().lower(), ttl_minutes=RESET_TOKEN_TTL_MINUTES)
        if token:
            send_password_reset_email(str(account["username"]).strip().lower(), _link("reset_password", token))
    except Exception:
        pass  # never reveal a failure that depends on the account existing


def _set_password_and_sign_out(username: str, new_password: str, reason: str) -> None:
    from api.store import revoke_all_refresh_tokens
    from db.users import update_neon_user_password

    if not update_neon_user_password(username, _hash_password(new_password)):
        raise DatabaseUnavailable("database unavailable")
    try:  # P1-54: sign the account out everywhere (web sessions and app sessions)
        from ui.auth_sessions import revoke_user_sessions

        revoke_user_sessions(username)
    except Exception:
        pass
    revoke_all_refresh_tokens(username, reason)
    from api.devices import remove_all

    remove_all(username)  # P1-64: signed-out phones stop getting this account's pushes


def confirm_password_reset(token: str, new_password: str) -> str:
    from db.password_reset import consume_reset_token, peek_reset_token

    token = (token or "").strip()
    if not token or len(token) > 128 or not token.replace("-", "").replace("_", "").isalnum():
        raise AccountError("This link is invalid or has expired. Please request a new one.")
    account = peek_reset_token(token)  # the rule also checks the password against the email
    if not account:
        raise AccountError("This link is invalid or has expired. Please request a new one.")
    _check_password(new_password, email=account)
    username = consume_reset_token(token)
    if not username:
        raise AccountError("This link is invalid or has expired. Please request a new one.")
    _set_password_and_sign_out(username, new_password, "password_reset")
    return username


def change_password(account: Dict[str, Any], current: str, new: str) -> None:
    from api.store import check_password

    username = str(account["username"]).strip().lower()
    if not check_password(account, current):
        raise AccountError("Your current password is incorrect.")
    if new == current:
        raise AccountError("Choose a password you haven't used for this account.")
    _check_password(new, email=username, username=account.get("full_name"))
    _set_password_and_sign_out(username, new, "password_change")


# ---- email preferences ----------------------------------------------------------------------------
def get_email_prefs(username: str) -> Dict[str, bool]:
    from db.email_prefs import get_prefs

    return get_prefs(username)


def set_email_prefs(username: str, changes: Dict[str, bool]) -> Dict[str, bool]:
    from db.email_prefs import KINDS, get_prefs, set_prefs

    clean = {k: bool(v) for k, v in changes.items() if k in KINDS and v is not None}
    if clean and not set_prefs(username, **clean):
        raise DatabaseUnavailable("database unavailable")
    return get_prefs(username)


# ---- account deletion (P2-82) ---------------------------------------------------------------------
def delete_account(account: Dict[str, Any], password: str) -> Dict[str, int]:
    """Delete the signed-in account after re-checking its password (db.account_deletion)."""
    from api.store import _conn, check_password
    from db.account_deletion import DeletionBlocked
    from db.account_deletion import delete_account as _delete

    if not check_password(account, password):
        raise AccountError("Your password is incorrect.")
    conn = _conn()
    try:
        return _delete(conn, str(account["username"]).strip().lower())
    except DeletionBlocked as e:
        raise AccountError(str(e), status=409) from e
    finally:
        conn.close()


# ---- billing ---------------------------------------------------------------------------------------
class BillingUnavailable(RuntimeError):
    """The billing service couldn't be reached or answered unexpectedly (502)."""


def checkout_url(username: str, plan: str, interval: str) -> Dict[str, str]:
    """Stripe checkout (new subscribers) or plan-change portal (existing ones), the
    same call the web Billing page makes. Requires a verified email, as on the web."""
    if not is_verified(username):
        raise AccountError("Please verify your email before upgrading.", status=403)
    from ui.checkout import create_checkout_url

    url, err = create_checkout_url(username, plan, interval)
    if not url:
        raise BillingUnavailable(err or "no url")
    mode = "portal" if "billing.stripe.com" in url else "checkout"
    return {"url": url, "mode": mode}


def portal_url(username: str, flow: Optional[str]) -> Dict[str, str]:
    """Stripe Customer Portal (manage plan, payment method, invoices; flow='cancel'
    opens the cancellation screen), via the billing service like the web page."""
    import requests

    from config import BILLING_API_BASE
    from ui.checkout import billing_auth_headers

    base = (BILLING_API_BASE or "").rstrip("/")
    if not base.startswith(("https://", "http://")):
        raise BillingUnavailable("billing service URL not configured")
    auth = billing_auth_headers(username)
    if not auth:
        raise BillingUnavailable("billing token unavailable")
    body: Dict[str, Any] = {"email": username}
    if flow:
        body["flow"] = flow
    try:
        from ui.checkout import _build_return_urls

        _success, ret = _build_return_urls(username)
        if ret:
            body["return_url"] = ret
    except Exception:
        pass
    try:
        resp = requests.post(f"{base}/create-portal-session", json=body, headers=auth, timeout=30)
    except Exception as e:
        raise BillingUnavailable(type(e).__name__) from e
    if resp.status_code in (400, 404):  # billing service: 400 "No subscription found for this account yet."
        raise AccountError("No subscription to manage yet. Choose a plan first.", status=404)
    if resp.status_code != 200:
        raise BillingUnavailable(f"billing service HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError as e:
        raise BillingUnavailable("billing service returned non-JSON") from e
    url = str(data.get("portal_url") or "").strip()
    if not url:
        raise BillingUnavailable("no portal_url")
    return {"url": url, "mode": "portal"}
