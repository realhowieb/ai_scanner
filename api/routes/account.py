"""Sign-up, email verification, passwords, email preferences, billing links, data export and account deletion."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api import account as acct
from api import models, ratelimit
from api.deps import (
    _AUTH,
    _fresh_account,
    _me_out,
    _settings,
    _token_pair,
    _track,
    _user,
    current_account,
    forget_account,
)
from api.scans import json_safe


class SignupBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)
    username: str = Field(min_length=1, max_length=40, description="Shown in the app; can also be used to sign in on the web")
    accept_terms: bool = Field(description="The usage agreement checkbox on the web sign-up form")
    client: Optional[str] = Field(default=None, max_length=80)
    attribution: Optional[Dict[str, str]] = Field(
        default=None, description="Web sign-ups: first-visit utm_* tags and referrer, for the acquisition funnel")


class TokenBody(BaseModel):
    token: str = Field(min_length=10, max_length=128)


class EmailBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)


class ResetConfirmBody(BaseModel):
    token: str = Field(min_length=10, max_length=128)
    new_password: str = Field(min_length=1, max_length=256)


class PasswordChangeBody(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class EmailPrefsUpdate(BaseModel):
    digest: Optional[bool] = None
    evening: Optional[bool] = None
    alerts: Optional[bool] = None


class CheckoutBody(BaseModel):
    plan: str = Field(pattern="^(pro|premium)$")
    interval: str = Field(default="month", pattern="^(month|year)$")


class PortalBody(BaseModel):
    flow: Optional[str] = Field(default=None, pattern="^cancel$", description="'cancel' opens the cancellation screen")


_RESET_SENT = ("If that email is registered, a reset link has been sent. "
               "Check your inbox (and spam folder).")


def register(app: FastAPI) -> None:
    @app.post("/v1/auth/signup", response_model=models.SignupResult, status_code=201,
              responses={400: {"description": "Invalid input or password rule"}, 409: {"description": "Email or username taken"},
                         429: {"description": "Too many sign-ups from this address"}})
    def signup(body: SignupBody, request: Request, _l: None = Depends(ratelimit.limit("signup"))) -> Dict[str, Any]:
        """Create a Free account (same rules as the web form) and sign in. A verification
        email is sent; verifying is needed to upgrade and for alert emails."""
        res = acct.signup(body.email, body.password, body.username, body.accept_terms)
        if body.attribution is not None:
            _track(body.attribution, "signup_completed", username=res["email"], plan="basic")
        return {**_token_pair(res["email"], _settings(request), body.client),
                "email": res["email"], "verification_sent": res["verification_sent"]}

    @app.post("/v1/auth/verify-email", response_model=models.Message,
              responses={400: {"description": "Invalid or expired link"}})
    def verify_email(body: TokenBody, _l: None = Depends(ratelimit.limit("verify"))) -> Dict[str, Any]:
        """Confirm an email address with the token from the verification link."""
        acct.verify_email(body.token)
        return {"message": "Email verified."}

    @app.post("/v1/me/verify-email", response_model=models.Message, responses={401: {"description": "Not signed in"}})
    def resend_verification(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Send a new verification link to the signed-in account's email (3 per hour)."""
        user = _user(account)
        if acct.is_verified(user):
            return {"message": "Your email is already verified."}
        ratelimit.check("verify_resend", user)
        if not acct.send_verification(user):
            raise HTTPException(503, "We couldn't send the email right now. Try again later.")
        return {"message": "Verification email sent. Check your inbox (and spam)."}

    @app.post("/v1/auth/password-reset", response_model=models.Message, status_code=202,
              responses={429: {"description": "Too many requests from this address"}})
    def password_reset(body: EmailBody, _l: None = Depends(ratelimit.limit("password_reset"))) -> Dict[str, Any]:
        """Email a reset link. Same answer whether or not the account exists."""
        acct.request_password_reset(body.email)
        return {"message": _RESET_SENT}

    @app.post("/v1/auth/password-reset/confirm", response_model=models.Message,
              responses={400: {"description": "Invalid link or password rule"}})
    def password_reset_confirm(body: ResetConfirmBody, _l: None = Depends(ratelimit.limit("verify"))) -> Dict[str, Any]:
        """Set a new password with the token from the reset link; signs the account out everywhere."""
        acct.confirm_password_reset(body.token, body.new_password)
        return {"message": "Password updated. You've been signed out on all devices; sign in with the new password."}

    @app.post("/v1/me/password", response_model=models.TokenPair,
              responses={400: {"description": "Wrong current password or password rule"}, 401: {"description": "Not signed in"}})
    def change_password(body: PasswordChangeBody, request: Request,
                        account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Change the password. Every other session (web and app) is signed out; this
        device gets a new token pair."""
        ratelimit.check("login", ratelimit.client_ip(request))
        acct.change_password(_fresh_account(account), body.current_password, body.new_password)
        forget_account(_user(account))
        return _token_pair(_user(account), _settings(request), None)

    @app.get("/v1/me/export", responses={**_AUTH, 429: {"description": "Too many exports"}},
             summary="Download your data")
    def export_me(account: Dict[str, Any] = Depends(current_account)) -> JSONResponse:
        """Your account, watchlists, alerts and alert rules (with recent alert events), journal,
        email settings, devices and saved scans, as one JSON file. No passwords, tokens or
        paper-trading keys. Sections that couldn't be read are null and listed in `unavailable`."""
        from api.export import build_export

        user = _user(account)
        ratelimit.check("export", user)
        body = json_safe(build_export(user, _me_out(account)))
        day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        return JSONResponse(body, headers={"Content-Disposition": f'attachment; filename="hsf-data-{day}.json"',
                                           "Cache-Control": "no-store"})

    @app.get("/v1/me/email-preferences", response_model=models.EmailPrefs, responses=_AUTH)
    def email_prefs(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return acct.get_email_prefs(_user(account))

    @app.patch("/v1/me/email-preferences", response_model=models.EmailPrefs, responses=_AUTH)
    def email_prefs_update(body: EmailPrefsUpdate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return acct.set_email_prefs(_user(account), body.model_dump(exclude_none=True))

    @app.post("/v1/billing/checkout", response_model=models.BillingLink,
              responses={**_AUTH, 403: {"description": "Verify your email first"},
                         502: {"description": "Billing service unavailable"}})
    def billing_checkout(body: CheckoutBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Stripe checkout for Pro or Premium (monthly or yearly when enabled). Existing
        subscribers get Stripe's plan-change screen instead (mode=portal). Open the URL in a browser."""
        return acct.checkout_url(_user(account), body.plan, body.interval)

    @app.post("/v1/billing/portal", response_model=models.BillingLink,
              responses={**_AUTH, 404: {"description": "No subscription yet"}, 502: {"description": "Billing service unavailable"}})
    def billing_portal(body: PortalBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Stripe Customer Portal: payment method, invoices, plan, cancellation (flow=cancel)."""
        return acct.portal_url(_user(account), body.flow)


class DeleteAccountBody(BaseModel):
    password: str = Field(min_length=1, max_length=256)
    confirm: Literal["DELETE"] = Field(description='Type "DELETE": the user confirmed permanent deletion')


def register_delete(app: FastAPI) -> None:
    @app.delete("/v1/me", status_code=204, summary="Delete your account",
                responses={**_AUTH, 400: {"description": "Wrong password"},
                           409: {"description": "Cancel your paid subscription first (or admin account)"},
                           422: {"description": 'confirm must be "DELETE"'}, 429: {"description": "Too many attempts"}})
    def delete_me(body: DeleteAccountBody, account: Dict[str, Any] = Depends(current_account)) -> None:
        """Permanently deletes your account and its data (watchlists, alerts, journal, paper keys,
        settings, saved scans, sessions, devices). Refused while a paid subscription is active:
        cancel it first via POST /v1/billing/portal {"flow": "cancel"}. Can't be undone."""
        ratelimit.check("delete_account", _user(account))
        acct.delete_account(_fresh_account(account), body.password)
        forget_account(_user(account))
