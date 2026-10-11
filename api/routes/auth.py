"""Sign-in, token refresh and sign-out, /v1/me and Today."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from api import devices, models, ratelimit, store, tokens
from api.deps import _BAD_LOGIN, _UNAUTHORIZED, _me_out, _settings, _token_pair, current_account, entitlements_for

log = logging.getLogger("hsf_api")



class LoginBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)
    client: Optional[str] = Field(default=None, max_length=80)


class RefreshBody(BaseModel):
    refresh_token: str = Field(min_length=20, max_length=200)


class LogoutBody(RefreshBody):
    push_token: Optional[str] = Field(default=None, max_length=600,
                                      description="This device's push token, so it stops getting this account's pushes")


def register(app: FastAPI) -> None:
    @app.post("/v1/auth/login", response_model=models.TokenPair,
              responses={401: {"description": "Wrong email or password"}, 429: {"description": "Rate limited"},
                         503: {"description": "Database unavailable"}})
    def login(body: LoginBody, request: Request, _limited: None = Depends(ratelimit.limit("login"))) -> Dict[str, Any]:
        from db.users import is_login_rate_limited, record_login_attempt

        settings = _settings(request)
        email = body.email.strip().lower()
        if is_login_rate_limited(email):
            raise HTTPException(429, "Too many failed sign-in attempts. Try again in a few minutes.")
        account = store.get_account(email)
        if account is None:
            store.burn_password_check()
            ok = False
        else:
            ok = store.check_password(account, body.password) and account.get("is_active") is not False
        record_login_attempt(email, success=ok, failure_reason=None if ok else "api_login_failed")
        if not ok:
            raise HTTPException(401, _BAD_LOGIN)
        return _token_pair(str(account["username"]).strip().lower(), settings, body.client)

    @app.post("/v1/auth/refresh", response_model=models.TokenPair,
              responses={401: {"description": "Unknown, expired or reused refresh token"}})
    def refresh(body: RefreshBody, request: Request) -> Dict[str, Any]:
        settings = _settings(request)
        status, username = store.use_refresh_token(tokens.hash_refresh_token(body.refresh_token))
        if status == "reused":
            log.warning("refresh token reuse: all sessions revoked for one account")
            if username:
                devices.remove_all(username)  # P1-64: no pushes to a possibly stolen session
        elif status == "grace":
            log.info("refresh token retried within the grace window")
        account = store.get_account(username) if status in ("ok", "grace") and username else None
        if not account or account.get("is_active") is False:
            raise HTTPException(401, _UNAUTHORIZED)
        return _token_pair(str(account["username"]).strip().lower(), settings, None)

    @app.post("/v1/auth/logout", status_code=204)
    def logout(body: LogoutBody) -> None:
        username = store.revoke_refresh_token(tokens.hash_refresh_token(body.refresh_token))
        if username and body.push_token:
            devices.remove_token(username, body.push_token.strip())

    @app.get("/v1/me", response_model=models.Me, responses={401: {"description": "Not signed in"}})
    def me(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return _me_out(account)

    @app.get("/v1/today", response_model=models.Today, responses={401: {"description": "Not signed in"}})
    def today(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        from api.today import build_today

        return build_today(dt.datetime.now(dt.timezone.utc), entitlements_for(account)["entitlements"])
