"""HSF API service (P1-59). Run: uvicorn api.main:app

Endpoints (v1): GET /healthz · POST /v1/auth/login · POST /v1/auth/refresh ·
POST /v1/auth/logout · GET /v1/me · GET /v1/today. OpenAPI docs at /docs.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from api import models, store, tokens
from api.settings import Settings, load_settings

log = logging.getLogger("hsf_api")

_UNAUTHORIZED = "Invalid or expired token"
_BAD_LOGIN = "Email or password is incorrect."


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="HSFinest.AI API", version="1.0.0")
    app.state.settings = settings
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])

    @app.exception_handler(store.DatabaseUnavailable)
    def _db_down(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": "Service temporarily unavailable. Try again shortly."},
                            status_code=503, headers={"Retry-After": "30"})

    try:  # connection drops mid-query are an outage too, not a server bug
        import psycopg

        app.add_exception_handler(psycopg.OperationalError, _db_down)
    except ImportError:  # pragma: no cover
        pass

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/healthz", response_model=models.Health)
    def healthz() -> Dict[str, bool]:
        """Liveness only: no database call (for the Render health check)."""
        return {"ok": True}

    _routes(app)
    return app


class LoginBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)
    client: Optional[str] = Field(default=None, max_length=80)


class RefreshBody(BaseModel):
    refresh_token: str = Field(min_length=20, max_length=200)


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _token_pair(username: str, settings: Settings, client: Optional[str]) -> Dict[str, Any]:
    refresh = tokens.new_refresh_token()
    store.save_refresh_token(tokens.hash_refresh_token(refresh), username, settings.refresh_ttl_s, client)
    return {"access_token": tokens.create_access_token(username, settings), "token_type": "bearer",
            "expires_in": settings.access_ttl_s, "refresh_token": refresh}


# Declared as a security scheme so /docs shows an Authorize button (paste the
# access token only; the docs page adds "Bearer ").
_bearer = HTTPBearer(auto_error=False, description="Access token from POST /v1/auth/login")


def current_account(request: Request,
                    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> Dict[str, Any]:
    """The signed-in, active account for a `Authorization: Bearer <access token>` header."""
    token = (creds.credentials if creds and (creds.scheme or "").lower() == "bearer" else "").strip()
    if not token:
        raise HTTPException(401, _UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})
    username = tokens.verify_access_token(token, _settings(request))
    account = store.get_account(username) if username else None
    if not account or account.get("is_active") is False:
        raise HTTPException(401, _UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})
    return account


def entitlements_for(account: Dict[str, Any]) -> Dict[str, Any]:
    from auth.tiering import TIER_ORDER, has_min_tier
    from ui.app_session import ALERT_LIMIT_BY_TIER, compute_entitlements

    is_admin = bool(account.get("is_admin"))
    tier = str(account.get("tier") or "basic").strip().lower()
    tier = "admin" if is_admin else (tier if tier in TIER_ORDER else "basic")
    return {"tier": tier, "is_admin": is_admin,
            "entitlements": compute_entitlements(tier_obj=tier, is_admin=is_admin, has_min_tier_fn=has_min_tier),
            "alert_limit": ALERT_LIMIT_BY_TIER.get(tier, 1)}


def _routes(app: FastAPI) -> None:
    @app.post("/v1/auth/login", response_model=models.TokenPair,
              responses={401: {"description": "Wrong email or password"}, 429: {"description": "Rate limited"},
                         503: {"description": "Database unavailable"}})
    def login(body: LoginBody, request: Request) -> Dict[str, Any]:
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
        elif status == "grace":
            log.info("refresh token retried within the grace window")
        account = store.get_account(username) if status in ("ok", "grace") and username else None
        if not account or account.get("is_active") is False:
            raise HTTPException(401, _UNAUTHORIZED)
        return _token_pair(str(account["username"]).strip().lower(), settings, None)

    @app.post("/v1/auth/logout", status_code=204)
    def logout(body: RefreshBody) -> None:
        store.revoke_refresh_token(tokens.hash_refresh_token(body.refresh_token))

    @app.get("/v1/me", response_model=models.Me, responses={401: {"description": "Not signed in"}})
    def me(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        from ui.plan_labels import plan_label

        ent = entitlements_for(account)
        return {"email": str(account["username"]).strip().lower(),
                "name": account.get("full_name") or None,
                "plan": ent["tier"], "plan_label": plan_label(ent["tier"]),
                "is_admin": ent["is_admin"], "alert_limit": ent["alert_limit"],
                "entitlements": ent["entitlements"]}

    @app.get("/v1/today", response_model=models.Today, responses={401: {"description": "Not signed in"}})
    def today(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        from api.today import build_today

        return build_today(dt.datetime.now(dt.timezone.utc), entitlements_for(account)["entitlements"])


def _failing_app(message: str):
    """ASGI app that refuses to start with `message`, so uvicorn logs
    "Application startup failed" with the real reason and exits."""
    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            await receive()
            await send({"type": "lifespan.startup.failed", "message": message})
            return
        raise RuntimeError(message)

    return app


def _module_app():
    """Module-level app for `uvicorn api.main:app` (settings from the environment).
    Importing never raises, so tests can import create_app without the secret."""
    try:
        return create_app()
    except RuntimeError as e:
        log.error("HSF API not started: %s", e)
        return _failing_app(f"HSF API not started: {e}")


app = _module_app()
