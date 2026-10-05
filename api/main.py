"""HSF API service (P1-59). Run: uvicorn api.main:app

Endpoints (v1): GET /healthz · POST /v1/auth/login · POST /v1/auth/refresh ·
POST /v1/auth/logout · GET /v1/me · GET /v1/today · GET /v1/scans/latest ·
GET /v1/stocks/{ticker} · /v1/watchlists · /v1/alerts · sign-up, email
verification, password reset/change, email preferences, billing links. Full list in docs/API.md;
OpenAPI docs at /docs.
"""
from __future__ import annotations

import datetime as dt
import functools
import inspect
import json
import logging
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from api import account as acct
from api import models, ratelimit, store, tokens, user_data
from api.scans import json_safe
from api.settings import Settings, load_settings

log = logging.getLogger("hsf_api")
access_log = logging.getLogger("hsf_api.access")
if not access_log.handlers:  # uvicorn only configures its own loggers; one JSON line per request to stdout
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    access_log.addHandler(_h)
    access_log.setLevel(logging.INFO)
    access_log.propagate = False
_SAFE_RID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

_UNAUTHORIZED = "Invalid or expired token"
_BAD_LOGIN = "Email or password is incorrect."


class _ReleasingRoute(APIRoute):
    """Runs each endpoint, then ends the transaction on that worker thread's warm
    database connection. Several db.* helpers never call close(), so without this
    a worker thread sits idle in a transaction holding table locks until its next
    request (API acceptance run). Same thread as the endpoint, so it's the right
    connection."""

    def __init__(self, path: str, endpoint: Callable[..., Any], **kwargs: Any):
        if not inspect.iscoroutinefunction(endpoint):
            inner = endpoint

            @functools.wraps(inner)
            def endpoint(*args: Any, **kw: Any) -> Any:
                try:
                    return inner(*args, **kw)
                finally:
                    from db.engine import release_thread_connection

                    release_thread_connection()
        super().__init__(path, endpoint, **kwargs)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="HSFinest.AI API", version="1.0.0")
    app.router.route_class = _ReleasingRoute
    app.state.settings = settings
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST", "PATCH", "DELETE"],
                           allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
                           expose_headers=["X-Request-ID"])

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        """X-Request-ID on every response (a caller's own id is kept when it looks
        safe) and one JSON access-log line: method, route template, status, time.
        Never logs headers, bodies, query strings or tokens."""
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _SAFE_RID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            route = request.scope.get("route")
            access_log.info(json.dumps({
                "request_id": rid, "method": request.method,
                "route": getattr(route, "path", None) or "unmatched", "status": status,
                "ms": round((time.perf_counter() - t0) * 1000, 1)}))
        response.headers["X-Request-ID"] = rid
        return response

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

    @app.exception_handler(user_data.NotFound)
    def _not_found(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": f"No such {exc.args[0] if exc.args else 'item'}."}, status_code=404)

    @app.exception_handler(user_data.LimitReached)
    def _limit(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(user_data.Conflict)
    def _conflict(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(acct.AccountError)
    def _account_error(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=getattr(exc, "status", 400))

    @app.exception_handler(acct.BillingUnavailable)
    def _billing_down(_request: Request, exc: Exception) -> JSONResponse:
        log.warning("billing service call failed: %s", str(exc)[:120])
        return JSONResponse({"detail": "Billing is temporarily unavailable. Please try again in a minute."},
                            status_code=502)

    @app.get("/readyz", response_model=models.Ready, responses={503: {"description": "Database unavailable"}})
    def readyz() -> Dict[str, Any]:
        """Readiness: the database answers, plus the latest market scan's age for
        freshness monitoring. /healthz stays the liveness check."""
        store.ping()
        latest, age = None, None
        try:
            from api.today import market_runs

            runs = market_runs()
            if runs:
                created = runs[0]["created_at"]
                latest = created.isoformat()
                age = round((dt.datetime.now(dt.timezone.utc) - created).total_seconds() / 60.0, 1)
        except store.DatabaseUnavailable:
            raise
        except Exception:  # scan freshness is informational
            pass
        return {"ok": True, "database": "ok", "latest_scan_at": latest, "scan_age_minutes": age}

    _routes(app)
    _data_routes(app)
    _account_routes(app)
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
                "email_verified": acct.is_verified(str(account["username"]).strip().lower()),
                "entitlements": ent["entitlements"]}

    @app.get("/v1/today", response_model=models.Today, responses={401: {"description": "Not signed in"}})
    def today(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        from api.today import build_today

        return build_today(dt.datetime.now(dt.timezone.utc), entitlements_for(account)["entitlements"])


TICKER = Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$", description="Ticker symbol, e.g. AAPL or BRK.B")
_AUTH = {401: {"description": "Not signed in"}}
_OWNED = {**_AUTH, 404: {"description": "Not found (or not yours)"}}


class WatchlistCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    make_default: bool = False


class WatchlistUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    make_default: bool = Field(default=False, description="true makes this the default watchlist")


class TickersBody(BaseModel):
    tickers: List[str] = Field(min_length=1, max_length=user_data.MAX_TICKERS_PER_REQUEST)


class NoteBody(BaseModel):
    note: Optional[str] = Field(default=None, max_length=user_data.MAX_NOTE_LEN)


class AlertCreate(BaseModel):
    type: str = Field(description="breakout, watchlist, price, move, rvol, ema_cross or ewo_cross")
    ticker: Optional[str] = Field(default=None, max_length=12)
    threshold: Optional[float] = None
    direction: Optional[str] = Field(default=None, max_length=12)
    watchlist_only: bool = False


class AlertUpdate(BaseModel):
    enabled: bool


def _user(account: Dict[str, Any]) -> str:
    return str(account["username"]).strip().lower()


def _data_routes(app: FastAPI) -> None:
    # ---- step 5: scans and stock detail ----
    @app.get("/v1/scans/latest", response_model=models.LatestScan, responses=_AUTH)
    def scans_latest(account: Dict[str, Any] = Depends(current_account),
                     limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0, le=10_000),
                     min_score: int = Query(0, ge=0, le=100),
                     signal: Optional[str] = Query(None, pattern="^(golden_cross|breakout|prebreakout|gapper|gainer)$",
                                                   description="Only setups with this signal")) -> Dict[str, Any]:
        """The latest market scan's HSF setups, ranked as in the Scanner, up to the plan's row cap."""
        from api.scans import latest_scan

        ent = entitlements_for(account)
        if signal == "prebreakout" and not ent["entitlements"].get("can_early_breakout"):
            raise HTTPException(403, "PreBreakout is a Premium feature.")
        return latest_scan(ent["entitlements"], ent["tier"], limit=limit, offset=offset,
                           min_score=min_score, signal=signal)

    @app.get("/v1/stocks/{ticker}", response_model=models.StockDetail, responses=_AUTH)
    def stock(ticker: str = TICKER, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Stock Intelligence for one ticker: score, signals, reasons and risks, lifecycle,
        daily bars, and your watchlists and alerts on it."""
        from api.scans import daily_bars, stock_detail

        t = ticker.strip().upper()
        user = _user(account)
        out = stock_detail(t, entitlements_for(account)["entitlements"])
        try:
            bars = daily_bars(t)
        except Exception:  # chart data is optional; the page still renders
            bars = {"bars": [], "as_of": None}
        out.update({"bars": bars["bars"], "bars_as_of": bars["as_of"],
                    "watchlists": user_data.watchlists_with(user, t),
                    "alerts": json_safe(user_data.alerts_for(user, t))})
        return out

    # ---- step 6: watchlists ----
    @app.get("/v1/watchlists", response_model=List[models.Watchlist], responses=_AUTH)
    def watchlists(account: Dict[str, Any] = Depends(current_account)) -> List[Dict[str, Any]]:
        return user_data.list_watchlists(_user(account))

    @app.post("/v1/watchlists", response_model=models.WatchlistDetail, status_code=201,
              responses={**_AUTH, 403: {"description": "Watchlist limit reached"},
                         409: {"description": "A watchlist with that name exists"}})
    def watchlist_create(body: WatchlistCreate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return json_safe(user_data.create_watchlist(_user(account), body.name, body.make_default))

    @app.get("/v1/watchlists/{watchlist_id}", response_model=models.WatchlistDetail, responses=_OWNED)
    def watchlist_get(watchlist_id: int, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return json_safe(user_data.get_watchlist(_user(account), watchlist_id))

    @app.patch("/v1/watchlists/{watchlist_id}", response_model=models.WatchlistDetail,
               responses={**_OWNED, 409: {"description": "A watchlist with that name exists"}})
    def watchlist_update(watchlist_id: int, body: WatchlistUpdate,
                         account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return json_safe(user_data.update_watchlist(_user(account), watchlist_id, name=body.name,
                                                    make_default=body.make_default))

    @app.delete("/v1/watchlists/{watchlist_id}", status_code=204, responses=_OWNED)
    def watchlist_delete(watchlist_id: int, account: Dict[str, Any] = Depends(current_account)) -> None:
        user_data.delete_watchlist(_user(account), watchlist_id)

    @app.post("/v1/watchlists/{watchlist_id}/tickers", response_model=models.TickersResult, responses=_OWNED)
    def watchlist_add(watchlist_id: int, body: TickersBody,
                      account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return user_data.add_tickers(_user(account), watchlist_id, body.tickers)

    @app.delete("/v1/watchlists/{watchlist_id}/tickers/{ticker}", status_code=204, responses=_OWNED)
    def watchlist_remove(watchlist_id: int, ticker: str = TICKER,
                         account: Dict[str, Any] = Depends(current_account)) -> None:
        user_data.remove_ticker(_user(account), watchlist_id, ticker.upper())

    @app.patch("/v1/watchlists/{watchlist_id}/tickers/{ticker}", status_code=204, responses=_OWNED)
    def watchlist_note(watchlist_id: int, body: NoteBody, ticker: str = TICKER,
                       account: Dict[str, Any] = Depends(current_account)) -> None:
        user_data.set_note(_user(account), watchlist_id, ticker.upper(), body.note)

    # ---- step 6: alerts ----
    @app.get("/v1/alerts", response_model=models.Alerts, responses=_AUTH)
    def alerts(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        ent = entitlements_for(account)
        items = user_data.list_alerts(_user(account))
        return json_safe({"limit": ent["alert_limit"], "used": len(items),
                          "email_enabled": bool(ent["entitlements"].get("can_email_alerts")), "alerts": items})

    @app.post("/v1/alerts", response_model=models.Alert, status_code=201,
              responses={**_AUTH, 403: {"description": "Plan alert limit reached"},
                         409: {"description": "You already have this alert"},
                         422: {"description": "Invalid type, ticker, threshold or direction"}})
    def alert_create(body: AlertCreate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Same types and rules as the web app. Free 1 alert, Pro 5, Premium 25."""
        ent = entitlements_for(account)
        try:
            created = user_data.create_alert(_user(account), ent["alert_limit"], body.type, ticker=body.ticker,
                                             threshold=body.threshold, direction=body.direction,
                                             watchlist_only=body.watchlist_only)
        except user_data.Conflict:
            raise
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return json_safe(created)

    @app.patch("/v1/alerts/{alert_id}", response_model=models.Alert, responses=_OWNED)
    def alert_update(alert_id: int, body: AlertUpdate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return json_safe(user_data.set_alert_enabled(_user(account), alert_id, body.enabled))

    @app.delete("/v1/alerts/{alert_id}", status_code=204, responses=_OWNED)
    def alert_delete(alert_id: int, account: Dict[str, Any] = Depends(current_account)) -> None:
        user_data.delete_alert(_user(account), alert_id)

    @app.get("/v1/alerts/events", response_model=List[models.AlertEvent], responses=_AUTH)
    def alert_events(account: Dict[str, Any] = Depends(current_account),
                     limit: int = Query(20, ge=1, le=100)) -> List[Dict[str, Any]]:
        """Your most recent fired alerts, newest first."""
        return json_safe(user_data.alert_events(_user(account), limit))


class SignupBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)
    username: str = Field(min_length=1, max_length=40, description="Shown in the app; can also be used to sign in on the web")
    accept_terms: bool = Field(description="The usage agreement checkbox on the web sign-up form")
    client: Optional[str] = Field(default=None, max_length=80)


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


def _account_routes(app: FastAPI) -> None:
    @app.post("/v1/auth/signup", response_model=models.SignupResult, status_code=201,
              responses={400: {"description": "Invalid input or password rule"}, 409: {"description": "Email or username taken"},
                         429: {"description": "Too many sign-ups from this address"}})
    def signup(body: SignupBody, request: Request, _l: None = Depends(ratelimit.limit("signup"))) -> Dict[str, Any]:
        """Create a Free account (same rules as the web form) and sign in. A verification
        email is sent; verifying is needed to upgrade and for alert emails."""
        res = acct.signup(body.email, body.password, body.username, body.accept_terms)
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
        acct.change_password(account, body.current_password, body.new_password)
        return _token_pair(_user(account), _settings(request), None)

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
